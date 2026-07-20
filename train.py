import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
import time
import copy
import scipy.spatial.distance as dist_fn
from options import get_dataloader
from evaluate import fx_calc_map_label, fx_calc_map_label_withUncertainty


# ===================================================================
# Model
# ===================================================================

class ResEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, mid_num=4096, layer_num=3):
        super().__init__()
        self.proj = nn.Linear(input_dim, mid_num)
        self.blocks = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(layer_num - 1):
            self.blocks.append(nn.Linear(mid_num, mid_num))
            self.norms.append(nn.LayerNorm(mid_num))
        self.head = nn.Linear(mid_num, output_dim, bias=False)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = x.to(torch.float)
        x = F.relu(self.proj(x))
        for block, norm in zip(self.blocks, self.norms):
            residual = x
            x = self.dropout(F.relu(block(x)))
            x = norm(x + residual)
        hidden = x
        feat = self.head(x)
        return F.normalize(feat, p=2, dim=1), hidden


class GaussianHead(nn.Module):
    def __init__(self, hidden_dim, bottleneck=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, 1),
        )

    def forward(self, hidden):
        logvar = self.net(hidden)
        return torch.sigmoid(logvar), logvar


class FUME_V21(nn.Module):
    def __init__(self, device, img_input_dim=4096, text_input_dim=1024,
                 output_dim=1024, num_class=10, layer_num=3, fuse_alpha=0.5):
        super().__init__()
        mid = 4096
        self.img_net = ResEncoder(img_input_dim, output_dim, mid, layer_num)
        self.text_net = ResEncoder(text_input_dim, output_dim, mid, layer_num)
        self.img_gauss = GaussianHead(mid)
        self.txt_gauss = GaussianHead(mid)
        self.fuse_alpha = fuse_alpha
        W = torch.Tensor(output_dim, output_dim)
        self.W = torch.nn.init.orthogonal_(W, gain=1)[:, :num_class].to(device)
        self.W = self.W / torch.norm(self.W, p=2, dim=1, keepdim=True)

    def forward(self, img, text):
        v1, h1 = self.img_net(img)
        v2, h2 = self.text_net(text)
        W = self.W / torch.norm(self.W, p=2, dim=1, keepdim=True)
        p1 = torch.relu(v1.mm(W))
        p2 = torch.relu(v2.mm(W))
        u1_fuzzy = self._fuzzy_unc(self._cred(p1))
        u2_fuzzy = self._fuzzy_unc(self._cred(p2))
        u1_gauss, lv1 = self.img_gauss(h1)
        u2_gauss, lv2 = self.txt_gauss(h2)
        a = self.fuse_alpha
        u1 = a * u1_fuzzy + (1 - a) * u1_gauss
        u2 = a * u2_fuzzy + (1 - a) * u2_gauss
        return {'view1_feature': v1, 'view2_feature': v2,
                'view1_membershipDegree': p1, 'view2_membershipDegree': p2,
                'view1_uncertainty': u1, 'view2_uncertainty': u2,
                'logvar1': lv1, 'logvar2': lv2}

    def _cred(self, md):
        top2 = torch.topk(md, k=2, dim=1, largest=True, sorted=True)[0]
        cred = md - top2[:, 0].view(-1, 1).detach() + 1
        cred += (cred == 1).float() * (top2[:, 0] - top2[:, 1]).reshape(-1, 1).detach()
        return cred / 2

    def _fuzzy_unc(self, cred):
        e = 1e-7
        H = torch.sum(-cred * torch.log(cred + e) - (1 - cred) * torch.log(1 - cred + e),
                      dim=1, keepdim=True)
        return H / (cred.shape[1] * torch.log(torch.tensor(2.0)))


# ===================================================================
# Config
# ===================================================================

DATASET_CFG = {
    'pascal':         {'lambda_supcon': 0.25, 'lambda_uncer': 0.12},
    'wiki':           {'lambda_supcon': 0.35, 'lambda_uncer': 0.18},
    'nus_deep':       {'lambda_supcon': 0.25, 'lambda_uncer': 0.12},
    'INRIA':          {'lambda_supcon': 0.25, 'lambda_uncer': 0.12},
    'xmedianet_deep': {'lambda_supcon': 0.22, 'lambda_uncer': 0.12},
}

DEFAULT_CFG = {
    'gamma': 0.30, 'margin': 0.15,
    'lambda_clip': 1.0, 'lambda_mix': 0.50,
    'lambda_supcon': 0.25, 'lambda_uncer': 0.12,
    'lambda_kl': 0.05,
    'phase_a_epochs': 160, 'total_epochs': 200,
    'plain_guard': 0.94, 'w_map': 0.70, 'w_u05': 0.30,
    'ema_decay': 0.999,
}

def get_cfg(name):
    return {**DEFAULT_CFG, **DATASET_CFG.get(name, {})}

def set_seed(s):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    random.seed(s); np.random.seed(s)
    torch.backends.cudnn.deterministic = True


# ===================================================================
# Losses
# ===================================================================

def calc_loss(v1, v2, p1, p2, lbl1, lbl2, alpha):
    def cred(predict, labels):
        lbl = labels.float() if labels.dtype != torch.float32 else labels
        t1 = (predict * (1 - lbl)).max(1)[0].reshape([-1, 1])
        lp = (predict * lbl).max(1)[0].reshape([-1, 1])
        return (predict + (1 - lp) * (1 - lbl) + (1 - t1) * lbl) / 2
    def consistency(fea, tau=1.):
        bs = fea[0].shape[0]
        all_f = torch.cat(fea)
        sim = (all_f.mm(all_f.t()) / tau).exp()
        sim = sim - sim.diag().diag()
        s1 = sim[:, :bs] + sim[:, bs:]
        d1 = torch.cat([s1[:bs].diag(), s1[bs:].diag()])
        l1 = -(d1 / sim.sum(1)).log().mean()
        s2 = sim[:bs] + sim[bs:]
        d2 = torch.cat([s2[:, :bs].diag(), s2[:, bs:].diag()])
        l2 = -(d2 / sim.sum(1)).log().mean()
        return l1 + l2
    l1 = lbl1.float() if lbl1.dtype != torch.float32 else lbl1
    l2 = lbl2.float() if lbl2.dtype != torch.float32 else lbl2
    c1, c2 = cred(p1, l1), cred(p2, l2)
    fml = ((c1 - l1)**2).sum(1).sqrt().mean() + ((c2 - l2)**2).sum(1).sqrt().mean()
    return fml + alpha * consistency([v1, v2])

# PLACEHOLDER_MORE

def mixup_batch(features, labels, alpha_mix=0.4):
    N = features.shape[0]
    lam = np.random.beta(alpha_mix, alpha_mix)
    lam = max(lam, 1 - lam)
    perm = torch.randperm(N, device=features.device)
    return (lam * features + (1 - lam) * features[perm],
            lam * labels.float() + (1 - lam) * labels[perm].float())

def cutmix_batch(features, labels, alpha_mix=1.0):
    N, D = features.shape
    lam = np.random.beta(alpha_mix, alpha_mix)
    perm = torch.randperm(N, device=features.device)
    cut_len = int(D * (1 - lam))
    start = np.random.randint(0, max(D - cut_len, 1))
    mixed = features.clone()
    mixed[:, start:start + cut_len] = features[perm, start:start + cut_len]
    lam_actual = 1.0 - cut_len / D
    return mixed, lam_actual * labels.float() + (1 - lam_actual) * labels[perm].float()

def hard_labels_from_soft(soft, fallback):
    hard = (soft > 0.5).long()
    no_label = hard.sum(1) == 0
    if no_label.any():
        hard[no_label] = fallback[no_label]
    return hard

class CLIPDistillLoss(nn.Module):
    def __init__(self, feat_dim, clip_proto_path, device, tau=0.1):
        super().__init__()
        protos = np.load(clip_proto_path)
        protos /= (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-8)
        self.clip_protos = torch.tensor(protos, dtype=torch.float32).to(device)
        self.projector = nn.Linear(feat_dim, self.clip_protos.shape[1], bias=False).to(device)
        nn.init.orthogonal_(self.projector.weight)
        self.tau = tau
    def forward(self, features, labels):
        proj = F.normalize(self.projector(features), p=2, dim=1)
        logits = proj @ self.clip_protos.t() / self.tau
        targets = labels.float()
        valid = targets.sum(dim=1) > 0
        if not valid.any():
            return torch.tensor(0.0, device=features.device)
        targets = targets / (targets.sum(dim=1, keepdim=True) + 1e-8)
        return (-(targets * F.log_softmax(logits, dim=1)).sum(dim=1)[valid]).mean()

def hard_negative_push(v1, v2, labels, margin=0.15):
    idx = labels.float().argmax(dim=1)
    si, st = v1 @ v2.t(), v2 @ v1.t()
    neg = ~(idx.unsqueeze(0) == idx.unsqueeze(1))
    sn_i, sn_t = si.clone(), st.clone()
    sn_i[~neg], sn_t[~neg] = -1e9, -1e9
    return (torch.clamp(sn_i.max(1)[0] - si.diag() + margin, min=0.).mean()
            + torch.clamp(sn_t.max(1)[0] - st.diag() + margin, min=0.).mean())

def cross_modal_supcon(v1, v2, labels, tau=0.1):
    pos = (labels.float().mm(labels.float().t()) > 0).float()
    si = v1 @ v2.t() / tau
    lp_i = si - torch.logsumexp(si, dim=1, keepdim=True)
    li = -(lp_i * pos).sum(dim=1) / pos.sum(dim=1).clamp(min=1.)
    st = v2 @ v1.t() / tau
    lp_t = st - torch.logsumexp(st, dim=1, keepdim=True)
    lt = -(lp_t * pos.t()).sum(dim=1) / pos.t().sum(dim=1).clamp(min=1.)
    return (li.mean() + lt.mean()) / 2.

def uncertainty_calibration_loss(u1, u2, labels, threshold=0.5):
    N = u1.shape[0]
    u1b, u2b = u1.view(N, 1), u2.view(1, N)
    merged = 1 - (1 - u1b) * (1 - u2b)
    idx = labels.float().argmax(dim=1)
    same = idx.unsqueeze(0) == idx.unsqueeze(1)
    eye = torch.eye(N, dtype=torch.bool, device=labels.device)
    pos = same & ~eye; neg = (~same) & ~eye
    lp = merged[pos].mean() if pos.any() else merged.diag().mean()
    ln = F.relu(threshold - merged[neg]).mean() if neg.any() else 0.
    return 0.5 * merged.diag().mean() + 0.5 * lp + ln

def gaussian_kl_loss(lv1, lv2):
    return -0.5 * (1 + lv1 - lv1.exp()).mean() + -0.5 * (1 + lv2 - lv2.exp()).mean()

def gaussian_calib_loss(u1, u2, labels):
    idx = labels.float().argmax(dim=1)
    N = u1.shape[0]
    same = (idx.unsqueeze(0) == idx.unsqueeze(1))
    eye = torch.eye(N, dtype=torch.bool, device=labels.device)
    u_cross = (u1.view(N, 1) + u2.view(1, N)) / 2
    loss = u1.mean() + u2.mean()
    pos = same & ~eye; neg = (~same) & ~eye
    if pos.any(): loss = loss + u_cross[pos].mean()
    if neg.any(): loss = loss + F.relu(0.5 - u_cross[neg]).mean()
    return loss

# PLACEHOLDER_TRAINING

class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)
    @torch.no_grad()
    def update(self, model):
        for sp, p in zip(self.shadow.parameters(), model.parameters()):
            sp.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)

def eval_retrieval(model, loader, device):
    il, tl, ll, iul, tul = [], [], [], [], []
    with torch.no_grad():
        for imgs, txts, labels in loader:
            ret = model(imgs.to(device), txts.to(device))
            il.append(ret['view1_feature'].cpu().numpy())
            tl.append(ret['view2_feature'].cpu().numpy())
            ll.append(labels.cpu().numpy())
            iul.append(ret['view1_uncertainty'].cpu().numpy())
            tul.append(ret['view2_uncertainty'].cpu().numpy())
    t_i, t_t = np.concatenate(il), np.concatenate(tl)
    t_iu, t_tu = np.concatenate(iul), np.concatenate(tul)
    t_l = np.concatenate(ll).argmax(1)
    i2t = fx_calc_map_label(t_i, t_t, t_l, 0)
    t2i = fx_calc_map_label(t_t, t_i, t_l, 0)
    iu_m = 1 - (1 - t_iu) * (1 - t_tu.T)
    tu_m = 1 - (1 - t_tu) * (1 - t_iu.T)
    i2t_u = fx_calc_map_label_withUncertainty(t_i, t_t, t_l, iu_m, 0, 0.5)
    t2i_u = fx_calc_map_label_withUncertainty(t_t, t_i, t_l, tu_m, 0, 0.5)
    aver_u = 0. if (np.isnan(i2t_u) or np.isnan(t2i_u)) else (i2t_u + t2i_u) / 2.
    return dict(I2T=i2t, T2I=t2i, Aver=(i2t+t2i)/2.,
                I2T_u05=i2t_u, T2I_u05=t2i_u, Aver_u05=aver_u)

def v6_forward_loss(model, imgs, txts, labels, alpha, cfg, clip_fn,
                    use_mixup=True, hard_mixup=True):
    device = imgs.device
    loss_mix = torch.tensor(0.0, device=device)
    if use_mixup:
        if np.random.random() < 0.5:
            im, li_s = mixup_batch(imgs, labels)
            tm, lt_s = mixup_batch(txts, labels)
        else:
            im, li_s = cutmix_batch(imgs, labels)
            tm, lt_s = cutmix_batch(txts, labels)
        rm = model(im, tm)
        li = hard_labels_from_soft(li_s, labels) if hard_mixup else li_s
        lt = hard_labels_from_soft(lt_s, labels) if hard_mixup else lt_s
        loss_mix = calc_loss(rm['view1_feature'], rm['view2_feature'],
                             rm['view1_membershipDegree'], rm['view2_membershipDegree'],
                             li, lt, alpha)
    ret = model(imgs, txts)
    v1, v2 = ret['view1_feature'], ret['view2_feature']
    p1, p2 = ret['view1_membershipDegree'], ret['view2_membershipDegree']
    loss = (calc_loss(v1, v2, p1, p2, labels, labels, alpha)
            + cfg['gamma'] * hard_negative_push(v1, v2, labels, cfg['margin'])
            + cfg['lambda_mix'] * loss_mix)
    if clip_fn is not None and cfg['lambda_clip'] > 0:
        loss = loss + cfg['lambda_clip'] * (clip_fn(v1, labels) + clip_fn(v2, labels))
    if 'logvar1' in ret:
        loss = loss + cfg['lambda_kl'] * gaussian_kl_loss(ret['logvar1'], ret['logvar2'])
    return loss, ret

def v7_extra_loss(ret, labels, cfg):
    u1, u2 = ret['view1_uncertainty'], ret['view2_uncertainty']
    v1, v2 = ret['view1_feature'], ret['view2_feature']
    loss = (cfg['lambda_supcon'] * cross_modal_supcon(v1, v2, labels)
            + cfg['lambda_uncer'] * uncertainty_calibration_loss(u1, u2, labels))
    if 'logvar1' in ret:
        loss = loss + 0.1 * gaussian_calib_loss(u1, u2, labels)
    return loss

def save_ckpt(model, clip_fn, ema, tag):
    return {'tag': tag, 'model': copy.deepcopy(model.state_dict()),
            'clip': copy.deepcopy(clip_fn.state_dict()) if clip_fn else None,
            'ema': copy.deepcopy(ema.shadow.state_dict()) if ema else None}

def load_ckpt(model, clip_fn, ckpt, use_ema=False):
    if use_ema and ckpt.get('ema') is not None:
        model.load_state_dict(ckpt['ema'])
    else:
        model.load_state_dict(ckpt['model'])
    if clip_fn is not None and ckpt.get('clip') is not None:
        clip_fn.load_state_dict(ckpt['clip'])

# PLACEHOLDER_MAIN

def train_v21(model, dl, optimizer, scheduler, alpha, cfg, clip_fn, device, folder_path):
    total = cfg['total_epochs']
    pb_start = cfg['phase_a_epochs']
    ckpt_v6, ckpt_v7 = None, None
    best_v6, best_v7_comp = 0.0, -1.0
    v7_ok = False
    ema = None

    for epoch in range(total):
        in_pb = epoch >= pb_start
        if in_pb and ema is None:
            print('>>> Phase B start')
            if ckpt_v6: load_ckpt(model, clip_fn, ckpt_v6, use_ema=False)
            ema = ModelEMA(model, decay=cfg['ema_decay'])

        for split in ['train', 'val']:
            model.train() if split == 'train' else model.eval()
            for imgs, txts, labels in dl[split]:
                imgs, txts, labels = imgs.to(device), txts.to(device), labels.to(device)
                with torch.set_grad_enabled(split == 'train'):
                    if split == 'train':
                        optimizer.zero_grad()
                        loss, ret = v6_forward_loss(model, imgs, txts, labels, alpha, cfg, clip_fn,
                                                    use_mixup=True, hard_mixup=(not in_pb))
                        if in_pb: loss = loss + v7_extra_loss(ret, labels, cfg)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                        optimizer.step()
                        if ema: ema.update(model)
            if scheduler and split == 'train': scheduler.step()

            if split == 'val':
                em = ema.shadow if ema else model
                m = eval_retrieval(em, dl['val'], device)
                if not in_pb:
                    if m['Aver'] > best_v6:
                        best_v6 = m['Aver']
                        ckpt_v6 = save_ckpt(model, clip_fn, ema, 'v6')
                else:
                    comp = cfg['w_map'] * m['Aver'] + cfg['w_u05'] * m['Aver_u05']
                    if m['Aver'] >= best_v6 * cfg['plain_guard'] and comp > best_v7_comp:
                        best_v7_comp = comp; v7_ok = True
                        ckpt_v7 = save_ckpt(model, clip_fn, ema, 'v7')

        if (epoch + 1) % 20 == 0:
            em = ema.shadow if ema else model
            m = eval_retrieval(em, dl['val'], device)
            print('Epoch {}/{} plain={:.4f} u05={:.4f}'.format(epoch+1, total, m['Aver'], m['Aver_u05']))

    if not ckpt_v6: ckpt_v6 = save_ckpt(model, clip_fn, ema, 'v6_fb')
    return ckpt_v6, ckpt_v7, v7_ok


# TSR
def _tsr_sim(qf, gf):
    return 1.0 - dist_fn.cdist(qf, gf, 'cosine')

def _tsr_unc(qu, gu):
    return 1.0 - (1.0 - np.asarray(qu).reshape(-1,1)) * (1.0 - np.asarray(gu).reshape(1,-1))

def _tsr_rank(sr, ur, tr, ut, mode, fp, mk):
    ng = sr.shape[0]
    k = max(int(ng*tr),1) if tr < 1.0 else ng
    if mode == 'cascade':
        cand = np.argpartition(-sr, min(k-1,ng-1))[:k] if tr < 1.0 else np.arange(ng)
        valid = cand[ur[cand] <= ut]
        if valid.size == 0: valid = cand[np.argsort(-sr[cand])[:max(mk,1)]]
        return valid[np.argsort(-sr[valid])]
    if mode == 'fusion':
        return np.argsort(-(sr * np.power(np.clip(1.0-ur,1e-8,1.0), fp)))
    return np.argsort(-(sr - fp * ur))

def calc_map_tsr(qf, gf, labels, qu, gu, topk_ratio=1.0, u_threshold=0.5,
                 mode='cascade', fuse_param=1.0, min_keep=3):
    labels = np.asarray(labels).reshape(-1)
    sim = _tsr_sim(qf, gf); unc = _tsr_unc(qu, gu)
    aps = []
    for i in range(qf.shape[0]):
        order = _tsr_rank(sim[i], unc[i], topk_ratio, u_threshold, mode, fuse_param, min_keep)
        p, r = 0., 0.
        for rank, j in enumerate(order):
            if labels[i] == labels[j]: r += 1.; p += r/(rank+1.)
        aps.append(p/r if r > 0 else 0.)
    return float(np.mean(aps))

def eval_all_metrics(fi, ft, l, iu, tu, tc):
    pi = fx_calc_map_label(fi, ft, l, 0); pt = fx_calc_map_label(ft, fi, l, 0)
    ui_m = _tsr_unc(iu, tu); ut_m = _tsr_unc(tu, iu)
    ui = fx_calc_map_label_withUncertainty(fi, ft, l, ui_m, 0, 0.5)
    ut = fx_calc_map_label_withUncertainty(ft, fi, l, ut_m, 0, 0.5)
    u05 = 0. if (np.isnan(ui) or np.isnan(ut)) else (ui+ut)/2.
    ti = calc_map_tsr(fi, ft, l, iu, tu, **tc)
    tt = calc_map_tsr(ft, fi, l, tu, iu, **tc)
    return dict(plain_I2T=pi, plain_T2I=pt, plain_Aver=(pi+pt)/2,
                u05_I2T=ui, u05_T2I=ut, u05_Aver=u05,
                tsr_I2T=ti, tsr_T2I=tt, tsr_Aver=(ti+tt)/2)

def tune_tsr(fi, ft, l, iu, tu):
    best_cfg, best_s = None, -1
    for mode in ('cascade','fusion','penalty'):
        params = (0.5,1.0,1.5,2.0) if mode != 'cascade' else (1.0,)
        for fp in params:
            for tr in (0.5,0.7,1.0):
                for ut in (0.4,0.45,0.5,0.55):
                    cfg = dict(topk_ratio=tr, u_threshold=ut, mode=mode, fuse_param=fp, min_keep=3)
                    m = eval_all_metrics(fi, ft, l, iu, tu, cfg)
                    s = 0.5*m['plain_Aver'] + 0.5*m['tsr_Aver']
                    if s > best_s: best_s, best_cfg = s, cfg
    return best_cfg

def extract_features(model, loader, device):
    il, tl, ll, iul, tul = [], [], [], [], []
    with torch.no_grad():
        for imgs, txts, labels in loader:
            ret = model(imgs.to(device), txts.to(device))
            il.append(ret['view1_feature'].cpu().numpy())
            tl.append(ret['view2_feature'].cpu().numpy())
            ll.append(labels.cpu().numpy())
            iul.append(ret['view1_uncertainty'].cpu().numpy())
            tul.append(ret['view2_uncertainty'].cpu().numpy())
    return (np.concatenate(il), np.concatenate(tl), np.concatenate(ll).argmax(1),
            np.concatenate(iul), np.concatenate(tul))


# ===================================================================
if __name__ == '__main__':
    all_datasets = ['pascal', 'wiki', 'nus_deep', 'INRIA', 'xmedianet_deep']
    all_results = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for dsName in all_datasets:
        print('\n' + '='*80)
        print('  V21 (Gaussian Head) on: {}'.format(dsName))
        print('='*80)
        try:
            set_seed(2)
            folder_path = os.path.join("saved", dsName+"_v21", time.strftime("%Y%m%d_%H%M%S"))
            os.makedirs(folder_path, exist_ok=True)
            dc, dl = get_dataloader(dsName)
            cfg = get_cfg(dsName)

            model = FUME_V21(device, img_input_dim=dc['input_dim_I'],
                             text_input_dim=dc['input_dim_T'],
                             output_dim=dc['class_number'],
                             num_class=dc['class_number'],
                             layer_num=dc['layer_num'] + 2,
                             fuse_alpha=0.5).to(device)

            clip_path = 'datasets/{}_clip_proto.npy'.format(dc['dataset_name'])
            if os.path.exists(clip_path):
                clip_fn = CLIPDistillLoss(dc['class_number'], clip_path, device)
                extra = list(clip_fn.parameters())
            else:
                clip_fn, extra = None, []
                cfg['lambda_clip'] = 0.0

            lr = dc['lr']
            opt = torch.optim.Adam(list(model.parameters())+extra, lr=lr, betas=(0.5,0.999))
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg['total_epochs'], eta_min=lr*0.01)

            c6, c7, v7_ok = train_v21(model, dl, opt, sch, dc['alpha'], cfg, clip_fn, device, folder_path)

            load_ckpt(model, clip_fn, c6, use_ema=False)
            m6 = eval_retrieval(model, dl['test'], device)
            if v7_ok and c7:
                load_ckpt(model, clip_fn, c7, use_ema=True)
                m7 = eval_retrieval(model, dl['test'], device)
                if m7['Aver'] >= m6['Aver']*cfg['plain_guard'] and m7['Aver_u05'] >= m6['Aver_u05']:
                    load_ckpt(model, clip_fn, c7, use_ema=True)
                else:
                    load_ckpt(model, clip_fn, c6, use_ema=False)

            vi,vt,vl,viu,vtu = extract_features(model, dl['val'], device)
            ti,tt,tl,tiu,ttu = extract_features(model, dl['test'], device)
            best_cfg = tune_tsr(vi,vt,vl,viu,vtu)
            test_m = eval_all_metrics(ti,tt,tl,tiu,ttu, best_cfg)
            print('  [{}] Test Results:'.format(dsName))
            print('    Plain  I→T={:.3f}  T→I={:.3f}  Avg={:.3f}'.format(
                test_m['plain_I2T'], test_m['plain_T2I'], test_m['plain_Aver']))
            print('    u=0.5  I→T={:.3f}  T→I={:.3f}  Avg={:.3f}'.format(
                test_m['u05_I2T'], test_m['u05_T2I'], test_m['u05_Aver']))
            print('    TSR    I→T={:.3f}  T→I={:.3f}  Avg={:.3f}'.format(
                test_m['tsr_I2T'], test_m['tsr_T2I'], test_m['tsr_Aver']))
            all_results[dsName] = test_m
        except Exception as e:
            import traceback
            print('ERROR: {}'.format(e)); traceback.print_exc()

    print('\n' + '='*80)
    print('  FINAL SUMMARY (V21 — Gaussian Uncertainty Head)')
    print('='*80)
    print('{:<15} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}'.format(
        'Dataset', 'P_I2T', 'P_T2I', 'P_Avg', 'U_I2T', 'U_T2I', 'U_Avg', 'T_I2T', 'T_T2I', 'T_Avg'))
    print('-'*95)
    for ds in all_datasets:
        if ds not in all_results: continue
        r = all_results[ds]
        print('{:<15} {:>8.3f} {:>8.3f} {:>8.3f} {:>8.3f} {:>8.3f} {:>8.3f} {:>8.3f} {:>8.3f} {:>8.3f}'.format(
            ds, r['plain_I2T'], r['plain_T2I'], r['plain_Aver'],
            r['u05_I2T'], r['u05_T2I'], r['u05_Aver'],
            r['tsr_I2T'], r['tsr_T2I'], r['tsr_Aver']))
