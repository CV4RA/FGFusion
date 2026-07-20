"""
Generate CLIP text prototypes for FUME V6+ CLIP distillation.

Output: datasets/{dataset_name}_clip_proto.npy  shape (num_classes, 512)

Requirements (install one):
  pip install git+https://github.com/openai/CLIP.git
  pip install open_clip_torch

Usage:
  python generate_clip_protos.py                    # all datasets
  python generate_clip_protos.py --dataset nus_deep # one dataset
"""

import argparse
import os
import numpy as np
from class_names import get_class_names, DATASET_NUM_CLASSES


def load_clip_model(device):
    try:
        import clip
        model, _ = clip.load('ViT-B/32', device=device)
        tokenize = clip.tokenize
        return model, tokenize, 'openai-clip'
    except ImportError:
        pass

    try:
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='openai')
        import torch
        model = model.to(device)
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        def tokenize(texts):
            return tokenizer(texts)
        return model, tokenize, 'open-clip'
    except ImportError:
        pass

    raise ImportError(
        'CLIP not installed. Run:\n'
        '  pip install git+https://github.com/openai/CLIP.git\n'
        'or:\n'
        '  pip install open_clip_torch'
    )


def encode_text_prototypes(model, tokenize, class_names, device, backend):
    import torch

    templates = [
        'a photo of a {}.',
        'a photo of {}.',
        'an image of a {}.',
    ]

    all_embeds = []
    model.eval()
    with torch.no_grad():
        for name in class_names:
            prompts = [t.format(name) for t in templates]
            tokens = tokenize(prompts).to(device)
            if backend == 'openai-clip':
                feats = model.encode_text(tokens)
            else:
                feats = model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            proto = feats.mean(dim=0)
            proto = proto / proto.norm()
            all_embeds.append(proto.cpu().numpy())

    return np.stack(all_embeds, axis=0).astype(np.float32)


def generate_for_dataset(dataset_name, device):
    out_path = os.path.join('datasets', '{}_clip_proto.npy'.format(dataset_name))
    names = get_class_names(dataset_name)
    n_expected = DATASET_NUM_CLASSES[dataset_name]

    if len(names) != n_expected:
        print('  WARNING: {} names for {}, expected {}'.format(
            len(names), dataset_name, n_expected))
        if len(names) > n_expected:
            names = names[:n_expected]
        else:
            names = names + ['object category {}'.format(i)
                             for i in range(len(names), n_expected)]

    print('Generating {} ({} classes)...'.format(out_path, len(names)))
    print('  Sample names:', names[:3], '...')

    model, tokenize, backend = load_clip_model(device)
    protos = encode_text_prototypes(model, tokenize, names, device, backend)

    os.makedirs('datasets', exist_ok=True)
    np.save(out_path, protos)
    print('  Saved shape {} -> {}'.format(protos.shape, out_path))
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='all',
                        help='dataset name or "all"')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    import torch
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print('CUDA not available, using CPU.')

    datasets = list(DATASET_NUM_CLASSES.keys()) if args.dataset == 'all' else [args.dataset]

    for ds in datasets:
        if ds not in DATASET_NUM_CLASSES:
            print('Unknown dataset:', ds)
            continue
        try:
            generate_for_dataset(ds, device)
        except Exception as e:
            print('FAILED on {}: {}'.format(ds, e))


if __name__ == '__main__':
    main()
