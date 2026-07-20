from torch.utils.data.dataset import Dataset
import torch
import scipy.io as sio
import h5py
import numpy as np


def ind2vec(ind, N=None):
    ind = np.asarray(ind)
    if N is None:
        N = ind.max() + 1
    return np.arange(N) == np.repeat(ind, N, axis=1)


class MyCustomDataset(Dataset):
    def __init__(self, dataset='wiki', state='train'):

        if dataset == 'pascal':
            all_data = sio.loadmat("datasets/pascal.mat")
            img_train = all_data['img_train']
            text_train = all_data['text_train']
            I_test = all_data['img_test']
            T_test = all_data['text_test']
            label_train_img = all_data['label_train']
            label_test_img = all_data['label_test']

            indices_train = np.arange(len(img_train))
            np.random.shuffle(indices_train)

            I_train = img_train[indices_train]
            T_train = text_train[indices_train]
            labels_train = ind2vec(label_train_img[indices_train]).astype(int)
            labels_test = ind2vec(label_test_img).astype(int)
            I_val = I_test
            T_val = T_test
            labels_val = labels_test

        elif dataset == 'wiki':
            all_data = sio.loadmat("datasets/wiki.mat")
            img_train = all_data['train_imgs_deep']
            text_train = all_data['train_texts_doc']
            I_test = all_data['test_imgs_deep'][231:, :]
            T_test = all_data['test_texts_doc'][231:, :]
            I_val = all_data['test_imgs_deep'][0:231, :]
            T_val = all_data['test_texts_doc'][0:231, :]
            label_train_img = all_data['train_imgs_labels'].squeeze()
            label_test_img = all_data['test_imgs_labels'].squeeze()[231:]
            label_val_img = all_data['test_imgs_labels'].squeeze()[0:231]

            indices_train = np.arange(len(img_train))
            np.random.shuffle(indices_train)

            I_train = img_train[indices_train]
            T_train = text_train[indices_train]
            labels_train = ind2vec(label_train_img[indices_train].reshape(-1, 1)).astype(int)
            labels_test = ind2vec(label_test_img.reshape(-1, 1)).astype(int)
            labels_val = ind2vec(label_val_img.reshape(-1, 1)).astype(int)

        elif dataset == 'xmedianet_deep':
            all_data = sio.loadmat('datasets/XMediaNet5View_Doc2Vec.mat')
            img_train = all_data['train'][0][0] + 0
            text_train = all_data['train'][0][1]
            I_test = all_data['test'][0][0] + 0
            T_test = all_data['test'][0][1]
            I_val = all_data['valid'][0][0] + 0
            T_val = all_data['valid'][0][1]

            indices_train = np.arange(len(img_train))
            np.random.shuffle(indices_train)
            indices_test = np.arange(len(I_test))

            I_train = img_train[indices_train]
            T_train = text_train[indices_train]
            labels_train = all_data['train_labels'][0][0][0][indices_train]
            labels_test = all_data['test_labels'][0][0][0][indices_test]
            labels_val = all_data['valid_labels'][0][0][0]

            labels_train = ind2vec(labels_train.reshape(-1, 1), 200).astype(int)
            labels_test = ind2vec(labels_test.reshape(-1, 1), 200).astype(int)
            labels_val = ind2vec(labels_val.reshape(-1, 1), 200).astype(int)

        elif dataset == 'nus_deep':
            with h5py.File('datasets/nus_wide-10k_deep_doc2vec-corr-ae.h5py', 'r') as file:
                img_train = file['train_imgs_deep'][:]
                text_train = file['train_texts'][:]
                I_val = file['valid_imgs_deep'][:]
                T_val = file['valid_texts'][:]
                I_test = file['test_imgs_deep'][:]
                T_test = file['test_texts'][:]

                indices_train = np.arange(len(img_train))
                np.random.shuffle(indices_train)

                I_train = img_train[indices_train]
                T_train = text_train[indices_train]

                label_train_img = file['train_imgs_labels'][:].reshape(-1, 1)
                label_val_img = file['valid_imgs_labels'][:].reshape(-1, 1)
                label_test_img = file['test_imgs_labels'][:].reshape(-1, 1)

                labels_train = ind2vec(label_train_img[indices_train]).astype(int)
                labels_test = ind2vec(label_test_img).astype(int)
                labels_val = ind2vec(label_val_img).astype(int)

        elif dataset == 'INRIA':
            all_data = sio.loadmat('datasets/INRIA-Websearch.mat')
            img_train = all_data['tr_img']
            I_test = all_data['te_img']
            text_train = all_data['tr_txt']
            T_test = all_data['te_txt']
            I_val = all_data['val_img']
            T_val = all_data['val_txt']
            labels_train = all_data['tr_img_lab']
            labels_test = all_data['te_img_lab']
            labels_val = all_data['val_img_lab']

            indices_train = np.arange(len(img_train))
            np.random.shuffle(indices_train)

            I_train = img_train[indices_train]
            T_train = text_train[indices_train]
            labels_train = ind2vec(labels_train[indices_train]).astype(int)
            labels_test = ind2vec(labels_test).astype(int)
            labels_val = ind2vec(labels_val).astype(int)

        if state == 'train':
            self.I, self.T, self.labels = I_train, T_train, labels_train
        elif state == 'test':
            self.I, self.T, self.labels = I_test, T_test, labels_test
        elif state == 'val':
            self.I, self.T, self.labels = I_val, T_val, labels_val

        self.I = torch.FloatTensor(self.I)
        self.T = torch.FloatTensor(self.T)
        self.labels = torch.LongTensor(self.labels)

    def __getitem__(self, index):
        return self.I[index], self.T[index], self.labels[index]

    def __len__(self):
        return len(self.I)
