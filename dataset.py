import numpy as np
import pandas as pd
import torch as t
from sklearn.preprocessing import OneHotEncoder

from torch.utils.data import Dataset
from tqdm.auto import tqdm

from utils import *


class ECGDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = None if y is None else t.LongTensor(y)

    def __getitem__(self, index):
        if self.y is None:
            return self.x[index]
        else:
            return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


def create_dataset() -> Tuple[ECGDataset, ECGDataset, ECGDataset]:
    """
    获取Dataset
    :return:
    """
    excluded = {33, 38, 47, 52, 54, 71, 74}
    valid = set(range(8, 14))
    test = set(range(1, 8))
    train = set(range(1, 76)) - excluded - valid - test
    root_path = 'dataset_preprocessed/'
    if not os.path.exists(root_path):
        os.mkdir(root_path)
    if not (os.path.exists(root_path + 'train_set_feature.npy') and os.path.exists(root_path + 'train_set_label.npy')):
        print("开始初始化训练集")
        train_set_feature = []
        train_set_label = []
        for i in tqdm(train):
            feature, label = slice_ECG(read_ECG_data(i), read_ECG_label(i))
            train_set_feature.extend(feature)
            train_set_label.extend(label)
        train_set_feature = np.array(train_set_feature)
        train_set_label = np.array(train_set_label)
        np.save('dataset_preprocessed/train_set_feature.npy', train_set_feature)
        np.save('dataset_preprocessed/train_set_label.npy', train_set_label)
        print("训练集初始化完毕")
        del train_set_feature, train_set_label

    if not (os.path.exists(root_path + 'valid_set_feature.npy') and os.path.exists(root_path + 'valid_set_label.npy')):
        print("开始初始化验证集")
        valid_set_feature = []
        valid_set_label = []
        for i in tqdm(valid):
            feature, label = slice_ECG(read_ECG_data(i), read_ECG_label(i))
            valid_set_feature.extend(feature)
            valid_set_label.extend(label)
        valid_set_feature = np.array(valid_set_feature)
        valid_set_label = np.array(valid_set_label)
        np.save('dataset_preprocessed/valid_set_feature.npy', valid_set_feature)
        np.save('dataset_preprocessed/valid_set_label.npy', valid_set_label)
        print("验证集初始化完毕")
        del valid_set_feature, valid_set_label

    if not (os.path.exists(root_path + 'test_set_feature.npy') and os.path.exists(root_path + 'test_set_label.npy')):
        print("开始初始化测试集")
        test_set_feature = []
        test_set_label = []
        for i in tqdm(test):
            feature, label = slice_ECG(read_ECG_data(i), read_ECG_label(i))
            test_set_feature.extend(feature)
            test_set_label.extend(label)
        test_set_feature = np.array(test_set_feature)
        test_set_label = np.array(test_set_label)
        np.save('dataset_preprocessed/test_set_feature.npy', test_set_feature)
        np.save('dataset_preprocessed'
                '/test_set_label.npy', test_set_label)
        print("测试集初始化完毕")
        del test_set_feature, test_set_label

    enc = OneHotEncoder(sparse=False)
    train_set_feature = np.load('dataset_preprocessed/train_set_feature.npy').astype(np.float32).transpose(0, 2, 1)
    train_set_label = np.load('dataset_preprocessed/train_set_label.npy').astype(int)
    print("训练集shape:{}", train_set_feature.shape)
    valid_set_feature = np.load('dataset_preprocessed/valid_set_feature.npy').astype(np.float32).transpose(0, 2, 1)
    valid_set_label = np.load('dataset_preprocessed/train_set_label.npy').astype(int)
    print("验证集shape:{}", valid_set_feature.shape)
    test_set_feature = np.load('dataset_preprocessed/test_set_feature.npy').astype(np.float32).transpose(0, 2, 1)
    test_set_label = np.load('dataset_preprocessed/test_set_label.npy').astype(int)
    print("测试集shape:{}", test_set_feature.shape)

    print(train_set_label.shape)
    print(valid_set_label.shape)
    print(test_set_label.shape)

    train_set = ECGDataset(train_set_feature, train_set_label)
    valid_set = ECGDataset(valid_set_feature, valid_set_label)
    test_set = ECGDataset(test_set_feature, None)
    print("Dataset初始化完毕")
    return train_set, valid_set, test_set
