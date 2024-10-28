import os.path
from typing import Union, List, Tuple, Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter

data_root = 'dataset'


def read_ECG_data(record_id: int) -> pd.DataFrame:
    """
    读取单个记录的心电信号
    :param record_id:
    :return:
    """
    file_path = f'{data_root}/a{record_id:02}.csv'
    data = pd.read_csv(file_path)
    data = data.iloc[1:, :]
    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.interpolate(method='linear')
    data = data.astype('float')
    data.columns = ['TimeStrap', 'AECG1', 'AECG2', 'AECG3', 'AECG4']
    return data


def read_ECG_label(record_id: int) -> set:
    """
    读取事件标签
    :param record_id:
    :return:
    """
    file_path = f'{data_root}/a{record_id:02}.fqrs.txt'
    with open(file_path) as f:
        lines = f.readlines()
        ll = []
        for line in lines:
            ll.append(int(line.strip()))
        ll = set(ll)
    return ll


def smooth_ECG(data: pd.DataFrame, smooth_window_size: int = 10) -> pd.DataFrame:
    """
    获得平滑的心电信号进行绘图使用
    :param data:
    :param smooth_window_size:设置平滑窗口，默认20
    :return:
    """
    for i in range(4):
        data[f'AECG{i + 1}'] = np.convolve(data[f'AECG{i + 1}'].astype(float),
                                           np.ones(smooth_window_size) / smooth_window_size,
                                           mode='same')
    return data


def slice_ECG(data: pd.DataFrame, labels: set, window_size: int = 100, step: int = 1) -> Tuple[List[pd.DataFrame], List[int]]:
    """
    使用滑动窗口切片数据集
    :param data:
    :param labels:
    :param window_size:
    :param step:
    :return:
    """
    tmp = data.copy()
    tmp.insert(0, "index", np.array([str(_) for _ in range(60000)]))
    # print(tmp.head())
    res_x = []
    res_y = []
    for i in range(0, 60000 - window_size + 1, step):
        t = tmp.iloc[i:i + window_size, 2:]
        ss = set(tmp.iloc[i:i + window_size, 0].astype(int))
        if ss & labels:
            label = 1
        else:
            label = 0
        # print(ss)
        # print(t.head())
        res_x.append(t)
        res_y.append(label)
        # print(t.shape)
    # print(len(res))
    return res_x, res_y


def plot_ECG_per_record(data: pd.DataFrame, smooth: bool = False, smooth_window_size: int = 10):
    """
    绘制单个记录的心电信号
    :param smooth_window_size:
    :param smooth:
    :param data:
    :return:
    """
    if smooth:
        data = smooth_ECG(data, smooth_window_size)
    fig, axes = plt.subplots(4, 1, figsize=(12, 6))
    # formatter = ScalarFormatter(useOffset=False)
    # formatter.set_scientific(True)
    # 绘制每个通道
    for i in range(4):
        axes[i].plot(data['TimeStrap'], data[f'AECG{i + 1}'])
        axes[i].set_title(f'Channel {i + 1}')
        axes[i].set_xlabel('Time (seconds)')
        axes[i].set_ylabel('Amplitude (uV)')

        axes[i].xaxis.set_major_locator(plt.MaxNLocator(5))  # x轴主要刻度，最多显示5个
        axes[i].yaxis.set_major_locator(plt.MaxNLocator(5))

        # axes[i].xaxis.set_major_formatter(formatter)
        # axes[i].yaxis.set_major_formatter(formatter)
        # axes[i].legend()

    # 调整子图之间的间距
    plt.tight_layout()

    # 显示图表
    plt.show()
