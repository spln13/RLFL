import matplotlib.pyplot as plt
import numpy as np

import os
import torch


# 创建自定义的 Dataset
def read_data(dataset, idx, is_train=True, is_local_test=False):
    if is_local_test:
        test_data_dir = os.path.join('../dataset', dataset, 'local_test/')

        test_file = test_data_dir + str(idx) + '.npz'
        with open(test_file, 'rb') as f:
            test_data = np.load(f, allow_pickle=True)['data'].tolist()

        return test_data

    if is_train:
        train_data_dir = os.path.join('../dataset', dataset, 'train/')

        train_file = train_data_dir + str(idx) + '.npz'
        with open(train_file, 'rb') as f:
            train_data = np.load(f, allow_pickle=True)['data'].tolist()

        return train_data

    else:
        test_data_dir = os.path.join('../dataset', dataset, 'test/')

        test_file = test_data_dir + str(idx) + '.npz'
        with open(test_file, 'rb') as f:
            test_data = np.load(f, allow_pickle=True)['data'].tolist()

        return test_data


def read_client_data(dataset, idx, is_train=True, is_local_test=False):
    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train = torch.Tensor(train_data['x']).type(torch.float32)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train, is_local_test)
        X_test = torch.Tensor(test_data['x']).type(torch.float32)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data


def save_mask_picture_barchart(client_id, cfg_mask):
    # 假设 cfg_mask 是你的二维数组
    # 转换为numpy数组以便于绘图
    cfg_mask_cpu = [mask.cpu().numpy() for mask in cfg_mask]  # 确保每个tensor都被移至CPU

    plt.figure(figsize=(10, len(cfg_mask_cpu)))  # 根据层数调整图形大小
    # 绘制每个层的通道掩码
    for i, mask in enumerate(cfg_mask_cpu):
        plt.subplot(len(cfg_mask_cpu), 1, i + 1)  # 创建子图
        plt.bar(range(len(mask)), mask)  # 绘制条形图
        plt.title(f'Layer {i + 1}')  # 设置标题
        plt.ylim(0, 1)  # 设置y轴的范围
        plt.xticks(range(len(mask)))  # 设置x轴刻度标签
        plt.ylabel('Channel Mask')

    # 保存图像
    plt.savefig(str(client_id) + '_cfg_mask_strip_barchart.png', bbox_inches='tight')  # 保存为png文件
    plt.close()  # 关闭绘图窗口


def save_mask_picture_heatmap(client_id, cfg_mask):
    cfg_mask = [mask.cpu() for mask in cfg_mask]  # 确保每个tensor都被移至CPU
    # 计算需要填充的最大长度
    max_length = max(mask.size(0) for mask in cfg_mask)  # 使用Tensor的size方法而不是len

    # 将每个 Tensor 转换为列表并进行填充
    cfg_mask_padded = np.array([mask.tolist() + [-1] * (max_length - mask.size(0)) for mask in cfg_mask])

    # 绘制热力图
    plt.figure(figsize=(12, len(cfg_mask_padded)))  # 根据层数调整图形大小
    plt.imshow(cfg_mask_padded, cmap='viridis', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(label='Channel Mask', ticks=[-1, 0, 1])
    plt.title('Channel Masks Across Layers')
    plt.xlabel('Channels')
    plt.ylabel('Layers')
    plt.yticks(range(len(cfg_mask_padded)), [f'Layer {i + 1}' for i in range(len(cfg_mask_padded))])  # 设置y轴刻度标签
    plt.xticks(range(max_length))  # 设置x轴刻度标签
    plt.grid(False)  # 关闭网格线
    plt.savefig(str(client_id) + '_cfg_mask_strip_heatmap.png', bbox_inches='tight')  # 保存为png文件
    plt.close()  # 关闭绘图窗口
