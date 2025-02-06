import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """ ResNet 的基本块，用于 ResNet-18/34 结构 """
    expansion = 1  # 输出通道倍增系数

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 如果输入通道数和输出通道数不同，或者 stride > 1，则使用 1x1 卷积调整维度
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # 残差连接
        return F.relu(out)


class SmallResNet(nn.Module):
    """ 适用于 MNIST / CIFAR-10 的小型 ResNet """
    def __init__(self, num_classes=10):
        super(SmallResNet, self).__init__()
        self.in_channels = 16  # 初始通道数减少

        # 第一个卷积层（去掉 7x7 卷积和最大池化）
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)  # 适用于小尺寸
        self.bn1 = nn.BatchNorm2d(16)

        # ResNet 残差块
        self.layer1 = self._make_layer(16, 2, stride=1)  # 保持大小不变
        self.layer2 = self._make_layer(32, 2, stride=2)  # 下采样
        self.layer3 = self._make_layer(64, 2, stride=2)  # 下采样
        self.layer4 = self._make_layer(128, 2, stride=2)  # 下采样

        # 平均池化 + 全连接
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # 适配不同输入尺寸
        self.fc = nn.Linear(128, num_classes)

    def _make_layer(self, out_channels, blocks, stride):
        """ 创建 ResNet 的一个层 """
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels  # 更新通道数
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


if __name__ == '__main__':
    # 测试模型
    model = SmallResNet(num_classes=10)
    print(model)
    x = torch.randn(1, 3, 32, 32)  # CIFAR-10 的输入
    print(model(x).shape)


