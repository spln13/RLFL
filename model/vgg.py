import torch
import torch.nn as nn
import torch.nn.functional as F

# MiniVGG的配置
mini_vgg_cfg = [32, 64, 'M', 128, 128, 'M']


class MiniVGG(nn.Module):
    def __init__(self, cfg=None, num_classes=10, dataset='cifar10'):
        super(MiniVGG, self).__init__()
        self.dataset = dataset
        self.mask = None
        self.in_channels = 3 if dataset == 'cifar10' or dataset == 'ImageNet10' else 1
        if cfg is None:
            cfg = mini_vgg_cfg
        self.cfg = cfg
        self.feature = self.make_layers(cfg)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def make_layers(self, cfg, batchnorm=True):
        layers = []
        in_channels = self.in_channels
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batchnorm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 测试模型
if __name__ == "__main__":
    model = MiniVGG(num_classes=10)
    x = torch.randn(1, 3, 32, 32)  # CIFAR-10 输入大小为 32x32 RGB 图像
    output = model(x)
    print(output.shape)  # 应该输出 torch.Size([1, 10])
