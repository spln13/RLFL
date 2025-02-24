import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.vgg import MiniVGG
from utils import read_client_data
from torch.utils.data import DataLoader
from tqdm import tqdm


class Client(object):
    def __init__(self, client_id, device, model, pruning_rate, training_intensity, save_path, dataset, batch_size=16, s=0.0001, kd_epochs=10, kd_alpha=0.5, temperature=2.0):
        self.id = client_id
        self.device = device
        self.model = model
        self.dataset = dataset
        self.pruning_rate = pruning_rate
        self.training_intensity = training_intensity
        self.batch_size = batch_size
        self.s = s
        self.kd_epochs = kd_epochs
        self.kd_alpha = kd_alpha
        self.temperature = temperature
        self.model_path = save_path + '_client_' + str(client_id) + '.pth'  # checkpoint path
        pass


    def load_train_data(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)

    def load_test_data(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=False)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)
        pass

    def load_small_model(self):
        return MiniVGG(cfg=[32, 64, 128, 128, 256, 256, 512, 512], dataset=self.dataset)

    def load_model(self):
        checkpoint = torch.load(self.model_path)
        cfg = checkpoint['cfg']
        mask = checkpoint['mask']
        if self.model == 'vgg':
            model = MiniVGG(cfg=cfg, dataset=self.dataset)
        else:
            raise NotImplementedError
        model.load_state_dict(checkpoint['state_dict'])
        model.mask = mask
        return model

    def save_model(self, model, cfg, mask):
        """保存模型到disk"""
        torch.save({
            'cfg': cfg,
            'mask': mask,
            'state_dict': model.state_dict(),
        }, self.model_path)
        pass

    def train(self, sr=False):
        """模型训练制定epoch，需要统计训练时间，sr为是否加上network slimming的正则项"""
        model = self.load_model()
        epochs = self.training_intensity  # 训练强度
        model = model.to(self.device)
        train_loader = self.load_train_data()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        losses = []
        s = self.s
        start_time = time.time() # 记录训练开始时间
        for epoch in range(epochs):
            if epoch in [int(epochs * 0.5), int(epochs * 0.75)]:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.1
            # training
            model.train()
            train_loader_tqdm = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
            for batch_idx, (data, target) in train_loader_tqdm:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                # pred = output.data.max(1, keepdim=True)[1]
                losses.append(loss.item())
                loss.backward()
                if sr:  # update batchnorm
                    for m in model.modules():
                        if isinstance(m, nn.BatchNorm2d):
                            m.weight.grad.data.add_(s * torch.sign(m.weight.data))  # L1
                optimizer.step()
                train_loader_tqdm.set_description(f'Train Epoch: {epoch} Loss: {loss.item():.6f}')
        self.save_model(model, model.cfg, model.mask)
        end_time = time.time()
        total_time = end_time - start_time
        return total_time


    def finetune(self, model_teacher, model_student, epochs=10, alpha=0.5, temperature=2.0):
        """
        参数:
            model_teacher: 教师模型 (已训练好，参数固定)
            model_student: 学生模型 (需要被微调)
            epochs: 训练轮数
            alpha: 损失函数中, 交叉熵和蒸馏损失的混合系数
            temperature: 温度系数 T，用于放大/柔化教师模型的logits
        """
        train_loader = self.load_train_data()  # 用户自定义函数，加载训练数据
        model_teacher.eval()  # 固定教师模型的参数
        model_student.train()  # 学生模型设置为训练模式

        # 优化器只更新学生模型的参数
        optimizer = torch.optim.SGD(model_student.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        # 常规交叉熵损失
        criterion_ce = nn.CrossEntropyLoss()
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0

            # 用tqdm显示训练进度
            for data, target in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()

                # 教师模型输出 (不需要计算梯度)
                with torch.no_grad():
                    logits_teacher = model_teacher(data)

                # 学生模型输出
                logits_student = model_student(data)

                # 1) 学生模型与真实标签之间的交叉熵损失
                loss_ce = criterion_ce(logits_student, target)

                # 2) 知识蒸馏损失 (KL 散度)
                #   将教师和学生的logits除以温度T，再计算softmax
                #   注: 使用F.log_softmax + F.softmax 或者 F.softmax + F.log_softmax 都可
                #       但要保证一边是log_softmax, 另一边是softmax, 并且reduction='batchmean'
                student_softmax = F.log_softmax(logits_student / temperature, dim=1)
                teacher_softmax = F.softmax(logits_teacher / temperature, dim=1)
                loss_kd = F.kl_div(student_softmax, teacher_softmax, reduction='batchmean') * (temperature ** 2)

                # 最终损失函数: alpha部分来自交叉熵, (1-alpha)部分来自蒸馏损失
                loss = alpha * loss_ce + (1 - alpha) * loss_kd
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            losses.append(avg_loss)
            print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

        return model_student

    def test(self):
        model = self.load_model()
        test_loader = self.load_test_data(batch_size=128)
        model = model.to(self.device)
        model.eval()
        correct = 0
        for data, target in test_loader:
            data, target = data.to(self.device), target.to(self.device)
            with torch.no_grad():
                data, target = data, target
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        print('\nclient_id: {}, Test acc: {}/{} ({:.1f}%)\n'.format(self.id, correct,
                                                                                      len(test_loader.dataset),
                                                                                      100. * correct / len(
                                                                                          test_loader.dataset)))

    def prune(self, pruning_rate):
        model = self.load_model()
        cfg = model.cfg
        mask = model.mask
        total = 0
        for layer_mask in mask:
            total += len(layer_mask)
        bn = torch.zeros(total)  # bn用于存储模型中所有bn层中缩放因子的绝对值
        index = 0
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                size = m.weight.data.shape[0]
                bn[index:index + size] = m.weight.data.abs().clone()
                index += size

        y, i = torch.sort(bn)  # 对缩放因子 升序排列
        threshold_index = int(total * pruning_rate)
        threshold = y[threshold_index]  # 获得缩放因子门槛值，低于此门槛值的channel被prune掉
        pruned = 0
        new_cfg = []  # 每个bn层剩余的通道数或者是maxpooling层, 用于初始化模型
        new_cfg_mask = []  # 存储layer_mask数组
        layer_index = 0  # 当前layer下标 当前层为batchnorm才++
        for k, m in enumerate(model.modules()):
            if isinstance(m, nn.BatchNorm2d):
                weight_copy = m.weight.data.clone()
                layer_mask = weight_copy.ge(threshold).float()  # 01数组
                indices = [i for i, x in enumerate(mask[layer_index]) if x == 1.0]  # 获取之前mask中所有保留通道的下标
                if torch.sum(layer_mask) == 0:  # 如果所有通道都被剪枝了，则保留权重最大的一个通道
                    _, idx = torch.max(weight_copy, 0)
                    layer_mask[idx.item()] = 1.0
                layer_mask = layer_mask.to(self.device)
                pruned += layer_mask.shape[0] - torch.sum(layer_mask)
                m.weight.data.mul_(layer_mask)
                m.bias.data.mul(layer_mask)
                idx = 0
                for _, tag in enumerate(layer_mask):
                    if tag == 0.:  # 该通道应该被剪枝
                        old_mask_index = indices[idx]  # 获取对应之前mask中的下标
                        idx += 1
                        mask[layer_index][old_mask_index] = 0.  # 将mask中对应通道
                new_cfg.append(int(torch.sum(layer_mask)))
                new_cfg_mask.append(layer_mask.clone())
                print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                      format(k, layer_mask.shape[0], int(torch.sum(layer_mask))))
                layer_index += 1
            elif isinstance(m, nn.MaxPool2d):
                new_cfg.append('M')
        model.cfg = new_cfg
        model.mask = new_cfg_mask
        new_model = MiniVGG(cfg=new_cfg, dataset=self.dataset).to(self.device)
        layer_id_in_cfg = 0
        start_mask = torch.ones(3)  # 当前layer_id的层开始时的通道 cifar初始三个输入通道全部保留
        end_mask = new_cfg_mask[layer_id_in_cfg]  # 当前layer_id的层结束时的通道
        for [m0, m1] in zip(model.modules(), new_model.modules()):
            if isinstance(m0, nn.BatchNorm2d):
                idx1 = np.squeeze(
                    np.argwhere(np.asarray(end_mask.cpu().numpy())))  # idx1是end_mask值非0的下标 squeeze()转换为1维数组
                if idx1.size == 1:  # 若只有一个元素则会成为标量，需要转成数组
                    idx1 = np.resize(idx1, (1,))
                m1.weight.data = m0.weight.data[idx1.tolist()].clone()
                m1.bias.data = m0.bias.data[idx1.tolist()].clone()
                m1.running_mean = m0.running_mean[idx1.tolist()].clone()
                m1.running_var = m0.running_var[idx1.tolist()].clone()
                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(new_cfg_mask):  # do not change in Final FC
                    end_mask = new_cfg_mask[layer_id_in_cfg]
            elif isinstance(m0, nn.Conv2d):
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()  # [out_channels, int_channels, H, W]
                w1 = w1[idx1.tolist(), :, :, :].clone()
                m1.weight.data = w1.clone()
            elif isinstance(m0, nn.Linear):
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                m1.weight.data = m0.weight.data[:, idx0].clone()
                m1.bias.data = m0.bias.data.clone()


        return new_model


    def first_evaluate(self):
        """
        算法开始时，训练小模型得到训练时间T
        """
        model = self.load_small_model()
        # train model
        epoch = 1
        model = model.to(self.device)
        train_loader = self.load_train_data()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        losses = []
        start_time = time.time()
        for epoch in range(epoch):
            model.train()
            train_loader_tqdm = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
            for batch_idx, (data, target) in train_loader_tqdm:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
                train_loader_tqdm.set_description(f'Train Epoch: {epoch} Loss: {loss.item():.6f}')
        end_time = time.time()
        return end_time - start_time
