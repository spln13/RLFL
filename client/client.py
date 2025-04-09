import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.vgg import MiniVGG
from utils.utils import read_client_data
from torch.utils.data import DataLoader
from tqdm import tqdm


class Client(object):
    def __init__(self, client_id, device, model, training_intensity, save_path, dataset, batch_size=16, s=0.0001, kd_epochs=10, kd_alpha=0.5, temperature=2.0):
        self.id = client_id
        self.device = device
        self.model = model
        self.dataset = dataset
        self.training_intensity = training_intensity
        self.batch_size = batch_size
        self.s = s
        self.kd_epochs = kd_epochs
        self.kd_alpha = kd_alpha
        self.temperature = temperature
        self.pr = 1.
        self.model_path = save_path + '/client/' + 'client_' + str(client_id) + '.pth'  # checkpoint path
        self.aggregated_model_path = save_path + 'aggregated' + '_client_' + str(client_id) + '.pth'
        self.information_entropy = self.get_information_entropy()
        self.model_pool_base_path = save_path + '/model_pool/' + '_client_' + str(client_id) + '_'
        self.model_pruning_rate_list = []
        self.last_pruning_rate = 0
        pass


    def get_model_pool_path(self, pr):
        return self.model_pool_base_path + '_pr' + str(pr) + '.pth'

    def save_aggregated_model(self, model, cfg, mask):
        """保存模型到disk"""
        torch.save({
            'cfg': model.cfg,
            'mask': model.mask,
            'state_dict': model.state_dict(),
        }, self.aggregated_model_path)


    def load_aggregated_model(self):
        """
        加载聚合后的模型
        """
        checkpoint = torch.load(self.aggregated_model_path)
        cfg = checkpoint['cfg']
        mask = checkpoint['mask']
        if self.model == 'vgg':
            model = MiniVGG(cfg=cfg, dataset=self.dataset)
        else:
            raise NotImplementedError
        model.load_state_dict(checkpoint['state_dict'])
        model.mask = mask
        return model

    def load_model_from_pool(self, pr):
        """从初始的模型池中加载一个模型"""
        # given pruning rate, load a pruned model from model pool
        pruned_model_path = self.model_pool_base_path + str(pr) + '.pth'
        checkpoint = torch.load(pruned_model_path)
        cfg = checkpoint['cfg']
        model = MiniVGG(cfg)
        model.load_state_dict(checkpoint['state_dict'])
        return model


    def load_train_data(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)

    def load_test_data(self, is_local_test=False, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=False, is_local_test=is_local_test)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)
        pass

    def load_small_model(self):
        # 这里应该load model pool中最小的模型
        pr = self.model_pruning_rate_list[0]
        return self.load_model_from_pool(pr)

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
        start_time = time.time()  # 记录训练开始时间
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
        end_time = time.time()
        total_time = end_time - start_time
        # 在client本地测试集跑一下得到acc一并返回给server
        self.save_model(model, model.cfg, model.mask)
        acc = self.local_test()
        print("[client{} accuracy] accuracy: {}".format(self.id, acc))
        return acc, total_time


    def knowledge_distillation(self, model_teacher, model_student, epochs=10, alpha=0.5, temperature=2.0):
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

    def local_test(self) -> float:
        # client使用本地小测试集进行测试，返回准确率供PPO模型参考
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
        acc = 100. * correct / len(test_loader.dataset)
        return acc


    def local_do(self, pruning_rate, tensity):
        # 首先从self.model_pruning_rate_list获取最接近的pruning_rate
        # 然后根据pruning_rate和tensity进行训练, tensity就是本地训练的epochs
        self.pr = pruning_rate
        self.training_intensity = tensity
        pr = min(self.model_pruning_rate_list, key=lambda x: abs(x - pruning_rate))
        # 根据pr去load模型
        if pr == self.last_pruning_rate:
            # 直接使用聚合后的模型进行训练
            acc, training_time = self.train()
        else:
            # 剪枝率不一样了
            # 需要重新init一个剪枝率为pr的模型，将aggregated model蒸馏到这个模型上
            aggregated_model = self.load_aggregated_model()  # teacher model
            # 将aggregated model蒸馏到self model上
            new_model = self.load_model_from_pool(pr)
            self.knowledge_distillation(aggregated_model, new_model, tensity)
            self.save_model(new_model, new_model.cfg, new_model.mask)


    def get_information_entropy(self):
        """
        获取每个client的信息熵, 客户端初始化时调用赋值
        """
        return 1


    def first_evaluate(self):
        """
        算法开始时，训练小模型得到训练时间T
        """
        model = self.load_small_model()
        # train model 50 轮
        # epoch = 1
        epoch = 50
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
        training_time = end_time - start_time

        # 在client本地测试集跑一下得到acc一并返回给server
        acc = self.local_test()

        return acc, training_time
