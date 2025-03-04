import numpy as np
import torch
import torch.nn as nn

from model.vgg import MiniVGG
from PPO.PPO import ContinuousPPOAgent
from PPO.pruning_rate_ppo import PruningPPOAgent
from PPO.training_tensity_ppo import TrainIntensityPPOAgent


class Server(object):
    def __init__(self, device, clients, dataset, memory_capacity):
        self.device = device
        self.clients = clients  # list of Client objects
        self.dataset = dataset
        self.memory_capacity = memory_capacity
        self.pruningPPOAgent = PruningPPOAgent(5, 3)
        self.tensityPPOAgent = TrainIntensityPPOAgent(5, 3)
        self.history = {
            "round": [],
            "client_accs": [],
            "client_times": []
        }

    def aggregate(self):
        cluster_model = MiniVGG()
        for param in cluster_model.parameters():
            param.data.zero_()  # 将簇模型参数都设置为0
        client_models = []

        for client in self.clients:
            client_model = client.load_model()
            client_mask = client_model.mask
            ratio = 1. / len(self.clients)
            layer_idx_in_mask = 0
            if self.dataset == 'MNIST' or self.dataset == 'emnist_noniid':
                start_mask_client = torch.ones(1).bool()  # 开始的mask是输入图片的通道, 为rgb三通道 若是MNIST则改为1通道
            else:
                start_mask_client = torch.ones(3).bool()  # 开始的mask是输入图片的通道, 为rgb三通道 若是MNIST则改为1通道
            end_mask_client = torch.tensor(client_mask[layer_idx_in_mask], dtype=torch.int).bool()
            client_model = client_model.to(self.device)
            cluster_model = cluster_model.to(self.device)
            start_mask_client = start_mask_client.to(self.device)
            end_mask_client = end_mask_client.to(self.device)
            for client_layer, cluster_layer in zip(client_model.modules(), cluster_model.modules()):
                start_indices = [i for i, x in enumerate(start_mask_client) if x]
                end_indices = [i for i, x in enumerate(end_mask_client) if x]
                if isinstance(client_layer, nn.BatchNorm2d):
                    # with torch.no_grad():
                    #     cluster_layer.weight.data[end_indices] += ratio * client_layer.weight.data
                    #     cluster_layer.bias.data[end_indices] += ratio * client_layer.bias.data
                    # cluster_layer.running_mean.data[end_indices] += ratio * client_layer.running_mean.data
                    # cluster_layer.running_var.data[end_indices] += ratio * client_layer.running_var.data
                    layer_idx_in_mask += 1
                    start_mask_client = end_mask_client[:]
                    if layer_idx_in_mask < len(client_mask):
                        end_mask_client = client_mask[layer_idx_in_mask]
                if isinstance(client_layer, nn.Conv2d):
                    with torch.no_grad():
                        for i, start_idx in enumerate(start_indices):
                            cluster_layer.weight.data[end_indices, start_idx, :, :] += ratio * client_layer.weight.data[
                                                                                               :, i, :, :]

                if isinstance(client_layer, nn.Linear):
                    with torch.no_grad():
                        for i, start_idx in enumerate(start_indices):
                            cluster_layer.weight.data[end_indices, start_idx] += ratio * client_layer.weight.data[:, i]
                        cluster_layer.bias.data[end_indices] += ratio * client_layer.bias.data
                    layer_idx_in_mask += 1
                    start_mask_client = end_mask_client[:]
                    if layer_idx_in_mask < len(client_mask):
                        end_mask_client = client_mask[layer_idx_in_mask]
        # 此时cluster_model已完成异构模型聚合
        # 每个client根据自己的模型结构，从cluster_model中获取子模型
        for i, client in enumerate(self.clients):
            client_model = client_models[i]
            client_mask = client.mask
            layer_idx_in_mask = 0
            if self.dataset == 'MNIST' or self.dataset == 'emnist_noniid':
                start_mask_client = torch.ones(1).bool()  # 开始的mask是输入图片的通道, 为rgb三通道 若是MNIST则改为1通道
            else:
                start_mask_client = torch.ones(3).bool()  # 开始的mask是输入图片的通道, 为rgb三通道 若是MNIST则改为1通道
            end_mask_client = torch.tensor(client_mask[layer_idx_in_mask], dtype=torch.int).bool()
            for client_layer, cluster_layer in zip(client_model.modules(), cluster_model.modules()):
                start_indices = [i for i, x in enumerate(start_mask_client) if x]
                end_indices = [i for i, x in enumerate(end_mask_client) if x]
                if isinstance(client_layer, nn.BatchNorm2d):
                    # with torch.no_grad():
                    #     client_layer.weight.data = cluster_layer.weight.data[end_indices].clone()
                    #     client_layer.bias.data = cluster_layer.bias.data[end_indices].clone()
                    # client_layer.running_mean.data = cluster_layer.running_mean.data[end_indices].clone()
                    # client_layer.running_var.data = cluster_layer.running_var.data[end_indices].clone()
                    layer_idx_in_mask += 1
                    start_mask_client = end_mask_client[:]
                    if layer_idx_in_mask < len(client_mask):
                        end_mask_client = client_mask[layer_idx_in_mask]
                if isinstance(client_layer, nn.Linear):
                    m0 = cluster_layer.weight.data[end_indices, :].clone()
                    with torch.no_grad():
                        client_layer.weight.data = m0[:, start_indices].clone()
                        client_layer.bias.data = cluster_layer.bias.data[end_indices].clone()
                    layer_idx_in_mask += 1
                    start_mask_client = end_mask_client[:]
                    if layer_idx_in_mask < len(client_mask):
                        end_mask_client = client_mask[layer_idx_in_mask]
                if isinstance(client_layer, nn.Conv2d):
                    m0 = cluster_layer.weight.data[end_indices, :, :, :].clone()
                    with torch.no_grad():
                        client_layer.weight.data = m0[:, start_indices].clone()
            client.save_model(client_model, client_model.cfg, client_model.mask)


    def Run(self):
        """
        开始算法过程
        """
        # 给每个client分配初始模型
        for client in self.clients:
            client.init_first_model()
        # 每个client 初始化训练
        training_times = []
        for client in self.clients:
            training_time = client.first_evaluate()
            training_times.append(training_time)

        # 根据训练时间，第一次分配模型大小以及训练强度 调openAI

    def run(self, num_rounds=10):
        """
        总的流程：
          1) 给所有客户端分发初始模型；客户端做一次“小规模测试训练”并把准确率、训练时间上传
          2) 服务器调用 PPO，得到每个客户端的(训练强度, 剪枝率)
          3) 进入主循环：执行N轮:
             a) 下发本轮的剪枝率/训练强度给客户端 => 客户端 train => 返回(准确率, 时间)
             b) 聚合 aggravate(...)
             c) 基于本轮训练信息, 调用 PPO, 得到下一轮新的动作 (训练强度, 剪枝率)
             d) 存储 RL 经验, 若内存满则 ppo.update()
        """
        # -------------------------
        # (1) 初始模型下发 & 客户端测试训练
        # -------------------------
        print("[Server] Step 1: Distribute initial model, do a test/assessment training")
        init_assess_info = []

        for client in self.clients:
            client.init_first_model()  # 下发初始模型

        for client in self.clients:
            # 让客户端进行一次小规模测试训练(或评估)
            # 假设客户端返回 (acc, time_used) 表示准确率和训练时间
            acc, time_used = client.first_evaluate()
            init_assess_info.append((acc, time_used))


        # -------------------------
        # (2) 服务器调用 PPO => 给每个客户端分配 (训练强度, 剪枝率)
        # -------------------------
        # 这里需要先构造 "state"。若您有多维特征，可将 (acc, time, 其它信息) 组合到 state 中。
        # 为简化，假设 state 就放 (acc, time_used) 2维；真实实现中可自行添加
        prune_ratios = []
        train_intensities = []
        for i, (acc, t) in enumerate(init_assess_info):
            # state_ppo1 / state_ppo2 => 可以相同, 也可不相同
            # 示例：直接拼成2维
            state_ppo1 = np.array([acc, t], dtype=np.float32)
            state_ppo2 = np.array([acc, t], dtype=np.float32)

            # 调用 prune_ppo
            prune_ratio, prune_lp = self.pruningPPOAgent.select_action(state_ppo1)
            # 建议做一次 clip 或 offset => 保证不至于太小
            prune_ratio = 0.05 + 0.95 * prune_ratio

            # 调用 train_ppo
            train_intensity, train_lp = self.tensityPPOAgent.select_action(state_ppo2)
            # 同理，可做 clip => 保证 epoch > 1
            train_intensity = max(1, int(train_intensity * 10))  # 示例: 0~1 => 1~10

            prune_ratios.append(prune_ratio)
            train_intensities.append(train_intensity)

            # 可以把本次(action, logprob)等信息临时存起来，下轮计算reward时要用
            # 也可等联邦主循环时再行保存

        print("[Server] PPO generated prune_ratios:", prune_ratios)
        print("[Server] PPO generated train_intensities:", train_intensities)

        # -------------------------
        # (3) 进入主联邦循环
        # -------------------------

        round_count = 0
        for r in range(num_rounds):
            round_count += 1
            print(f"\n[Server] Round {round_count} / {num_rounds}")

            # a) 下发本轮策略 (剪枝率, 训练强度)
            client_infos = []  # 存放本轮各客户端 (acc, time, prune, trainint)

            for i, client in enumerate(self.clients):
                pr = prune_ratios[i]
                ti = train_intensities[i]

                # 让客户端进行本地训练 (我们只给出接口, 不实现细节)
                # train(...) 接收 (prune_ratio, train_epochs) 等参数
                acc, time_used = client.train(pr, ti)

                # 将本次结果记录
                client_infos.append((acc, time_used, pr, ti))

            # b) 聚合所有客户端模型 => 更新 self.global_model
            #   aggravate(...) 内部可以做 FedAvg 或其他聚合策略
            self.aggregate()
            # self.global_model = self.aggregator(self.clients)
            print("[Server] Model aggregated, new global model updated.")

            # c) 基于本轮信息, 再次调用 PPO => 产生下一轮动作
            #    这里我们要先定义如何计算 (state, reward)
            #    以及如何在 memory 中保存 (state, action, reward, done)

            next_prune_ratios = []
            next_train_intensities = []

            # 假设 done 一般为 False, 除非在某些场景回合结束
            done = False if r < (num_rounds - 1) else True

            for i, (acc, time_used, old_pr, old_ti) in enumerate(client_infos):
                # ------- 1) 构造 state --------
                # 示例：state依然简单地放 (acc, time_used)
                s1 = np.array([acc, time_used], dtype=np.float32)  # for prune PPO
                s2 = np.array([acc, time_used], dtype=np.float32)  # for train PPO

                # ------- 2) 构造 reward (只是示例公式) -------
                # 例如: PPO1 reward = acc - 0.2*(模型大小比?)
                #      这里简单地使用: R1 = acc - old_pr (越多剪枝可能越影响精度)
                #      您可参考论文需求自定义
                R1 = float(acc) - old_pr

                # PPO2 reward = - (time 差异?), 这里就简单地 -time 用作示例
                # or 与上一轮最慢和最快 time 的差相关
                R2 = - float(time_used)

                # ------- 3) 把 (state, action, logprob, reward, done) 存进 PPO memory -------
                # “action”和“logprob” 需要是上一轮 select_action 时得到的
                #   => 可能需要在上一轮就存下
                #   => 这里为了演示, 假设我们把 prune_ppo 上次的 logprob 也保存了
                #      (实际上要么您在上一轮就 store_transition, 要么你这里先获取/回溯)

                # 这里示例写法: 先 naive 地再调一次 select_action(...) 获取 logprob
                #  - 真实实现中，最好把 old_logprob 在上一轮就存好。
                # ---- prune part ----
                _, old_lp_prune = self.pruningPPOAgent.select_action(s1)
                self.pruningPPOAgent.store_transition(
                    state=s1,
                    action=old_pr,  # 上一轮动作(剪枝率)
                    logprob=old_lp_prune,  # 未必准确, 仅做示例
                    reward=R1,
                    done=done
                )

                # ---- train part ----
                _, old_lp_train = self.tensityPPOAgent.select_action(s2)
                self.tensityPPOAgent.store_transition(
                    state=s2,
                    action=old_ti,  # 上一轮动作(训练强度)
                    logprob=old_lp_train,  # 同理, 仅示例
                    reward=R2,
                    done=done
                )

                # ------- 4) 产生下一轮动作(剪枝率, 训练强度) -----------
                # 这样下一轮再用
                next_pr, lp_pr = self.pruningPPOAgent.select_action(s1)
                next_pr = 0.05 + 0.95 * next_pr  # 避免过小

                next_ti, lp_ti = self.tensityPPOAgent.select_action(s2)
                next_ti = max(1, int(next_ti * 10))

                next_prune_ratios.append(next_pr)
                next_train_intensities.append(next_ti)

            # d) 检查 memory 是否达到了更新阈值 => update PPO
            #   也可以选择每轮都更新;这里示例按某容量判断
            # prune PPO
            if len(self.pruningPPOAgent.memory['states']) >= self.memory_capacity:
                print("[Server] prune_ppo memory is full, start update ...")
                self.pruningPPOAgent.update()

            # train PPO
            if len(self.tensityPPOAgent.memory['states']) >= self.memory_capacity:
                print("[Server] train_ppo memory is full, start update ...")
                self.tensityPPOAgent.update()

            # e) 存储历史信息(可选)
            acc_list = [info[0] for info in client_infos]
            time_list = [info[1] for info in client_infos]
            self.history["round"].append(r + 1)
            self.history["client_accs"].append(acc_list)
            self.history["client_times"].append(time_list)

            # f) 准备进入下一个 round
            prune_ratios = next_prune_ratios
            train_intensities = next_train_intensities

        # ---- 结束所有轮次 ----
        print("[Server] Training finished.")
