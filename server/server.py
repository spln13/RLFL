import numpy as np
import torch
import torch.nn as nn

from model.vgg import MiniVGG
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
            client.save_model(client_model, client_model.cfg, client_model.mask)  # 每个client保存自己模型到本地


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
        init_results = []

        for client in self.clients:
            client.init_first_model()  # 下发初始模型

        for client in self.clients:
            # 让客户端进行一次小规模测试训练(或评估)
            # 假设客户端返回 (acc, time_used) 表示准确率和训练时间
            acc, time_used = client.first_evaluate()
            init_results.append((acc, time_used))

        # =============== STEP 2: 调用 PPO => 为每个client分配(剪枝率, 训练强度) ===============
        # 这里需要先定义"state"如何构造, 下面仅示例将 (acc, time_used) 组成 state
        prune_ratios = []
        prune_logprobs = []
        train_intensities = []
        train_logprobs = []

        for (acc, time_used) in init_results:
            # 1) 构造 state
            state_ppo1 = np.array([acc, time_used], dtype=np.float32)
            state_ppo2 = np.array([acc, time_used], dtype=np.float32)

            # 2) 剪枝率 PPO
            prune_action, prune_lp = self.pruningPPOAgent.select_action(state_ppo1)
            # 做一个限制, 避免过小
            prune_action = 0.05 + 0.95 * prune_action

            # 3) 训练强度 PPO
            train_action, train_lp = self.tensityPPOAgent.select_action(state_ppo2)
            # 映射到 [1, 10] => epoch
            # 如果train_action是负值, int时要clip
            train_action = max(1, int(train_action * 10))

            # 保存下来
            prune_ratios.append(prune_action)
            prune_logprobs.append(prune_lp)
            train_intensities.append(train_action)
            train_logprobs.append(train_lp)

        print("[Server] PPO generated prune_ratios:", prune_ratios)
        print("[Server] PPO generated train_intensities:", train_intensities)

        # =============== STEP 3: 进入主循环 =================
        round_count = 0
        for r in range(num_rounds):
            round_count += 1
            done = (r == num_rounds - 1)  # 是否最后一轮
            print(f"\n[Server] Round {round_count} / {num_rounds}")

            # (a) 下发本轮 (prune_ratio, train_intensity) 给客户端并训练
            client_infos = []
            for i, client in enumerate(self.clients):
                pr = prune_ratios[i]
                ti = train_intensities[i]

                # 客户端本地训练, 传入(剪枝率, 训练轮数)
                acc, time_used = client.train(pr, ti)

                # 记录这次训练得到的信息
                client_infos.append({
                    "acc": acc,
                    "time": time_used,
                    "old_prune": pr,
                    "old_prune_lp": prune_logprobs[i],
                    "old_train_int": ti,
                    "old_train_lp": train_logprobs[i]
                })

            # (b) 聚合 => 得到新的 global_model
            # self.global_model = self.aggregator(self.clients)
            self.aggregate()
            print("[Server] Aggregation done, global_model updated.")

            # (c) 计算 PPO 的 (state, action, reward) 并存储
            #     同时为下一轮生成新的 (prune_ratio, train_intensity)
            next_prune_ratios = []
            next_prune_lp = []
            next_train_intensities = []
            next_train_lp = []

            for info in client_infos:
                acc = info["acc"]
                time_used = info["time"]
                old_pr = info["old_prune"]
                old_pr_lp = info["old_prune_lp"]
                old_ti = info["old_train_int"]
                old_ti_lp = info["old_train_lp"]

                # ------- 1) 构造 state -------
                # 仍然用 (acc, time) 组成 state
                state_ppo1 = np.array([acc, time_used], dtype=np.float32)
                state_ppo2 = np.array([acc, time_used], dtype=np.float32)

                # ------- 2) Reward 设计(示例) -------
                # 您可自行定义: 例如 PPO1 reward = acc - old_pr(示例含义:精度越高越好,剪枝越多有惩罚)
                R1 = float(acc) - float(old_pr)
                # PPO2 reward = -time_used(示例:时间越长回报越低)
                R2 = - float(time_used)

                # ------- 3) 存储到 PPO memory -------
                # Prune PPO
                self.pruningPPOAgent.store_transition(
                    state=state_ppo1,
                    action=old_pr,
                    logprob=old_pr_lp,
                    reward=R1,
                    done=done
                )
                # Train PPO
                self.tensityPPOAgent.store_transition(
                    state=state_ppo2,
                    action=old_ti,
                    logprob=old_ti_lp,
                    reward=R2,
                    done=done
                )

                # ------- 4) 生成下一轮新动作 -------
                #  这里再次 select_action, 以便下轮使用
                #  并做必要的映射 clip
                npr_action, npr_lp = self.pruningPPOAgent.select_action(state_ppo1)
                npr_action = 0.05 + 0.95 * npr_action

                nti_action, nti_lp = self.tensityPPOAgent.select_action(state_ppo2)
                nti_action = max(1, int(nti_action * 10))

                next_prune_ratios.append(npr_action)
                next_prune_lp.append(npr_lp)
                next_train_intensities.append(nti_action)
                next_train_lp.append(nti_lp)

            # (d) 内存满则更新
            if len(self.pruningPPOAgent.memory['states']) >= self.memory_capacity:
                print("[Server] prune_ppo memory is full, start update ...")
                self.pruningPPOAgent.update()

            if len(self.tensityPPOAgent.memory['states']) >= self.memory_capacity:
                print("[Server] train_ppo memory is full, start update ...")
                self.tensityPPOAgent.update()

            # (e) 准备下轮
            prune_ratios = next_prune_ratios
            prune_logprobs = next_prune_lp
            train_intensities = next_train_intensities
            train_logprobs = next_train_lp

        print("[Server] Training finished. All rounds complete.")
        # ---- 结束所有轮次 ----
