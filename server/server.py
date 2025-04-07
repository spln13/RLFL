import os
import numpy as np
import torch
import torch.nn as nn

from model.vgg import MiniVGG
from PPO.pruning_rate_ppo import PruningPPOAgent
from PPO.training_tensity_ppo import TrainIntensityPPOAgent


class Server(object):
    def __init__(self, device, clients, dataset, memory_capacity, init_models_pr):
        self.device = device
        self.clients = clients  # list of Client objects
        self.dataset = dataset
        self.memory_capacity = memory_capacity
        self.init_models_pr = init_models_pr  # 算法初始化时剪枝得到模型的剪枝率列表
        self.pruningPPOAgent = PruningPPOAgent(5, 3)
        self.tensityPPOAgent = TrainIntensityPPOAgent(5, 3)
        self.init_models_save_path = './init_models/'
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


    def init_models(self):
        # 根据剪枝率列表，提前将各个剪枝率的模型剪枝好发送到客户端
        pr_list = self.init_models_pr
        for _, pr in enumerate(pr_list):
            #  生成剪枝率为pr的模型
            model = MiniVGG(dataset=self.dataset)
            # 这里生成model.mask
            mask = []  # 初始mask生成全1
            # for item in cfg:
            #     if item == 'M':
            #         continue
            #     arr = [1.0 for _ in range(item)]
            #     mask.append(torch.tensor(arr))
            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.Linear):
                    channels = module.weight.data.shape[0]
                    arr = [1.0 for _ in range(channels)]
                    mask.append(arr)
            model.mask = mask
            pruned_model = self.prune(model, pr)
            # save model
            if not os.path.exists(self.init_models_save_path):
                os.makedirs(self.init_models_save_path)
            torch.save({
                'cfg': pruned_model.cfg,
                'mask': pruned_model.mask,
                'state_dict': pruned_model.state_dict(),
            }, self.init_models_save_path + str(pr) + '.pth')


    def run(self, num_rounds=10):
        """
        总的流程：
          1) 设定模型剪枝率列表，提前将各个剪枝率的模型剪枝好发送到客户端
          2) 给所有客户端分发初始模型；客户端做一次“小规模测试训练”并把准确率、训练时间上传
          3) 服务器调用 PPO，得到每个客户端的(训练强度, 剪枝率)
          4) 进入主循环：执行N轮:
             a) 下发本轮的剪枝率/训练强度给客户端 => 客户端 train => 返回(准确率, 时间) (如果选择模型和上一轮大小不一致则需要KD)
             b) 聚合 aggravate(...)
             c) 基于本轮训练信息, 调用 PPO, 得到下一轮新的动作 (训练强度, 剪枝率)
             d) 存储 RL 经验, 若内存满则 ppo.update()
        """
        # -------------------------
        # (1) 初始模型下发 & 客户端测试训练
        # -------------------------
        print("[Server] Step 0: Init pruned models pool")
        self.init_models()

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
            print(f"acc: {acc}, time: {time_used}, prune_action: {prune_action}, prune_lp: {prune_lp}")
            # 3) 训练强度 PPO
            train_action, train_lp = self.tensityPPOAgent.select_action(state_ppo2)
            # 映射到 [1, 10] => epoch
            # 如果train_action是负值, int时要clip
            train_action = max(1, int(train_action * 10))
            print(f"acc: {acc}, time: {time_used}, train_action: {train_action}, train_lp: {train_lp}")
            # 保存下来
            prune_ratios.append(prune_action)
            prune_logprobs.append(prune_lp)
            train_intensities.append(train_action)
            train_logprobs.append(train_lp)

        print("[Server] PPO generated prune_ratios:", prune_ratios)
        print("[Server] PPO generated train_intensities:", train_intensities)

        # =============== STEP 3: 进入主循环 =================
        round_count = 0
        rewards1 = []
        rewards2 = []
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
            self.aggregate()  # 保存到self.path
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
                rewards1.append(R1)
                rewards2.append(R2)
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

    def prune(self, model, pruning_rate):
        # cfg = model.cfg
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
