def federated_train_round(
        prune_ppo,  # PPO1: 剪枝率 PPO
        trainint_ppo,  # PPO2: 训练强度 PPO
        client_states,  # 所有客户端的状态信息
        test_accuracies,  # 测试得到的准确率
):
    """
    进行一轮联邦训练，并基于 PPO 输出下一轮的剪枝率和训练强度
    """

    # 1) 计算各客户端 reward (示例：针对PPO1)
    # ------------------------------------------------
    # 根据论文/思路，做一些统计，比如：模型大小占比 size_ratio[i]、
    # 客户端准确率 acc[i]、本轮训练时长 time_cost[i] 等
    # 这里仅演示：R1 = alpha*acc[i] - beta*size_ratio[i]
    # 还可以加上其他项
    alpha, beta = 1.0, 0.5
    rewards_prune = []
    for i, (acc_i, size_i) in enumerate(zip(test_accuracies, size_ratio)):
        # reward
        R1_i = alpha * acc_i - beta * size_i
        rewards_prune.append(R1_i)

    # 2) 调用 prune_ppo.store_transition(...) 存储本轮的数据
    # ------------------------------------------------
    # 假设本轮 step 里, 我们在开始时就给每个客户端分配了 prune_ratio，
    # 并记录在 prune_actions[i], prune_logprobs[i] 中
    #   done=是否回合结束(视环境而定)
    for i, state in enumerate(client_states):
        s = state  # shape = (state_dim,)
        a = prune_actions[i]  # 剪枝率(本轮选取的)
        lp = prune_logprobs[i]
        r = rewards_prune[i]
        d = False
        prune_ppo.store_transition(s, a, lp, r, d)

    # 3) 当数据足够后, prune_ppo.update() 进行 PPO 参数更新
    # ------------------------------------------------
    # 可以选择每轮都更新，也可以攒几轮再更新
    prune_ppo.update()

    # 4) 利用更新后的 prune_ppo, 给下一轮做决策 => 得到新的“剪枝率”
    # ------------------------------------------------
    # 例如对每个客户端都调一次 select_action(...)
    next_prune_ratios = []
    next_prune_logprobs = []
    for state in next_round_client_states:
        # 调 Actor
        ratio, logp = prune_ppo.select_action(state)
        # 为了避免出现“过近于0”的极端剪枝，做一个函数变换:
        # 可以让 ratio = 0.05 + 0.95 * ratio
        # 让最小剪枝率为0.05, 最高为1.0(或其他阈值).
        ratio = 0.05 + 0.95 * ratio  # 假设 ratio 本身在[0,1], 这样可避免过小
        next_prune_ratios.append(ratio)
        next_prune_logprobs.append(logp)

    # 5) 对训练强度 PPO2 做同样流程
    # ------------------------------------------------
    # a) 计算 R2 = [min_time - max_time] + eta * avg_acc  (或自定义)
    # b) 将其存入 trainint_ppo 的 memory
    # c) trainint_ppo.update()
    # d) 下一轮 select_action => 得到训练强度
    #    同样可做clip或softmax等后处理

    # 6) 将 next_prune_ratios, next_training_intensities 等信息
    #    下发给客户端执行下一轮剪枝 & 本地训练。
    # ------------------------------------------------

    return next_prune_ratios, next_training_intensities
