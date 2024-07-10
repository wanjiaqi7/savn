# 实现了一个多进程强化学习训练的主循环，其中每个进程都独立地进行训练，并在每个训练步骤后更新共享的模型参数
from __future__ import division

import time
import random
import setproctitle
from datasets.glove import Glove
from datasets.data import get_data

from models.model_io import ModelOptions

from .train_util import (
    compute_loss,
    new_episode,
    run_episode,
    end_episode,
    transfer_gradient_to_shared,
    get_params,
    reset_player,
    SGD_step,
    compute_learned_loss,
)


def savn_train(
    rank,                      # 当前进程的标识符，用于在多进程训练中区分不同的进程
    args,                      # 训练参数，包含训练所需的各种配置信息
    create_shared_model,       # 创建共享模型的函数
    shared_model,              # 共享模型，在多进程训练中所有进程共享该模型
    initialize_agent,          # 初始化智能体的函数
    optimizer,                 # 优化器，用于更新模型参数
    res_queue,                 # 结果队列，用于进程间通信
    end_flag,                  # 结束标志，指示训练何时结束
):
    # 初始化
    glove = Glove(args.glove_file)             # 加载 GloVe 词向量文件，用于处理文本数据
    scenes, possible_targets, targets = get_data(args.scene_types, args.train_scenes)           #  获取训练场景、可能的目标和实际目标
 
    random.seed(args.seed + rank)                           # 设置随机种子以确保结果可重复
    idx = [j for j in range(len(args.scene_types))]
    random.shuffle(idx)

    setproctitle.setproctitle("Training Agent: {}".format(rank))          # 设置进程标题，以便在系统监控中区分不同的训练进程

    gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]          # 为当前进程分配的 GPU ID

    import torch

    torch.cuda.set_device(gpu_id)
    torch.manual_seed(args.seed + rank)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed + rank)

    player = initialize_agent(create_shared_model, args, rank, gpu_id=gpu_id)         # 初始化智能体

    model_options = ModelOptions()       # 模型选项，存储模型参数
   
    # 训练循环
    j = 0

    while not end_flag.value:

        start_time = time.time()
        new_episode(                                # 通过 new_episode 函数初始化一个新的训练场景
            args, player, scenes[idx[j]], possible_targets, targets[idx[j]], glove=glove
        )
        player.episode.exploring = True
        total_reward = 0
        player.eps_len = 0

        # theta <- shared_initialization
        params_list = [get_params(shared_model, gpu_id)]
        model_options.params = params_list[-1]
        loss_dict = {}
        reward_dict = {}
        episode_num = 0
        num_gradients = 0

        # Accumulate loss over all meta_train episodes.
        while True:
            # Run episode for k steps or until it is done or has made a mistake (if dynamic adapt is true).
            if args.verbose:
                print("New inner step")
            total_reward = run_episode(player, args, total_reward, model_options, True)

            if player.done:
                break

            if args.gradient_limit < 0 or episode_num < args.gradient_limit:

                num_gradients += 1

                # Compute the loss.
                learned_loss = compute_learned_loss(args, player, gpu_id, model_options)

                if args.verbose:
                    print("inner gradient")
                inner_gradient = torch.autograd.grad(
                    learned_loss["learned_loss"],
                    [v for _, v in params_list[episode_num].items()],
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True,
                )

                params_list.append(
                    SGD_step(params_list[episode_num], inner_gradient, args.inner_lr)
                )
                model_options.params = params_list[-1]

                # reset_player(player)
                episode_num += 1

                for k, v in learned_loss.items():
                    loss_dict["{}/{:d}".format(k, episode_num)] = v.item()

        loss = compute_loss(args, player, gpu_id, model_options)

        for k, v in loss.items():
            loss_dict[k] = v.item()
        reward_dict["total_reward"] = total_reward

        if args.verbose:
            print("meta gradient")

        # Compute the meta_gradient, i.e. differentiate w.r.t. theta.
        meta_gradient = torch.autograd.grad(                 
            loss["total_loss"],
            [v for _, v in params_list[0].items()],
            allow_unused=True,
        )

        end_episode(
            player,
            res_queue,
            title=args.scene_types[idx[j]],
            episode_num=0,
            total_time=time.time() - start_time,
            total_reward=total_reward,
        )

        # Copy the meta_gradient to shared_model and step.
        transfer_gradient_to_shared(meta_gradient, shared_model, gpu_id)
        optimizer.step()
        reset_player(player)

        j = (j + 1) % len(args.scene_types)

    player.exit()
