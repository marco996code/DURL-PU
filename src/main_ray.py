#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import random
import csv
import numpy as np
from tqdm import tqdm
import torch

from tensorboardX import SummaryWriter
from options import args_parser
from models import *
from utils import *
from datetime import datetime
from update import LocalUpdate, test_inference
from pprint import pprint
import IPython

from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.multiprocessing import Process
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import RandomSampler
import socket
import ray

# parallelize local client updates
# might need adjust the parameters
@ray.remote(num_cpus=3, num_gpus=0.15)
class LocalUpdateWrapper(LocalUpdate):
    pass


if __name__ == "__main__":
    start_time = time.time()

    # define paths
    path_project = os.path.abspath("..")
    args = args_parser()
    exp_details(args)

    if args.distributed_training:
        global_rank, world_size = get_dist_env()
        hostname = socket.gethostname()
        print("initing distributed training")
        dist.init_process_group(
            backend="nccl",
            rank=global_rank,
            world_size=world_size,
            init_method=args.dist_url,
        )
        args.world_size = world_size
        args.batch_size *= world_size

    device = "cuda" if args.gpu else "cpu"

    # load dataset and user groups
    set_seed(args.seed)
    (
        train_dataset,
        test_dataset,
        user_groups,
        memory_dataset,
        test_user_groups,
    ) = get_dataset(args)
    pprint(args)

    model_time = datetime.now().strftime("%d_%m_%Y_%H:%M:%S") + "_{}".format(
        str(os.getpid())
    )  # to avoid collision
    model_output_dir = "save/" + model_time
    args.model_time = model_time
    save_args_json(model_output_dir, args)
    logger = SummaryWriter(model_output_dir + "/tensorboard")
    args.start_time = datetime.now()

    # BUILD MODEL
    start_epoch = 0
    global_model = get_global_model(args, train_dataset).to(device)
    global_weights = global_model.state_dict()
    if args.distributed_training:
        global_model = DDP(global_model)
    else:
        global_model = torch.nn.DataParallel(global_model)
    global_model.train()

    # Training
    train_loss = []
    print_every = 200
    local_models = [copy.deepcopy(global_model) for _ in range(args.num_users)]

    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            global_model.parameters(), lr=args.lr, weight_decay=1e-6
        )
    else:
        train_lr = (
            args.lr * args.world_size * (args.batch_size / 256)
            if args.distributed_training
            else args.lr
        )
        optimizer = torch.optim.SGD(global_model.parameters(), lr=train_lr)

    total_epochs = int(args.epochs / args.local_ep / args.frac)
    schedule = [
        int(total_epochs * 0.3),
        int(total_epochs * 0.6),
        int(total_epochs * 0.9),
    ]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=schedule, gamma=0.3
    )
    print("output model:", model_output_dir)
    print(
        "number of users per round: {}".format(max(int(args.frac * args.num_users), 1))
    )
    print("total number of rounds: {}".format(total_epochs))

    # Initialize Ray
    # GPUs = GPUtil.getGPUs()
    # memory_usage = psutil.virtual_memory().percent
    # gpu_limit = max([GPU.memoryTotal for GPU in GPUs])
    ray.init(num_cpus=3 * args.num_users + 6, object_store_memory=1e10)

    local_update_clients = [
        LocalUpdateWrapper.remote(
            args=args,
            dataset=train_dataset,
            idx=idx,
            idxs=user_groups[idx],
            output_dir=model_output_dir,
        )
        for idx in range(args.num_users)
    ]

    for client in local_update_clients:
        client.init_model.remote(global_model)

    lr = optimizer.param_groups[0]["lr"]

    # fix epochs
    epoch = total_epochs
    for epoch in tqdm(range(start_epoch, total_epochs)):

        local_weights, local_losses = [], []
        print(f"\n | Global Training Round : {epoch+1} | Model : {model_time}\n")

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        res = ray.get(
            [
                local_update_clients[idx].update_ssl_weights.remote(
                    model=local_models[idx], global_round=epoch, lr=lr
                )
                for idx in idxs_users
            ]
        )
        local_weights = [s[0] for s in res]
        local_losses = [s[1] for s in res]
        local_models = ray.get(
            [local_update_clients[idx].get_model.remote() for idx in idxs_users]
        )

        # update global weights
        global_weights = average_weights(local_weights)
        if args.average_without_bn:
            for i in range(args.num_users):
                local_models[i] = load_weights_without_batchnorm(
                    local_models[i], global_weights
                )
        else:
            for i in range(args.num_users):
                local_models[i] = load_weights(local_models[i], global_weights)

        # update global weights
        global_model.load_state_dict(global_weights)
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)
        logger.add_scalar("train loss", loss_avg, epoch)

        # print global training loss after every 'i' rounds
        if (int(epoch * args.local_ep) + 1) % print_every == 0:
            print(f" \nAvg Training Stats after {epoch+1} global rounds:")
            print(f"Training Local Client Loss : {np.mean(np.array(train_loss))}")

        scheduler.step()
        lr = scheduler._last_lr[0]
        global_model.module.save_model(model_output_dir, step=epoch)

    global_model.module.save_model(model_output_dir, step=epoch)

    # evaluate representations
    print("evaluating representations: ", model_output_dir)
    test_acc = global_repr_global_classifier(args, global_model, args.finetuning_epoch)

    print(f" \n Results after {args.epochs} global rounds of training:")
    print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))
    print("\n Total Run Time: {0:0.4f}".format(time.time() - start_time))
    # PLOTTING (optional)
    pprint(args)
    suffix = "{}_{}_{}_{}_dec_ssl_{}".format(
        args.model, args.batch_size, args.epochs, args.save_name_suffix, args.ssl_method
    )
    write_log_and_plot(
        model_time,
        model_output_dir,
        args,
        suffix,
        test_acc,
    )
