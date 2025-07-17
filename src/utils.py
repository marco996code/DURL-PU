#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import matplotlib
import matplotlib.pyplot as plt
import  sys
matplotlib.use("Agg")
import shutil
from torch.optim.optimizer import Optimizer
import copy
from torch.utils.data import DataLoader, Subset
import torch
from torchvision import datasets, transforms
# from data_distr import *
from PIL import Image
import  cv2
import numpy as np
from torch import nn
import torch.nn.functional as F
from torchvision.models.resnet import *
from tqdm import tqdm
import IPython
from torch.utils.data import Dataset
import torchvision
import os
import json
import random
import csv
import math
import torchvision.transforms as T
import time
import glob
from datetime import datetime
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import RandomSampler
from torch.utils.data import DataLoader
from mae_model import *
from sampling import *
import torchvision.models as models
plt_param = {
    "legend.fontsize": 65,
    "figure.figsize": (54, 36),  # (72, 48)
    "axes.labelsize": 80,
    "axes.titlesize": 80,
    "font.size": 80,
    "xtick.labelsize": 80,
    "ytick.labelsize": 80,
    "lines.linewidth": 10,
    "lines.color": (0, 0, 0),
}

plt.rcParams.update(plt_param)

modified_cifar_data = None
modified_cifar_test_data = None


def get_classifier_dataset(args):
    if args.dataset.endswith("ssl"):
        args.dataset = args.dataset[:-3]  # remove the ssl
        print(args.dataset)
    train_dataset, test_dataset, _, _, _ = get_dataset(args)
    return train_dataset, test_dataset


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def get_dist_env():
    if "OMPI_COMM_WORLD_SIZE" in os.environ:
        world_size = int(os.getenv("OMPI_COMM_WORLD_SIZE"))
    else:
        world_size = int(os.getenv("SLURM_NTASKS"))

    if "OMPI_COMM_WORLD_RANK" in os.environ:
        global_rank = int(os.getenv("OMPI_COMM_WORLD_RANK"))
    else:
        global_rank = int(os.getenv("SLURM_PROCID"))
    return global_rank, world_size


def global_repr_global_classifier(args, global_model, test_epoch=60):
    # global representation, global classifier
    from models import ResNetCifarClassifier
    from update import test_inference

    device = "cuda"

    train_dataset, test_dataset = get_classifier_dataset(args)

    if args.ssl_method == "mae":
        global_model = global_model.module
        global_model_classifer = ViT_Classifier(
            global_model.encoder, num_classes=20
        ).to(device)
        global_model_classifer = global_model_classifer.cuda()
        optimizer = torch.optim.AdamW(
            global_model_classifer.head.parameters(), lr=3e-4, weight_decay=0.05
        )

    else:
        print("begin training classifier...")
        global_model_classifer = ResNetCifarClassifier(args=args)
        if hasattr(global_model, "module"):
            global_model = global_model.module
        global_model_classifer.load_state_dict(
            global_model.state_dict(), strict=False
        )  #
        global_model_classifer = global_model_classifer.cuda()
        for param in global_model_classifer.f.parameters():
            param.requires_grad = False

        # train only the last layer
        optimizer = torch.optim.Adam(
            global_model_classifer.fc.parameters(), lr=1e-3, weight_decay=1e-6
        )

    # remove the ssl in the training dataset name
    dist_sampler = (
        DistributedSampler(train_dataset)
        if args.distributed_training
        else RandomSampler(train_dataset)
    )
    trainloader = DataLoader(
        train_dataset,
        sampler=dist_sampler,
        batch_size=64,
        num_workers=16,
        pin_memory=False,
    )
    criterion = (
        torch.nn.NLLLoss().to(device)
        if args.ssl_method != "mae"
        else torch.nn.CrossEntropyLoss().to(device)
    )
    best_acc = 0

    # train global model on global dataset
    for epoch_idx in tqdm(range(test_epoch)):
        batch_loss = []
        if args.distributed_training:
            dist_sampler.set_epoch(epoch_idx)

        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = global_model_classifer(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print(
                    "Downstream Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch_idx + 1,
                        batch_idx * len(images),
                        len(trainloader.dataset),
                        100.0 * batch_idx / len(trainloader),
                        loss.item(),
                    )
                )
            batch_loss.append(loss.item())

        loss_avg = sum(batch_loss) / len(batch_loss)
        test_acc, test_loss = test_inference(args, global_model_classifer, test_dataset)
        if test_acc > best_acc:
            best_acc = test_acc
        print("\n Downstream Train loss: {} Acc: {}".format(loss_avg, best_acc))
    return best_acc


def write_log_and_plot(
    model_time, write_log_and_plotmodel_output_dir, args, suffix, test_acc, intermediate=False
):
    # mkdir_if_missing('save')
    if not os.path.exists("save/" + args.log_directory):
        os.makedirs("save/" + args.log_directory)

    log_file_name = (
        args.log_file_name + "_intermediate" if intermediate else args.log_file_name
    )
    elapsed_time = (
        (datetime.now() - args.start_time).seconds if hasattr(args, "start_time") else 0
    )
    with open(
        "save/{}/best_linear_statistics_{}.csv".format(
            args.log_directory, log_file_name
        ),
        "a+",
    ) as outfile:
        writer = csv.writer(outfile)

        res = [
            suffix,
            "",
            "",
            args.dataset,
            "acc: {}".format(test_acc),
            "num of user: {}".format(args.num_users),
            "frac: {}".format(args.frac),
            "epoch: {}".format(args.epochs),
            "local_ep: {}".format(args.local_ep),
            "local_bs: {}".format(args.local_bs),
            "lr: {}".format(args.lr),
            "backbone: {}".format(args.backbone),
            "dirichlet {}: {}".format(args.dirichlet, args.dir_beta),
            "imagenet_based_cluster: {}".format(args.imagenet_based_cluster),
            "partition_skew: {}".format(args.y_partition_skew),
            "partition_skew_ratio: {}".format(args.y_partition_ratio),
            "iid: {}".format(args.iid),
            "reg scale: {}".format(args.reg_scale),
            "cont opt: {}".format(args.model_continue_training),
            model_time,
            "elapsed_time: {}".format(elapsed_time),
        ]
        writer.writerow(res)

        name = "_".join(res).replace(": ", "_")
        print("writing best results for {}: {} !".format(name, test_acc))


def get_global_model(args, train_dataset):
    from models import SimCLR
    from pu import  SimCLR as PU

    if args.ssl_method == "simclr":
        global_model = SimCLR(args=args)
    elif args.ssl_method == "pu":
        global_model = PU(args=args)


    return global_model


def mkdir_if_missing(dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)


def save_args_json(path, args):
    mkdir_if_missing(path)
    arg_json = os.path.join(path, "args.json")
    with open(arg_json, "w") as f:
        args = vars(args)
        json.dump(args, f, indent=4, sort_keys=True)


def print_and_write(file_handle, text):
    print(text)
    if file_handle is not None:
        file_handle.write(text + "\n")
    return text


def adjust_learning_rate(optimizer, init_lr, epoch, full_epoch, local_epoch):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1.0 + math.cos(math.pi * epoch / full_epoch))
    for param_group in optimizer.param_groups:
        if "fix_lr" in param_group and param_group["fix_lr"]:
            param_group["lr"] = init_lr
        else:
            param_group["lr"] = cur_lr


def get_backbone(pretrained_model_name="resnet50", pretrained=False, full_size=False):
    f = []
    model = eval(pretrained_model_name)(pretrained=pretrained)

    for name, module in model.named_children():

        if name == "conv1" and not full_size:  # add not full_size
            module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        if full_size:
            print(name)
            if name != "fc":
                f.append(module)
        else:
            if not isinstance(module, nn.Linear) and not isinstance(
                module, nn.MaxPool2d
            ):
                f.append(module)

    # encoder
    f = nn.Sequential(*f)
    feat_dim = 2048 if "resnet50" in pretrained_model_name else 512
    print("feat dim:", feat_dim)
    return f, feat_dim

class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

class MLPHead(nn.Module):
    def __init__(self, in_channels, mlp_hidden_size, projection_size):
        super(MLPHead, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_size),
            nn.BatchNorm1d(mlp_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)

class backbone_byol(nn.Module):
    def __int__(self):
        super(backbone_byol,self).__int__()
        resnet = models.resnet18(pretrained=True)

        self.encoder = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.projection = MLPHead(512,4096,2048)

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.shape[0], h.shape[1])
        return self.projetion(h)

def get_backbone_byol(pretrained_model_name="resnet50", pretrained=False, full_size=False):
    f = []
    model = eval(pretrained_model_name)(pretrained=pretrained)

    for name, module in model.named_children():

        if name == "conv1" and not full_size:  # add not full_size
            module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        if full_size:
            print(name)
            if name != "fc":
                f.append(module)
        else:
            if not isinstance(module, nn.Linear) and not isinstance(
                module, nn.MaxPool2d
            ):
                f.append(module)

    # encoder
    f = nn.Sequential(*f, Flatten(), nn.Sequential(
            # nn.Flatten(start_dim=1),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
        ))
    feat_dim = 2048 if "resnet50" in pretrained_model_name else 512
    print("feat dim:", feat_dim)
    return f, feat_dim

def get_encoder_network(pretrained_model_name="resnet50", pretrained=False,):

    model = eval(pretrained_model_name)(pretrained=pretrained)
    # model.fc = nn.Sequential(
    #     nn.Linear(model.feature_dim, 4096),
    #     nn.BatchNorm1d(4096),
    #     nn.ReLU(inplace=True),
    #     nn.Linear(4096, 2048),
    # )

    return model

def get_dataset(args, **kwargs):
    """Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    global modified_cifar_data, modified_cifar_test_data
    start = time.time()


    if "cifar" in args.dataset:
        data_dir = "/home/data/"
        dataset_name = "CIFAR10"
        train_dataset_func = (
            eval(dataset_name + "Pair")
            if "ssl" in args.dataset
            else getattr(datasets, dataset_name)
        )
        img_size = 32
        if args.use_new_transform == 1:
            train_transform_ = get_transform_byol(img_size)
        elif args.use_new_transform == 2:
            train_transform_ = get_transform_byol_orch(img_size)
        elif args.use_new_transform == 3:
            train_transform_ = get_transform_byol_lightly(img_size)
        elif args.use_new_transform == 4:
            train_transform_ = get_transform_easyfl(img_size)
        else:
            train_transform_ = get_transform(img_size)
        # train_transform_ = SimCLRTransform(args.nor, img_size, False).train_transform
        test_transform_ = test_transform
        # test_transform_ = SimCLRTransform(args.nor, img_size, False).test_transform
        if args.ssl_method == "mae":
            train_transform_ = test_transform_ = get_transform_mae(img_size)

        train_dataset = train_dataset_func(
            root=data_dir,
            train=True,
            download=True,
            transform=train_transform_,
        )
        # train_dataset = train_dataset[:len(train_dataset) // 10]
        # train_dataset = torch.utils.data.ConcatDataset(train_dataset)
        print("dataset sample num:", train_dataset.data.shape)
        test_dataset = getattr(datasets, dataset_name)(
            data_dir, train=False, download=True, transform=test_transform_
        )
        memory_dataset = getattr(datasets, dataset_name)(
            data_dir, train=True, download=True, transform=test_transform_
        )

    elif "imagenet100" in args.dataset:  # tiny imagenet
        data_dir = "data/imagenet100_v2"
        if "ssl" in args.dataset:
            train_dataset = ImageFolderPair(
                root=os.path.join(data_dir, "train"),
                transform=get_transform_imagenet(224),
                rescale=False,
            )
        else:
            # ImageFolderInstance
            train_dataset = ImageFolderInstance(
                root=os.path.join(data_dir, "train"),
                transform=linear_transform_imagenet,
                rescale=False,
                save_data=args.load_dataset_to_memory,
            )
        test_dataset = ImageFolderInstance(
            os.path.join(data_dir, "val"),
            transform=test_transform_imagenet,
            rescale=False,
            save_data=args.load_dataset_to_memory,
        )
        memory_dataset = ImageFolderInstance(
            os.path.join(data_dir, "train"),
            transform=test_transform_imagenet,
            rescale=False,
            save_data=args.load_dataset_to_memory,
        )

    elif "aid" in args.dataset:  # tiny imagenet
        data_dir = "/home/code/AID/"
        if "ssl" in args.dataset:
            img_size = 64
            train_dataset = ImageFolderPair(
                root=os.path.join(data_dir, "test"),
                transform=get_transform_imagenet(img_size),
                # transform=SimCLRTransform(img_size,False),
                rescale=True,
                save_data=args.load_dataset_to_memory,
            )
        else:
            # ImageFolderInstance
            train_dataset = ImageFolderInstance(
                root=os.path.join(data_dir, "test"),
                transform=linear_transform_aid,
                rescale=False,
                save_data=args.load_dataset_to_memory,
            )
        test_dataset = ImageFolderInstance(
            root=os.path.join(data_dir, "train"),
            transform=test_transform_aid,
            rescale=True,
            save_data=args.load_dataset_to_memory,
        )
        memory_dataset = ImageFolderInstance(
            root=os.path.join(data_dir, "test"),
            transform=test_transform_aid,
            rescale=True,
            save_data=args.load_dataset_to_memory,
        )
    elif "ucm" in args.dataset:  # tiny imagenet
        data_dir = "/home/shaofanli/code/hsi/dataset/UCM"

        if "ssl" in args.dataset:
            img_size = 64
            train_dataset = ImageFolderPair(
                root=os.path.join(data_dir, "test"),
                transform=get_transform_imagenet(img_size),
                # transform=SimCLRTransform(img_size,False),
                rescale=True,
                save_data=args.load_dataset_to_memory,
            )
        else:
            # ImageFolderInstance
            train_dataset = ImageFolderInstance(
                root=os.path.join(data_dir, "test"),
                transform=linear_transform_aid,
                rescale=False,
                save_data=args.load_dataset_to_memory,
            )
        test_dataset = ImageFolderInstance(
            root=os.path.join(data_dir, "train"),
            transform=test_transform_aid,
            rescale=True,
            save_data=args.load_dataset_to_memory,
        )
        memory_dataset = ImageFolderInstance(
            root=os.path.join(data_dir, "test"),
            transform=test_transform_aid,
            rescale=True,
            save_data=args.load_dataset_to_memory,
        )
    elif "nwpu" in args.dataset:  # tiny imagenet
        data_dir = "/home//code/hsi/dataset/NWPU"

        if "ssl" in args.dataset:
            img_size = 64
            train_dataset = ImageFolderPair(
                root=os.path.join(data_dir, "test"),
                transform=get_transform_imagenet(img_size),
                # transform=SimCLRTransform(img_size,False),
                rescale=True,
                save_data=args.load_dataset_to_memory,
            )
        else:
            # ImageFolderInstance
            train_dataset = ImageFolderInstance(
                root=os.path.join(data_dir, "test"),
                transform=linear_transform_aid,
                rescale=False,
                save_data=args.load_dataset_to_memory,
            )
        test_dataset = ImageFolderInstance(
            root=os.path.join(data_dir, "train"),
            transform=test_transform_aid,
            rescale=True,
            save_data=args.load_dataset_to_memory,
        )
        memory_dataset = ImageFolderInstance(
            root=os.path.join(data_dir, "test"),
            transform=test_transform_aid,
            rescale=True,
            save_data=args.load_dataset_to_memory,
        )
    elif "fair" in args.dataset:
        data_dir= "/home/data/dataset/FAIR1M2.0/"


        img_size = 64
        if args.use_new_transform == 1:
            train_transform_ = get_transform_byol(img_size)
        elif args.use_new_transform == 2:
            train_transform_ = get_transform_byol_orch(img_size)
        elif args.use_new_transform == 3:
            train_transform_ = get_transform_byol_lightly(img_size)
        elif args.use_new_transform == 4:
            train_transform_ = get_transform_easyfl(img_size)
        else:
            train_transform_ = get_transform(img_size)
        if "ssl" in args.dataset:


            train_dataset = ImageFolderPair(
                root=os.path.join(data_dir, "train"),
                transform=train_transform_,
                #transform=get_transform_imagenet(img_size),
                # transform=SimCLRTransform(img_size,False),
                rescale=True,
                save_data=args.load_dataset_to_memory,
            )
        else:

            # ImageFolderInstance
            train_dataset = ImageFolderInstance(
                root=os.path.join(data_dir, "train"),
                transform=train_transform_,
                #transform=linear_transform_fairm,
                rescale=False,
                save_data=False,
            )
        test_dataset = ImageFolderInstance(
            root=os.path.join(data_dir, "test"),
            transform=test_transform,
            # transform=test_transform_fairm,
            rescale=True,
            save_data=False,
        )
        memory_dataset = ImageFolderInstance(
            root=os.path.join(data_dir, "train"),
            transform=test_transform,
           # transform=test_transform_fairm,
            rescale=True,
            save_data=False,
        )

    print("get dataset time: {:.3f}".format(time.time() - start))
    start = time.time()

    # sample training data among users
    if args.iid:
        ## iid=1 代表独立同分布
        # Sample IID user data from Mnist
        user_groups = cifar_iid(train_dataset, args.num_users)
        test_user_groups = cifar_iid(test_dataset, args.num_users)

    else:
        if args.dirichlet:
            print("Y dirichlet sampling")
            user_groups = cifar_noniid_dirichlet(
                train_dataset, args.num_users, args.dir_beta, vis=True
            )
            test_user_groups = cifar_noniid_dirichlet(
                test_dataset, args.num_users, args.dir_beta
            )

        elif args.imagenet_based_cluster:
            print("Feature Clustering dirichlet sampling")
            user_groups = cifar_noniid_x_cluster(
                train_dataset, args.num_users, "img_feature_load", args, vis=True
            )
            test_user_groups = cifar_noniid_x_cluster(
                test_dataset, args.num_users, "img_feature_load", args, test=True
            )

        elif args.y_partition_skew or args.y_partition:
            print("Y partition skewness sampling")
            user_groups = cifar_partition_skew(
                train_dataset, args.num_users, args.y_partition_ratio, vis=True
            )
            test_user_groups = cifar_partition_skew(
                test_dataset, args.num_users, args.y_partition_ratio
            )
        else:
            print("Use i.i.d. sampling")
            user_groups = cifar_iid(train_dataset, args.num_users)
            test_user_groups = cifar_iid(test_dataset, args.num_users)

    print("sample dataset time: {:.3f}".format(time.time() - start))
    print(
        "user data samples:", [len(user_groups[idx]) for idx in range(len(user_groups))]
    )

    return train_dataset, test_dataset, user_groups, memory_dataset, test_user_groups


def average_weights(w, avg_weights=None):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w[0].keys():
        for i in range(1, len(w)):
            w_avg[key] = w_avg[key] + w[i][key]

        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def load_weights_without_batchnorm(model, w):
    """
    Returns the average of the weights.
    """
    model.load_state_dict(
        {k: v for k, v in w.items() if "bn" not in k and "running" not in k},
        strict=False,
    )
    return model


def load_weights(model, w):
    """
    Returns the average of the weights.
    """
    model.load_state_dict({k: v for k, v in w.items()}, strict=False)
    return model


def exp_details(args):
    print("\nExperimental details:")
    print(f"    Model     : {args.model}")
    print(f"    Optimizer : {args.optimizer}")
    print(f"    Learning  : {args.lr}")
    print(f"    Global Rounds   : {args.epochs}\n")
    print(f"    Fraction of users  : {args.frac}")
    print(f"    Local Batch size   : {args.local_bs}")
    print(f"    Local Epochs       : {args.local_ep}\n")
    return


class CIFAR10Pair(datasets.CIFAR10):
    """CIFAR10 Dataset."""

    def __init__(
        self,
        class_id=None,
        tgt_class=None,
        sample_num=10000,
        imb_factor=1,
        imb_type="",
        with_index=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sample_num = sample_num

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return pos_1, pos_2, target


class CIFAR100Pair(CIFAR10Pair):
    """CIFAR100 Dataset."""

    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]

    test_list = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
    ]
    meta = {
        "filename": "meta",
        "key": "fine_label_names",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }


# https://github.com/tjmoon0104/pytorch-tiny-imagenet
class ImageFolderInstance(datasets.ImageFolder):
    """Folder datasets which returns the index of the image (for memory_bank)"""

    def __init__(
        self, root, transform=None, target_transform=None, rescale=True, save_data=True
    ):
        super(ImageFolderInstance, self).__init__(root, transform, target_transform)
        self.num = self.__len__()

        # load into memory
        self.data = []
        self.targets = []
        self.rescale = rescale
        self.tiny = "tiny" in root
        self.test = "test" in root
        data_path = os.path.join(root, "data.npy")
        target_path = os.path.join(root, "targets.npy")
        self.save_data = save_data

        s = time.time()
        if os.path.exists(data_path) and save_data:
            # print(data_path)
            self.data = np.load(data_path, allow_pickle=True)
            self.targets = np.load(target_path, allow_pickle=True)

        else:
            print("start caching dataset")
            for i in range(len(self.imgs)):
                path, target = self.imgs[i]
                if save_data:
                    self.data.append(
                        cv2.resize(np.asarray(self.loader(path)), (256, 256))
                    )  # resize
                self.targets.append(target)
            #  print(target)
            print("finish caching dataset {:.3f}".format(time.time() - s))
            s = time.time()

            if save_data:
                self.data = np.array(self.data)
                np.save(data_path, self.data)

            self.targets = np.array(self.targets)
            np.save(target_path, self.targets)

    def __getitem__(self, index):
        if not self.save_data:
            path, target = self.imgs[index]
            # print(self.imgs[index])
            image = self.loader(path)

        else:
            image, target = self.data[index], self.targets[index]  # self.imgs[index]
            image = Image.fromarray(image)

        if self.rescale:
            #image = image.resize((256, 256))
            image = image.resize((68, 68))

        if self.transform is not None:
            image = self.transform(image)
        return image, target


class ImageFolderPair(ImageFolderInstance):
    """Folder datasets which returns the index of the image (for memory_bank)"""

    def __init__(
        self, root, transform=None, target_transform=None, rescale=True, save_data=True
    ):
        super(ImageFolderPair, self).__init__(root, transform, target_transform)
        self.num = self.__len__()
        self.rescale = rescale
        self.save_data = save_data

    def __getitem__(self, index):
        if not self.save_data:
            path, target = self.imgs[index]
            image = self.loader(path)

        else:
            image, target = self.data[index], self.targets[index]  # self.imgs[index]
            image = Image.fromarray(image)
        if self.rescale:
            #image = image.resize((256, 256))
            image = image.resize((68, 68))

        # image
        if self.transform is not None:
            pos_1 = self.transform(image)
            pos_2 = self.transform(image)

        return pos_1, pos_2, target


def knn_monitor(
    net,
    memory_data_loader,
    test_data_loader,
    epoch,
    k=200,
    t=0.1,
    hide_progress=False,
    vis_tsne=False,
    save_fig_name="model_tsne.png",
    feature_only=False,
):
    net.eval()
    classes = len(memory_data_loader.dataset.classes)
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []

    with torch.no_grad():
        # generate feature bank
        for data, _ in tqdm(
            memory_data_loader,
            desc="Feature extracting",
            leave=False,
            disable=hide_progress,
        ):
            feature = net(data.cuda(non_blocking=True))
            feature = torch.flatten(feature, start_dim=1)
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)

        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        if feature_only:
            net.train()
            return feature_bank

        feature_labels = torch.tensor(
            memory_data_loader.dataset.targets, device=feature_bank.device
        )
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader, desc="kNN", disable=hide_progress)

        for data in test_bar:
            if len(data) == 2:
                data, target = data
            else:
                data, data2, target = data

            # with autocast():
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature = net(data)
            if type(feature) is tuple:  # mae model
                feature = feature[0]
            feature = torch.flatten(feature, start_dim=1)
            feature = F.normalize(feature, dim=1)

            pred_labels = knn_predict(
                feature, feature_bank, feature_labels, classes, k, t
            )
            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
            test_bar.set_postfix({"Accuracy": total_top1 / total_num * 100})
    net.train()
    return total_top1 / total_num * 100, feature_bank

# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(
        feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices
    )
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(
        feature.size(0) * knn_k, classes, device=sim_labels.device
    )
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(
        dim=-1, index=sim_labels.view(-1, 1), value=1.0
    )
    # weighted score ---> [B, C]
    pred_scores = torch.sum(
        one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1),
        dim=1,
    )

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels


get_transform_mae = lambda s: transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ]
)

get_transform = lambda s: transforms.Compose(
    [
        transforms.RandomResizedCrop(s),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        # transforms.RandomApply([transforms.GaussianBlur((3, 3), (1.0, 2.0))], p = 0.2), # added gaussian blur
        # GaussianBlur(kernel_size=int(0.1 * s)),效果一般
        # transforms.RandomApply([ transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
    ]
)
# 72.6
get_transform_byol = lambda s: transforms.Compose(
    [
        transforms.RandomResizedCrop(s),
        #transforms.RandomResizedCrop(s),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.3),
        # transforms.RandomApply([transforms.GaussianBlur((3, 3), (1.0, 2.0))], p = 0.2), # added gaussian blur
        # GaussianBlur(kernel_size=int(0.1 * s)),效果一般
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(1.0, 2.0))], p=0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]
)

get_transform_byol_orch = lambda s: transforms.Compose(
    [
        transforms.RandomResizedCrop(s, scale=(0.5, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=1.0),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

get_transform_byol_lightly = lambda s: transforms.Compose(
    [
        transforms.RandomResizedCrop(s, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262))
    ]
)

get_transform_easyfl = lambda s: transforms.Compose(
    [
        transforms.RandomResizedCrop(s),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
        # transforms.RandomApply([transforms.GaussianBlur((3, 3), (1.0, 2.0))], p = 0.2), # added gaussian blur
        # GaussianBlur(kernel_size=int(0.1 * s)),效果一般
        # transforms.RandomApply([ transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.5),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
    ]
)

get_transform_simsiam_lightly = lambda s: transforms.Compose(
    [
        transforms.RandomResizedCrop(s, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.1),
        transforms.RandomGrayscale(p=0.2),
        # transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
)


test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
    ]
)

get_transform_imagenet = lambda s: transforms.Compose(
    [
        transforms.RandomResizedCrop(s),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomApply(
            [transforms.GaussianBlur((3, 3), (1.0, 2.0))], p=0.2
        ),  # added gaussian blur
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


linear_transform_imagenet = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

linear_transform_fairm = transforms.Compose(
    [
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)



linear_transform_aid = transforms.Compose(
    [
        transforms.RandomResizedCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

test_transform_imagenet = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

test_transform_aid = transforms.Compose(
    [
        transforms.Resize((68, 68)),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
test_transform_fairm = transforms.Compose(
    [
        transforms.Resize((38, 38)),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


class AdamPIDOptimizer(Optimizer):
    r"""Implements AdamPID algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    The implementation of the L2 penalty follows changes proposed in
    `Decoupled Weight Decay Regularization`_.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, P=0., I=5., D=10.):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, P=P, I=I, D=D)
        super(AdamPIDOptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamPIDOptimizer, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            eps = group['eps']
            P = group['P']
            I = group['I']
            D = group['D']
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue;
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)

                state = self.state[p]
                # Lazy state initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['I_buff'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # last step of gradient values
                    state['grad_buff'] = torch.clone(d_p, memory_format=torch.preserve_format)
                    # this step of Differentiation Element
                    state['D_buff'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                # update the steps for each param group update
                state['step'] += 1
                # record the step after step update
                state_step = state['step']

                exp_avg = state['I_buff']
                exp_avg_sq = state['exp_avg_sq']
                g_buf = state['grad_buff']

                bias_correction1 = 1 - beta1 ** state_step
                bias_correction2 = 1 - beta2 ** state_step

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(d_p, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(d_p, d_p, value=1 - beta2)
                # Differentiation Element calculation and g_buff update
                D_buf = state['D_buff'] = d_p - g_buf
                state['grad_buff'] = torch.clone(d_p)

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                step_size = lr / bias_correction1
                # model parameter update
                p.data.addcdiv_(d_p, denom, value=-lr*P).addcdiv_(exp_avg, denom, value=-I*step_size)
                p.data.add_(D_buf, alpha=-lr*D/bias_correction1)

        return loss

class GaussianBlur(object):
    """blur a single image on CPU"""

    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img

class SimCLRTransform:
    """
    A stochastic data augmentation module that transforms any given data example randomly
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    data_format is array or image
    """

    def __init__(self, nor, size=32, gaussian=False,  data_format="array"):
        s = 1
        color_jitter = torchvision.transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        if gaussian:
            self.train_transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToPILImage(mode='RGB'),
                    # torchvision.transforms.Resize(size=size),
                    torchvision.transforms.RandomResizedCrop(size=size),
                    torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
                    torchvision.transforms.RandomApply([color_jitter], p=0.8),
                    torchvision.transforms.RandomGrayscale(p=0.2),
                    GaussianBlur(kernel_size=int(0.1 * size)),
                    # RandomApply(torchvision.transforms.GaussianBlur((3, 3), (1.0, 2.0)), p=0.2),
                    torchvision.transforms.ToTensor(),
                ]
            )
        else:
            if data_format == "array":
                if nor:
                    self.train_transform = torchvision.transforms.Compose(
                        [
                            # torchvision.transforms.ToPILImage(mode='RGB'),
                            # torchvision.transforms.Resize(size=size),
                            torchvision.transforms.RandomResizedCrop(size=size),
                            torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
                            torchvision.transforms.RandomApply([color_jitter], p=0.8),
                            torchvision.transforms.RandomGrayscale(p=0.2),
                            torchvision.transforms.ToTensor(),
                            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
                        ]
                    )
                else:
                    self.train_transform = torchvision.transforms.Compose(
                        [
                            # torchvision.transforms.ToPILImage(mode='RGB'),
                            # torchvision.transforms.Resize(size=size),
                            torchvision.transforms.RandomResizedCrop(size=size),
                            torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
                            torchvision.transforms.RandomApply([color_jitter], p=0.8),
                            torchvision.transforms.RandomGrayscale(p=0.2),
                            torchvision.transforms.ToTensor(),
                            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
                        ]
                    )

            else:
                self.train_transform = torchvision.transforms.Compose(
                    [
                        torchvision.transforms.RandomResizedCrop(size=size),
                        torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
                        torchvision.transforms.RandomApply([color_jitter], p=0.8),
                        torchvision.transforms.RandomGrayscale(p=0.2),
                        torchvision.transforms.ToTensor(),
                        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),

                    ]
                )
        if nor:
            self.test_transform = torchvision.transforms.Compose(
                [
                    # torchvision.transforms.Resize(size=size),
                    torchvision.transforms.ToTensor(),
                    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
                ]
            )

        else:
            self.test_transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize(size=size),
                    torchvision.transforms.ToTensor(),
                ]
            )


        self.fine_tune_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToPILImage(mode='RGB'),
                torchvision.transforms.Resize(size=size),
                torchvision.transforms.ToTensor(),
            ]
        )

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)

def get_subset(dataset, fraction=0.01):
    """Return a subset of the dataset containing the specified fraction of data."""
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    subset_size = int(np.floor(fraction * dataset_size))
    subset_indices = indices[:subset_size]
    return Subset(dataset, subset_indices)

def get_data_loaders(dataset,  nor, image_size=32, batch_size=512, num_workers=8):
    transformation = SimCLRTransform(nor, size=image_size, gaussian=False).test_transform

    if dataset == 'cifar100':
        data_path = "./data/cifar100"
        train_dataset = datasets.CIFAR100(data_path, download=True, transform=transformation)
        test_dataset = datasets.CIFAR100(data_path, train=False, download=True, transform=transformation)
    elif dataset == 'fair':
        data_dir = "/home/data/dataset/FAIR1M2/"
        # data_dir = "/home/shaofanli/code/hsi/dataset/FAIR1M2.0/train/tvt_vehicle/fair2m_cifar_remove_plane_A350_Boeing777_train/"
        train_dataset = ImageFolderInstance(
            root=os.path.join(data_dir, "train"),
            transform=linear_transform_fairm,
            rescale=False,
            save_data=True,
        )
        test_dataset = ImageFolderInstance(
            root=os.path.join(data_dir, "test"),
            transform=test_transform_fairm,
            rescale=True,
            save_data=True,
        )
        # Get 1% subset of the datasets
        train_dataset = get_subset(train_dataset, fraction=0.35)
        test_dataset = get_subset(test_dataset, fraction=0.5)
    elif dataset == 'ucm':
        data_dir = "/home/code/hsi/dataset/UCM"
        train_dataset = ImageFolderInstance(
            root=os.path.join(data_dir, "train"),
            transform=linear_transform_fairm,
            rescale=False,
            save_data=True,
        )
        test_dataset = ImageFolderInstance(
            root=os.path.join(data_dir, "test"),
            transform=test_transform_fairm,
            rescale=True,
            save_data=True,
        )
    elif dataset == 'nwpu':
        data_dir = "/home/code/hsi/dataset/NWPU"
        train_dataset = ImageFolderInstance(
            root=os.path.join(data_dir, "train"),
            transform=linear_transform_fairm,
            rescale=False,
            save_data=True,
        )
        test_dataset = ImageFolderInstance(
            root=os.path.join(data_dir, "test"),
            transform=test_transform_fairm,
            rescale=True,
            save_data=True,
        )
        # # Get 1% subset of the datasets
        # train_dataset = get_subset(train_dataset, fraction=0.35)
        # test_dataset = get_subset(test_dataset, fraction=0.5)
    else:
        data_path = "/home/data/cifar10"
        train_dataset = datasets.CIFAR10(data_path, download=True, transform=transformation)
        test_dataset = datasets.CIFAR10(data_path, train=False, download=True, transform=transformation)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    return train_loader, test_loader





term_width = 150
TOTAL_BAR_LENGTH = 30.
last_time = time.time()
begin_time = last_time

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)
    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.
    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1
    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')
    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time
    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)
    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')
    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))
    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


class LR_Scheduler(object):
    def __init__(self, optimizer, warmup_epochs, warmup_lr, num_epochs, base_lr, final_lr, iter_per_epoch,
                 constant_predictor_lr=False):
        self.base_lr = base_lr
        self.constant_predictor_lr = constant_predictor_lr
        warmup_iter = iter_per_epoch * warmup_epochs
        warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
        decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)
        cosine_lr_schedule = final_lr + 0.5 * (base_lr - final_lr) * (
                    1 + np.cos(np.pi * np.arange(decay_iter) / decay_iter))

        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
        self.optimizer = optimizer
        self.iter = 0
        self.current_lr = 0

    def step(self):
        for param_group in self.optimizer.param_groups:

            if self.constant_predictor_lr and param_group['name'] == 'predictor':
                param_group['lr'] = self.base_lr
            else:
                lr = param_group['lr'] = self.lr_schedule[self.iter]

        self.iter += 1
        self.current_lr = lr
        return lr

    def get_lr(self):
        return self.current_lr