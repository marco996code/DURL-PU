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
import collections.abc as container_abcs
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
import  sys
module_path = '/home/code/hsi/'
sys.path.append(module_path)
import torch_clustering
from  update import DatasetSplit
class dataset_with_indices(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        outs = self.dataset[idx]
        return [outs, idx]

def sknopp(cZ, lamd=25, max_iters=100):
    with torch.no_grad():
        N_samples, N_centroids = cZ.shape # cZ is [N_samples, N_centroids]
        probs = F.softmax(cZ * lamd, dim=1).T # probs should be [N_centroids, N_samples]

        r = torch.ones((N_centroids, 1), device=probs.device) / N_centroids # desired row sum vector
        c = torch.ones((N_samples, 1), device=probs.device) / N_samples # desired col sum vector

        inv_N_centroids = 1. / N_centroids
        inv_N_samples = 1. / N_samples

        err = 1e3
        for it in range(max_iters):
            r = inv_N_centroids / (probs @ c)  # (N_centroids x N_samples) @ (N_samples, 1) = N_centroids x 1
            c_new = inv_N_samples / (r.T @ probs).T  # ((1, N_centroids) @ (N_centroids x N_samples)).t() = N_samples x 1
            if it % 10 == 0:
                err = torch.nansum(torch.abs(c / c_new - 1))
            c = c_new
            if (err < 1e-2):
                break

        # inplace calculations.
        probs *= c.squeeze()
        probs = probs.T # [N_samples, N_centroids]
        probs *= r.squeeze()

        return probs * N_samples # Soft assignments

def local_clustering(mem_projections, N_local):
    # Local centroids: [# of centroids, D]; local clustering input (mem_projections.T): [m_size, D]
    with torch.no_grad():
        Z = mem_projections.weight.data.T.detach().clone()
        centroids = Z[np.random.choice(Z.shape[0], N_local, replace=False)]
        local_iters = 5
        # clustering
        for it in range(local_iters):
            assigns = sknopp(Z @ centroids.T, max_iters=10)
            choice_cluster = torch.argmax(assigns, dim=1)
            for index in range(N_local):
                selected = torch.nonzero(choice_cluster == index).squeeze()
                selected = torch.index_select(Z, 0, selected)
                if selected.shape[0] == 0:
                    selected = Z[torch.randint(len(Z), (1,))]
                centroids[index] = F.normalize(selected.mean(dim=0), dim=0)
    return  centroids

def convert_to_cuda(data):
    r"""Converts each NumPy array data field into a tensor"""
    elem_type = type(data)
    if isinstance(data, torch.Tensor):
        if data.is_cuda:
            return data
        return data.cuda(non_blocking=True)
    elif isinstance(data, container_abcs.Mapping):
        return {key: convert_to_cuda(data[key]) for key in data}
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        return elem_type(*(convert_to_cuda(d) for d in data))
    elif isinstance(data, container_abcs.Sequence) and not isinstance(data, str):
        return [convert_to_cuda(d) for d in data]
    else:
        return data
@torch.no_grad()
def extract_features(extractor, loader):
    extractor.eval()

    local_features = []
    local_labels = []
    for (inputs, indices) in tqdm(loader):
        images, labels = convert_to_cuda(inputs)
        local_labels.append(labels)
        local_features.append(extractor(images))
    local_features = torch.cat(local_features, dim=0)
    local_labels = torch.cat(local_labels, dim=0)

    indices = torch.Tensor(list(iter(loader.sampler))).long().cuda()

    features = torch.zeros(len(loader.dataset), local_features.size(1)).cuda()
    all_labels = torch.zeros(len(loader.dataset)).cuda()
    counts = torch.zeros(len(loader.dataset)).cuda()
    features.index_add_(0, indices, local_features)
    all_labels.index_add_(0, indices, local_labels.float())
    counts[indices] = 1.
    labels = (all_labels / counts).long()
    features /= counts[:, None]

    return features, labels

class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

def clustering( features, n_clusters):


    kwargs = {
        'metric': 'cosine',
        'distributed': True,
        'random_state': 0,
        'n_clusters': n_clusters,
        'verbose': True
    }
    clustering_model = torch_clustering.PyTorchKMeans(init='k-means++', max_iter=300, tol=1e-4, **kwargs)

    psedo_labels = clustering_model.fit_predict(features)
    cluster_centers = clustering_model.cluster_centers_
    return psedo_labels, cluster_centers



