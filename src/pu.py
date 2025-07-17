#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import torch
from torch import nn
import torch.nn.functional as F
from utils import get_backbone
import ot  # Python Optimal Transport

from torch.distributions.multivariate_normal import MultivariateNormal

def compute_means_and_covariances(x, pseudo_labels):
    n_samples = x.size(0)
    num_clusters = torch.max(pseudo_labels) + 1
    means = torch.zeros(num_clusters, x.size(1)).to(x.device)
    covariances = torch.zeros(num_clusters, x.size(1), x.size(1)).to(x.device)

    for k in range(num_clusters):
        cluster_mask = pseudo_labels == k
        cluster_points = x[cluster_mask]
        mean = cluster_points.mean(dim=0)
        means[k] = mean
        if cluster_points.size(0) > 1:
            centered = cluster_points - mean
            cov = torch.mm(centered.T, centered) / (cluster_points.size(0) - 1)
            covariances[k] = cov

    average_mean = torch.mean(means, dim=0)
    average_covariance = compute_weighted_average_covariance(x, pseudo_labels, covariances)

    return average_mean, average_covariance


def compute_weighted_average_covariance(x, pseudo_labels, covariances):
    num_clusters = torch.max(pseudo_labels) + 1
    cluster_sizes = torch.zeros(num_clusters).to(x.device)

    for k in range(num_clusters):
        cluster_sizes[k] = (pseudo_labels == k).sum()


    weights = cluster_sizes / cluster_sizes.sum()

    weighted_covariances = torch.einsum('n,nij->ij', weights, covariances)

    return weighted_covariances

def compute_wasserstein_distance(samples1, samples2):
    # 计算两个样本集的成本矩阵
    cost_matrix = ot.dist(samples1.numpy(), samples2.numpy(), metric='euclidean')

    # 假设样本是均匀分布的
    a, b = np.ones((samples1.shape[0],)) / samples1.shape[0], np.ones((samples2.shape[0],)) / samples2.shape[0]
    w_dist = ot.emd2(a, b, cost_matrix)  # 计算Wasserstein距离

    return w_dist

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

class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1] * x.shape[-2] * x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)


class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


class prediction_MLP(nn.Module):
    def __init__(
        self, in_dim=2048, hidden_dim=512, out_dim=2048
    ):  # bottleneck structure
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class SimCLR(nn.Module):
    def __init__(self, args, feature_dim=128, input_dim=32):
        super(SimCLR, self).__init__()
        self.input_dim = input_dim
        self.num_cluster = 10
        # encoder
        self.f, feat_dim = get_backbone(args.backbone, full_size=args.full_size)
        # projection head
        self.g = prediction_MLP(feat_dim, 512, 512)
        self.output_dim = feat_dim

    def compute_centers(self, x, psedo_labels):
        n_samples = x.size(0)
        if len(psedo_labels.size()) > 1:
            weight = psedo_labels.T
        else:
            weight = torch.zeros(10, n_samples).to(x)  # L, N
            weight[psedo_labels, torch.arange(n_samples)] = 1
        weight = F.normalize(weight, p=1, dim=1)  # l1 normalization
        centers = torch.mm(weight, x)
        centers = F.normalize(centers, dim=1)
        return centers

    def forward_k(self, im_k, psedo_labels):
        # compute key features
        with torch.no_grad():  # no gradient to keys
            k = self.f(im_k)  # keys: NxC
            k = self.g(k)
            k = nn.functional.normalize(k, dim=1)

            k = k.detach_()
            all_k = k

            k_centers = self.compute_centers(all_k, psedo_labels)
        return k, k_centers, all_k

    def get_encoder(self):
        return nn.Sequential(self.f, Flatten(), self.g)

    def calc_wasserstein_loss(self, z1, z2):
        z = torch.cat((z1, z2), 0)
        N = z.size(0)
        D = z.size(1)

        z_center = torch.mean(z, dim=0, keepdim=True)
        mean = z.mean(0)
        covariance = torch.mm((z - z_center).t(), z - z_center) / N
        #############calculation of part1
        part1 = torch.sum(torch.multiply(mean, mean))

        ######################################################
        #S, Q = torch.linalg.eig(covariance)
        # S, Q = torch.linalg.eigh(covariance)
        # covariance = covariance.contiguous()
        # S, Q = torch.linalg.eigh(covariance)
        S, Q = torch.eig(covariance, eigenvectors=True)
        # S =torch.abs(S)
        S = torch.abs(S)
        mS = torch.sqrt(torch.diag(S))
        # S = S.to(torch.complex64)  # 使用 torch.complex64 或 torch.complex128
        # mS = mS.to(torch.complex64)  # 使用 torch.complex64 或 torch.complex128
        # Q = Q.to(torch.complex64)
        covariance2 = torch.mm(torch.mm(Q, mS), Q.T)

        #############calculation of part2
        part2 = torch.trace(covariance - 2.0 / math.sqrt(D) * covariance2)
        wasserstein_loss = torch.sqrt(part1 + 1 + part2)

        return wasserstein_loss

    def compute_cluster_loss(self,
                             q_centers,
                             k_centers,
                             temperature,
                             psedo_labels):
        d_q = q_centers.mm(q_centers.T) / temperature
        d_k = (q_centers * k_centers).sum(dim=1) / temperature
        d_q = d_q.float()
        d_q[torch.arange(self.num_cluster), torch.arange(self.num_cluster)] = d_k

        # q -> k
        # d_q = q_centers.mm(k_centers.T) / temperature

        zero_classes = torch.arange(self.num_cluster).cuda()[torch.sum(F.one_hot(torch.unique(psedo_labels),
                                                                                 self.num_cluster), dim=0) == 0]
        mask = torch.zeros((self.num_cluster, self.num_cluster), dtype=torch.bool, device=d_q.device)
        mask[:, zero_classes] = 1
        d_q.masked_fill_(mask, -10)
        pos = d_q.diag(0)
        mask = torch.ones((self.num_cluster, self.num_cluster))
        mask = mask.fill_diagonal_(0).bool()
        neg = d_q[mask].reshape(-1, self.num_cluster - 1)
        loss = - pos + torch.logsumexp(torch.cat([pos.reshape(self.num_cluster, 1), neg], dim=1), dim=1)
        loss[zero_classes] = 0.
        loss = loss.sum() / (self.num_cluster - len(zero_classes))
        return loss

    def forward(self, pos1, pos2=None, add_feat=None, predo_label=None, scale=1.0, return_feat=False):
        temperature = 0.5
        batch_size = len(pos1)
        k, k_centers, all_k = self.forward_k(pos2, predo_label)
        q, q_centers, all_q = self.forward_k(pos1, predo_label)
        #q_centers = self.compute_centers(pos1, predo_label)
        cluster_loss_batch = self.compute_cluster_loss(q_centers, k_centers, 0.5, predo_label)

        # average_mean, average_cov = compute_means_and_covariances(k, predo_label)
        # average_cov = 0.5 * (average_cov + average_cov .t())
        # custom_dist = MultivariateNormal(average_mean, covariance_matrix=average_cov)
        #
        # #
        # standard_mean = torch.zeros_like(average_mean)
        # standard_cov = torch.eye(average_cov.size(0))
        # standard_dist = MultivariateNormal(standard_mean, covariance_matrix=standard_cov)

        pos1 = self.f(pos1)
        #predo_labels = predo_label+1
        feature1 = torch.flatten(pos1, start_dim=1)
        out1 = self.g(feature1)

        pos2 = self.f(pos2)

        feature2 = torch.flatten(pos2, start_dim=1)
        out2 = self.g(feature2)

        out_1, out_2 = F.normalize(out1, dim=-1), F.normalize(out2, dim=-1)
        wasserstein_loss = self.calc_wasserstein_loss(out_1, out_2 )

        out = torch.cat([out_1, out_2], dim=0)

        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = (
            torch.ones_like(sim_matrix)
            - torch.eye(2 * batch_size, device=sim_matrix.device)
        ).bool()

        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)

        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (-torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        loss = cluster_loss_batch.mean() + loss + wasserstein_loss
        if add_feat is not None:
            if type(add_feat) is list:
                add_feat = [add_feat[0].float().detach(), add_feat[1].float().detach()]
            else:
                add_feat = add_feat.float().detach()
                add_feat = [add_feat.float().detach(), add_feat.float().detach()]
            reg_loss = -0.5 * (
                (add_feat[0] * out_1).sum(-1).mean()
                + (add_feat[1] * out_2).sum(-1).mean()
            )
            loss = loss + scale * reg_loss

        if return_feat:
            return loss, feature1
        return loss

    def save_model(self, model_dir, suffix=None, step=None):
        model_name = (
            "model_{}.pth".format(suffix) if suffix is not None else "model.pth"
        )
        torch.save(
            {"model": self.state_dict(), "step": step},
            "{}/{}".format(model_dir, model_name),
        )

from byol import  *
class ResNetCifarClassifier(nn.Module):
    def __init__(self, args, num_class=10, feature_dim=128, data_dim=32):
        super(ResNetCifarClassifier, self).__init__()

        # encoder
        self.data_dim = data_dim
        num_class = 10

        if "imagenet100" in args.dataset:
            num_class = 100
        elif "imagenet" in args.dataset:
            num_class = 1000
        elif "aid" in args.dataset:
            num_class = 30

        if args.ssl_method == "byolm":
            model = BYOLModel(args=args)
            self.f, feature_dim = get_backbone(args.backbone, full_size=args.full_size)
            model.output_dim = feature_dim

            # self.f = backbone_byol().encoder
            # feature_dim = 2048
            # online_encoder_MlP = prediction_MLP_BYOL(feature_dim, 4096, 2048)
            # for (name, moudle) in online_encoder_MlP.named_children():
            #     self.f.append(moudle)
        elif args.ssl_method == "byol":
            model = BYOL(args=args)
            self.f = model.f
        else:
            model = SimCLR(args, feature_dim, data_dim)
            self.f = model.f


        # classifier
        self.fc = nn.Linear(model.output_dim, num_class, bias=True)

    def forward(self, x, return_feat=False):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        if return_feat:
            return F.log_softmax(out, dim=1), feature
        return F.log_softmax(out, dim=1)

    def save_model(self, model_dir, suffix=None, step=None):
        model_name = (
            "model_{}.pth".format(suffix) if suffix is not None else "model.pth"
        )
        torch.save(
            {"model": self.state_dict(), "step": step},
            "{}/{}".format(model_dir, model_name),
        )
