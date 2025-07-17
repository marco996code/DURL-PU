import torch
import torch.nn as nn
from utils import *
from options import *
from  utils import  get_encoder_network
from functools import wraps

class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
    ):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn

def byol_loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


class prediction_MLP_BYOL_0(nn.Module):
    def __init__(
            self, in_dim=2048, hidden_dim=512, out_dim=2048
    ):  # bottleneck structure
        super().__init__()
        self.layer1 = nn.Sequential(
            # nn.Flatten(start_dim=1),
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        # x = torch.flatten(x, start_dim=1)
        # print(x.size())
        x = self.layer1(x)
        x = self.layer2(x)
        return x

class prediction_MLP_BYOL(nn.Module):
    def __init__(
            self, in_dim=2048, hidden_dim=512, out_dim=2048
    ):  # bottleneck structure
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        # print(x.size())
        x = self.layer1(x)
        return x


def MLP(dim, projection_size, hidden_size=4096, sync_batchnorm=None):
    return nn.Sequential(
        nn.Linear(dim, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size)
    )


class BYOL(nn.Module):
    def __init__(self, args):
        super(BYOL, self).__init__()

        # online
        self.f, feat_dim = get_backbone(args.backbone, full_size=args.full_size)
        self.p = prediction_MLP_BYOL(feat_dim, hidden_dim=1024, out_dim=512)
        # self.p = prediction_MLP_BYOL(feat_dim, hidden_dim=4096, out_dim=2048)
        self.output_dim = feat_dim
        self.online_encoder = self._get_encoder()
        self.target_encoder = self._get_target_encoder()
        # self.f_target = self._get_target_encoder()

        for param_base, param_target in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            # param_target.data.copy_(param_base.data)
            param_target.requires_grad = False

        self.predictor = prediction_MLP_BYOL(512, hidden_dim=1024, out_dim=512)
        # self.predictor = prediction_MLP_BYOL(2048, hidden_dim=4096, out_dim=2048)
        self.target_ema_updater = EMA(0.99)

    # @torch.no_grad()
    # def update_target(self):
    #     tau = self.ema_value
    #     for online, target in zip(self.f.parameters(), self.f_target.parameters()):
    #         target.data = (1-tau) * target.data + tau * online.data
    #     for online, target in zip(self.p.parameters(), self.p_target.parameters()):
    #         target.data = (1-tau) * target.data + tau * online.data
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        return target_encoder

    def _get_encoder(self):
        return nn.Sequential(self.f, Flatten(), self.p)

    def update_moving_average(self):
        assert (
                self.target_encoder is not None
        ), "target encoder has not been created yet"
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def forward(self, pos1, pos2, add_feat=None, scale=1.0, return_feat=False):
        # feature = torch.flatten(self.f(pos1), start_dim=1)
        f1, f2 = self.online_encoder(pos1), self.online_encoder(pos2)
        p1, p2 = self.predictor(f1), self.predictor(f2)
        with torch.no_grad():
            # self.target_encoder = self._get_encoder()
            z1, z2 = self.target_encoder(pos1), self.target_encoder(pos2)
            z1, z2 = z1.detach(), z2.detach()
        loss_one = byol_loss_fn(p1, z2)
        loss_two = byol_loss_fn(p2, z1)
        loss = loss_one + loss_two
        # loss = - (F.cosine_similarity(p1, z2, dim=-1).mean() + F.cosine_similarity(p2, z1, dim=-1).mean()) / 2

        if add_feat is not None:
            print("add_feat")
            if type(add_feat) is list:
                add_feat = [add_feat[0].float().detach(), add_feat[1].float().detach()]
            else:
                add_feat = add_feat.float().detach()
                add_feat = [add_feat.float().detach(), add_feat.float().detach()]
            reg_loss = -0.5 * (
                (add_feat[0] * f1).sum(-1).mean()
                + (add_feat[1] * f2).sum(-1).mean()
            )
            loss = loss + scale * reg_loss

        if return_feat:
            return loss, f1
        return loss

    def save_model(self, model_dir, suffix=None, step=None):
        model_name = (
            "model_{}.pth".format(suffix) if suffix is not None else "model.pth"
        )
        torch.save(
            {"model": self.state_dict(), "step": step},
            "{}/{}".format(model_dir, model_name),
        )

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BYOLModel_0(nn.Module):
    def __init__(self, args):
        super(BYOLModel, self).__init__()
        self.online_encoder, feature_dim = get_backbone_byol(args.backbone, full_size=args.full_size)
        # self.online_encoder = backbone_byol()

        self.online_predictor = MLP(2048, 2048, 4096)
        # self.online_encoder = self._get_encoder()
        # # self.online_encoder.append(nn.Sequential(moudle for (name, moudle) in self.online_encoder_MlP.named_children()) )
        # self.online_predictor = prediction_MLP_BYOL(2048, 4096, 2048)
        self.target_encoder = None

        self.target_ema_updater = EMA(0.996)
        self.forward(torch.randn(256, 3, 32, 32), torch.randn(256, 3, 32, 32))


    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def _get_encoder(self):
        return nn.Sequential(self.online_encoder_beg, Flatten(), self.online_encoder_MlP)

    def update_moving_average(self):
        assert (
                self.target_encoder is not None
        ), "target encoder has not been created yet"
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def forward(self, image_one, image_two, add_feat=None, scale=1.0, return_feat=False):
        # image_one = nn.Flatten(image_one, 1)
        online_pred_one = self.online_encoder(image_one)
        online_pred_two = self.online_encoder(image_two)

        online_pred_one = self.online_predictor(online_pred_one)
        online_pred_two = self.online_predictor(online_pred_two)


        with torch.no_grad():
            target_encoder = self._get_target_encoder()
            target_proj_one = target_encoder(image_one)
            target_proj_two = target_encoder(image_two)

            target_proj_one = target_proj_one.detach()
            target_proj_two = target_proj_two.detach()

        loss_one = byol_loss_fn(online_pred_one, target_proj_two)
        loss_two = byol_loss_fn(online_pred_two, target_proj_one)
        loss = loss_one + loss_two

        if add_feat is not None:
            if type(add_feat) is list:
                add_feat = [add_feat[0].float().detach(), add_feat[1].float().detach()]
            else:
                add_feat = add_feat.float().detach()
                add_feat = [add_feat.float().detach(), add_feat.float().detach()]
            reg_loss = -0.5 * (
                (add_feat[0] * online_pred_one).sum(-1).mean()
                + (add_feat[1] * online_pred_two).sum(-1).mean()
            )
            loss = loss + scale * reg_loss

        if return_feat:
            return loss, online_pred_two
        return loss

    def save_model(self, model_dir, suffix=None, step=None):
        model_name = (
            "model_{}.pth".format(suffix) if suffix is not None else "model.pth"
        )
        torch.save(
            {"model": self.state_dict(), "step": step},
            "{}/{}".format(model_dir, model_name),
        )

class BYOLModel(nn.Module):
    def __init__(self, args):
        super(BYOLModel, self).__init__()
        # self.ema_value = 0.996
        self.stop_gradient = True
        # online
        self.online_encoder, feat_dim = get_backbone_byol(args.backbone, full_size=args.full_size)
        self.online_predictor = prediction_MLP_BYOL(512, hidden_dim=1024, out_dim=512)

        self.target_encoder = self._get_target_encoder()
        self.target_ema_updater = EMA(0.999)
        # self.forward(torch.randn(256, 3, 32, 32), torch.randn(256, 3, 32, 32))
        # self.reset_moving_average()


    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert (
                self.target_encoder is not None
        ), "target encoder has not been created yet"
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)


    def forward(self, image_one, image_two, add_feat=None, scale=1.0, return_feat=False):
        online_pred_one = self.online_encoder(image_one)
        online_pred_two = self.online_encoder(image_two)

        online_pred_one = self.online_predictor(online_pred_one)
        online_pred_two = self.online_predictor(online_pred_two)

        if self.stop_gradient:
            with torch.no_grad():
                # if self.target_encoder is None:
                #     self.target_encoder = self._get_target_encoder()
                target_proj_one = self.target_encoder(image_one)
                target_proj_two = self.target_encoder(image_two)

                target_proj_one = target_proj_one.detach()
                target_proj_two = target_proj_two.detach()

        loss_one = byol_loss_fn(online_pred_one, target_proj_two)
        loss_two = byol_loss_fn(online_pred_two, target_proj_one)
        loss = loss_one + loss_two

        if add_feat is not None:
            if type(add_feat) is list:
                add_feat = [add_feat[0].float().detach(), add_feat[1].float().detach()]
            else:
                add_feat = add_feat.float().detach()
                add_feat = [add_feat.float().detach(), add_feat.float().detach()]
            reg_loss = -0.5 * (
                (add_feat[0] * online_pred_one).sum(-1).mean()
                + (add_feat[1] * online_pred_one).sum(-1).mean()
            )
            loss = loss + scale * reg_loss

        if return_feat:
            return loss, online_pred_one
        return loss

    def save_model(self, model_dir, suffix=None, step=None):
        model_name = (
            "model_{}.pth".format(suffix) if suffix is not None else "model.pth"
        )
        torch.save(
            {"model": self.state_dict(), "step": step},
            "{}/{}".format(model_dir, model_name),
        )


if __name__ == '__main__':
    args = args_parser()
    net = BYOLModel(args=args)
    backbone, dim = get_backbone(args.backbone, full_size=args.full_size)
    res = torchvision.models.resnet18()
    print(backbone)

    # print(net)
    # pos1 = torch.rand(256,3,32,32)
    # pos2 = torch.rand(256, 3, 32, 32)
    # out = net(pos1,pos2)
    # print(out)