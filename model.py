import torch.nn as nn
import torch.nn.functional as F
from utils import *


class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)


class AdaptiveConcatPool2d(nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`,"
    "a module from fastai v1."
    def __init__(self, output_size=None):
        "Output will be 2*output_size or 2 if output_size is None"
        super().__init__()
        self.output_size = output_size or 1
        self.ap = nn.AdaptiveAvgPool2d(self.output_size)
        self.mp = nn.AdaptiveMaxPool2d(self.output_size)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)


# classifier with Fc after AdaptiveConcatPool2d
class ImgClassifier(nn.Module):
    def __init__(self, final_fmaps, out_dim):
        super(ImgClassifier, self).__init__()
        self.adapt_pool = AdaptiveConcatPool2d((1,1))
        self.flatten = Flatten()
        self.fc = nn.Linear(final_fmaps*2, out_dim)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.)

    def forward(self, x):
        x = self.adapt_pool(x)
        x = self.flatten(x)
        return self.fc(x)


# classifier with multi-sample dropout
# class ImgClassifier(nn.Module):
#     def __init__(self, final_fmaps, out_dim):
#         super(ImgClassifier, self).__init__()
#         self.adapt_pool = AdaptiveConcatPool2d((1,1))
#         self.flatten = Flatten()
#         self.dropouts = nn.ModuleList([nn.Dropout(0.1) for _ in range(5)])
#         self.fc = nn.Linear(final_fmaps*2, out_dim)
#         nn.init.xavier_uniform_(self.fc.weight)
#         nn.init.constant_(self.fc.bias, 0.)

#     def forward(self, x):
#         x = self.adapt_pool(x)
#         pooled_output = self.flatten(x)
#         for i, dropout in enumerate(self.dropouts):
#             if i == 0:
#                 out = self.fc(dropout(pooled_output))
#             else:
#                 out += self.fc(dropout(pooled_output))
#         return out / len(self.dropouts)


# base as resnet50
class HCDNet(nn.Module):
    def __init__(self, net):
        super(HCDNet, self).__init__()
        net_lyrs = [c for n,c in net.named_children()]
        self.backbone1 = nn.Sequential(*net_lyrs[:5])
        self.backbone2 = nn.Sequential(*net_lyrs[5:-2])
        self.classifier = ImgClassifier(net_lyrs[-1].in_features, 1)

    def forward(self, x):
        x = self.backbone1(x)
        x = self.backbone2(x)
        out = self.classifier(x)
        return out


# base as densenet169
# class HCDNet(nn.Module):
#     def __init__(self, net):
#         super(HCDNet, self).__init__()
#         net_lyrs = [c for n,c in net.features.named_children()]
# #         self.backbone = net.features
#         self.backbone1 = nn.Sequential(*net_lyrs[:6])
#         self.backbone2 = nn.Sequential(*net_lyrs[6:])
#         self.classifier = ImgClassifier(net.classifier.in_features, 1)

#     def forward(self, x):
#         x = self.backbone1(x)
#         x = self.backbone2(x)
#         x = F.relu(x, inplace=True)
#         out = self.classifier(x)
#         return out


class WeightEMA(object):
    def __init__(self, model, mu=0.95, sample_rate=1):
        # self.ema_model = copy.deepcopy(model)
        self.mu = mu
        self.sample_rate = sample_rate
        self.sample_cnt = sample_rate
        self.weight_copy = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.weight_copy[name] = (1 - mu) * param.data

    def _update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_average = (1 - self.mu) * param.data + self.mu * self.weight_copy[name]
                self.weight_copy[name] = new_average.clone()

    def set_weights(self, ema_model):
        for name, param in ema_model.named_parameters():
            if param.requires_grad:
                param.data = self.weight_copy[name]

    def on_batch_end(self, model):
        self.sample_cnt -= 1
        if self.sample_cnt == 0:
            self._update(model)
            self.sample_cnt = self.sample_rate


class NNAverage(object):
    def __init__(self, model, mu=0.5):
        self.mu = mu
        self.weight_copy = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.weight_copy[name] = 0

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.weight_copy[name] += self.mu * param.data

    def set_weights(self, avg_model):
        for name, param in avg_model.named_parameters():
            if param.requires_grad:
                param.data = self.weight_copy[name]


def model_optimizer_init(pretrained_net):
    model = HCDNet(copy.deepcopy(pretrained_net))

    params_backbone1 = [p for p in model.backbone1.parameters()]
    params_backbone2 = [p for p in model.backbone2.parameters()]
    params_cls = [p for p in model.classifier.parameters()]

    optimizer = torch.optim.Adam(params=[{'params': params_backbone1}])
    optimizer.add_param_group({'params': params_backbone2})
    optimizer.add_param_group({'params':params_cls})

    return model, optimizer


