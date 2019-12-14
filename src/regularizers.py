import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math


def get_L2norm_loss_self_driven(x, weight_L2norm):
    radius = x.norm(p=2, dim=1).detach()
    assert radius.requires_grad == False
    radius = radius + 0.3
    l = ((x.norm(p=2, dim=1) - radius) ** 2).mean()
    return weight_L2norm * l


def get_L2norm_loss_self_driven_hard(x, radius, weight_L2norm):
    l = (x.norm(p=2, dim=1).mean() - radius) ** 2
    return weight_L2norm * l


def get_entropy_loss(p_softmax, weight_entropy):
    mask = p_softmax.ge(0.000001)
    mask_out = torch.masked_select(p_softmax, mask)
    entropy = -(torch.sum(mask_out * torch.log(mask_out)))
    return weight_entropy * (entropy / float(p_softmax.size(0)))

