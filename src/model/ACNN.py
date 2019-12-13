import torch
import torch.nn as nn
import torch.nn.functional as F

from .se_layer import SELayer
from .lsoftmax import LSoftmaxLinear
from ..utils import init_weights


def _get_act_func(act_func, in_channels):
    if act_func == 'tanh':
        return nn.Tanh()
    elif act_func == 'leaky_relu':
        # return nn.LeakyReLU(0.05)
        return nn.LeakyReLU(0.01)
    elif act_func == 'relu':
        return nn.ReLU()
    elif act_func == 'prelu':
        return nn.PReLU(init=0.05)
    elif act_func == 'cprelu':
        return nn.PReLU(num_parameters=in_channels, init=0.05)
    elif act_func == 'elu':
        return nn.ELU()
    else:
        raise NotImplementedError


class ConvBlock(nn.Module):

    def __init__(self, in_channel, out_channel,
                 kernel_size, stride, padding, dilation, groups,
                 bias=True, padding_mode='zeros',
                 use_bn=True, use_act=True,
                 act='tanh'):
        super(ConvBlock, self).__init__()
        self.use_bn = use_bn
        self.use_act = use_act

        self.conv = nn.Conv2d(in_channel, out_channel,
                              kernel_size, stride,
                              padding, dilation,
                              groups, bias, padding_mode)
        self.bn = nn.BatchNorm2d(out_channel)
        self.tanh = _get_act_func(act, in_channel)

    def forward(self, x):

        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.use_act:
            x = self.tanh(x)
        return x


class ResidualBlock(nn.Module):
    
    def __init__(self, in_channel, out_channel,
                 kernel_size, stride, p=0.0):
        super(ResidualBlock, self).__init__()

        self.stride = stride

        self.bn1 = nn.BatchNorm2d(in_channel)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=p)

        self.conv1 = nn.Conv2d(in_channel, out_channel, (1, kernel_size),
                               stride=(1, stride), padding=(0, int((kernel_size - stride + 1) / 2)),
                               dilation=1, groups=1)

        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=p)

        self.conv2 = nn.Conv2d(out_channel, out_channel, (1, kernel_size),
                               stride=1, padding=(0, int((kernel_size - 1) / 2)),
                               dilation=1, groups=1)

        self.avg_pool = nn.AvgPool2d(kernel_size=(1, stride))

    def forward(self, x):

        net = self.conv1(self.relu1(self.bn1(x)))
        net = self.conv2(self.relu2(self.bn2(net)))

        if self.stride == 1:
            res = net + x
        else:
            res = net + self.avg_pool(x)

        return res


class ASPP(nn.Module):

    def __init__(self, in_channel, out_channel,
                 kernel_size, dilations=(1, 6, 12, 18),
                 use_bn=True, use_act=True, act_func='tanh'):
        super(ASPP, self).__init__()

        self.num_scale = len(dilations)

        self.convs = nn.ModuleList()
        for dilation in dilations:
            padding = int(dilation * (kernel_size - 1) / 2.0)
            self.convs.append(ConvBlock(in_channel, out_channel,
                                        (1, kernel_size), stride=1, padding=(0, padding),
                                        dilation=(1, dilation), groups=1, use_bn=use_bn,
                                        use_act=use_act, act=act_func))

    def forward(self, x):
        feats = [self.convs[i](x) for i in range(self.num_scale)]
        res = torch.cat(feats, dim=1)
        return res


class ASPP_SE(nn.Module):

    def __init__(self, in_channel, out_channel,
                 kernel_size, dilations=(1, 6, 12, 18), reduction=16,
                 use_bn=True, use_act=True):
        super(ASPP_SE, self).__init__()

        self.num_scale = len(dilations)
        self.reduction = reduction

        self.convs = nn.ModuleList()
        self.se_layers = nn.ModuleList()
        for dilation in dilations:
            padding = int(dilation * (kernel_size - 1) / 2.0)
            self.convs.append(ConvBlock(in_channel, out_channel,
                                        (1, kernel_size), stride=1, padding=(0, padding),
                                        dilation=(1, dilation), groups=1, use_bn=use_bn,
                                        use_act=use_act))
            self.se_layers.append(SELayer(out_channel, reduction=self.reduction))

    def forward(self, x):
        feats = [self.se_layers[i](self.convs[i](x)) for i in range(self.num_scale)]
        res = torch.cat(feats, dim=1)
        return res


class GAP(nn.Module):

    def __init__(self):
        super(GAP, self).__init__()

        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):

        return self.gap(x)


class DomainAttention(nn.Module):

    def __init__(self, in_channel, bank_num=3, reduction=16):
        super(DomainAttention, self).__init__()
        self.channel = in_channel
        self.bank_num = bank_num
        self.reduction = reduction

        self.gap1 = GAP()
        self.gap2 = GAP()

        self.SE_bank = nn.ModuleList()
        for i in range(self.bank_num):
            self.SE_bank.append(nn.Sequential(
                nn.Linear(self.channel, self.channel // self.reduction, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(self.channel // self.reduction, self.channel, bias=False))
            )

        self.fc2 = nn.Linear(self.channel, self.bank_num)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        gap1 = self.gap1(x).view(-1, self.channel)
        banks = []

        for i in range(len(self.SE_bank)):
            resp_i = self.SE_bank[i](gap1).view(-1, self.channel, 1)
            banks.append(resp_i)

        banks = torch.cat(banks, dim=2)

        gap2 = self.gap2(x).view(-1, self.channel)
        gap2 = self.softmax(self.fc2(gap2)).view(-1, self.bank_num, 1)

        net = torch.bmm(banks, gap2).view(-1, self.channel, 1, 1)
        net = self.sigmoid(net)

        return x + x * net


class ACNN(nn.Module):

    def __init__(self, aspp_bn=True, aspp_act=True,
                 lead=2, p=0.0, dilations=(1, 6, 12, 18)):
        super(ACNN, self).__init__()

        self.lead = lead
        self.dila_num = len(dilations)

        self.conv1 = nn.Conv2d(lead, 16, kernel_size=(3, 3), stride=1, padding=(0, 1),
                               dilation=(1, 1), groups=1)
        self.residual_1 = ResidualBlock(16, 16, 3, 1, p=p)
        self.bottleneck = nn.Conv2d(16, 64, kernel_size=1, stride=1,
                                    padding=0, dilation=1, groups=1)
        self.residual_2 = ResidualBlock(64, 64, 3, 2, p=p)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.aspp = ASPP(64, 64, 3, dilations=dilations,
                         use_bn=aspp_bn, use_act=aspp_act)
        self.gap = GAP()
        self.fc = nn.Linear(len(dilations) * 64, 4)

    def forward(self, x):

        net = self.conv1(x)

        net = self.residual_1(net)
        net = self.bottleneck(net)

        net = self.residual_2(net)
        net = self.relu(self.bn(net))

        net = self.gap(self.aspp(net)).view(-1, self.dila_num * 64)

        logits = self.fc(net)

        return net, logits


class ACNN_SE(nn.Module):

    def __init__(self, reduction=16, aspp_bn=True, aspp_act=True,
                 lead=2, p=0.0, dilations=(1, 6, 12, 18)):
        super(ACNN_SE, self).__init__()
        self.lead = lead
        self.dila_num = len(dilations)

        self.conv1 = nn.Conv2d(lead, 16, kernel_size=(3, 3), stride=1, padding=(0, 1),
                               dilation=(1, 1), groups=1)

        self.residual_1 = ResidualBlock(16, 16, 3, 1, p=p)

        self.bottleneck = nn.Conv2d(16, 64, kernel_size=1, stride=1,
                                    padding=0, dilation=1, groups=1)

        self.residual_2 = ResidualBlock(64, 64, 3, 2, p=p)

        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        self.aspp = ASPP(64, 64, 3, dilations=dilations,
                         use_bn=aspp_bn, use_act=aspp_act)
        self.se_layer = SELayer(self.dila_num * 64, reduction=reduction)

        self.gap = GAP()

        self.fc = nn.Linear(len(dilations) * 64, 4)

    def forward(self, x):

        net = self.conv1(x)

        net = self.residual_1(net)
        net = self.bottleneck(net)

        net = self.residual_2(net)
        net = self.relu(self.bn(net))

        net = self.gap(self.se_layer(self.aspp(net))).view(-1, 64 * self.dila_num)

        logits = self.fc(net)

        return net, logits


class MACNN_SE(nn.Module):

    def __init__(self, reduction=16, aspp_bn=True, aspp_act=True,
                 lead=2, p=0.0, dilations=(1, 6, 12, 18), act_func='tanh', f_act_func='tanh'):
        super(MACNN_SE, self).__init__()
        self.lead = lead
        self.dila_num = len(dilations)

        self.conv1 = nn.Conv2d(lead, 4, kernel_size=(3, 3), stride=1, padding=(0, 1),
                               dilation=(1, 1), groups=1)

        self.aspp_1 = ASPP(4, 4, 3, dilations=dilations, use_bn=aspp_bn,
                           use_act=aspp_act, act_func=act_func)
        self.se_layer_1 = SELayer(self.dila_num * 4, reduction=4)

        self.residual_1 = ResidualBlock(self.dila_num * 4, self.dila_num * 4, 3, 1, p=p)

        self.aspp_2 = ASPP(self.dila_num * 4, self.dila_num * 4, 3, dilations=dilations,
                           use_bn=aspp_bn, use_act=aspp_act, act_func=act_func)
        self.se_layer_2 = SELayer(self.dila_num ** 2 * 4, reduction=8)

        self.residual_2 = ResidualBlock(self.dila_num ** 2 * 4, self.dila_num ** 2 * 4, 3, 2, p=p)

        self.bn = nn.BatchNorm2d(self.dila_num ** 2 * 4)
        self.relu = nn.ReLU()

        self.aspp = ASPP(self.dila_num ** 2 * 4, self.dila_num ** 2 * 4, 3, dilations=dilations,
                         use_bn=aspp_bn, use_act=aspp_act, act_func=f_act_func)
        self.se_layer = SELayer(self.dila_num ** 3 * 4, reduction=reduction)

        self.gap = GAP()

        self.fc = nn.Linear(self.dila_num ** 3 * 4, 4)

    def forward(self, x):
        net = self.conv1(x)

        net = self.aspp_1(net)
        net = self.se_layer_1(net)

        net = self.residual_1(net)

        net = self.aspp_2(net)
        net = self.se_layer_2(net)

        net = self.residual_2(net)
        net = self.relu(self.bn(net))

        net = self.gap(self.se_layer(self.aspp(net))).view(-1, self.dila_num ** 3 * 4)

        logits = self.fc(net)

        return net, logits

    def get_feature_maps(self, x):

        net = self.conv1(x)

        net = self.aspp_1(net)
        net = self.se_layer_1(net)

        net = self.residual_1(net)

        net = self.aspp_2(net)
        net = self.se_layer_2(net)

        net = self.residual_2(net)
        net = self.relu(self.bn(net))

        net = torch.abs(net)
        net = torch.sum(net, dim=1).squeeze()

        return net


class MACNN_ATT(nn.Module):

    def __init__(self, reduction=16, aspp_bn=True, aspp_act=True,
                 lead=2, p=0.0, dilations=(1, 6, 12, 18), act_func='tanh', f_act_func='tanh'):
        super(MACNN_ATT, self).__init__()
        self.lead = lead
        self.dila_num = len(dilations)

        self.conv1 = nn.Conv2d(lead, 4, kernel_size=(3, 3), stride=1, padding=(0, 1),
                               dilation=(1, 1), groups=1)

        self.aspp_1 = ASPP(4, 4, 3, dilations=dilations, use_bn=aspp_bn,
                           use_act=aspp_act, act_func=act_func)
        self.se_layer_1 = SELayer(self.dila_num * 4, reduction=4)

        self.residual_1 = ResidualBlock(self.dila_num * 4, self.dila_num * 4, 3, 1, p=p)

        self.aspp_2 = ASPP(self.dila_num * 4, self.dila_num * 4, 3, dilations=dilations,
                           use_bn=aspp_bn, use_act=aspp_act, act_func=act_func)
        self.se_layer_2 = SELayer(self.dila_num ** 2 * 4, reduction=8)

        self.residual_2 = ResidualBlock(self.dila_num ** 2 * 4, self.dila_num ** 2 * 4, 3, 2, p=p)

        self.bn = nn.BatchNorm2d(self.dila_num ** 2 * 4)
        self.relu = nn.ReLU()

        self.aspp = ASPP(self.dila_num ** 2 * 4, self.dila_num ** 2 * 4, 3, dilations=dilations,
                         use_bn=aspp_bn, use_act=aspp_act, act_func=f_act_func)

        self.att = DomainAttention(in_channel=self.dila_num ** 3 * 4, reduction=reduction, bank_num=3)

        self.gap = GAP()

        self.fc = nn.Linear(self.dila_num ** 3 * 4, 4)

    def forward(self, x):
        net = self.conv1(x)

        net = self.aspp_1(net)
        net = self.se_layer_1(net)

        net = self.residual_1(net)

        net = self.aspp_2(net)
        net = self.se_layer_2(net)

        net = self.residual_2(net)
        net = self.relu(self.bn(net))

        net = self.gap(self.att(self.aspp(net))).view(-1, self.dila_num ** 3 * 4)

        logits = self.fc(net)

        return net, logits

    def get_feature_maps(self, x):

        net = self.conv1(x)

        net = self.aspp_1(net)
        net = self.se_layer_1(net)

        net = self.residual_1(net)

        net = self.aspp_2(net)
        net = self.se_layer_2(net)

        net = self.residual_2(net)
        net = self.relu(self.bn(net))

        net = torch.abs(net)
        net = torch.sum(net, dim=1).squeeze()

        return net


class ACNN_deep(nn.Module):

    def __init__(self, aspp_bn=True, aspp_act=True,
                 lead=2, p=0.0, dilations=(1, 6, 12, 18)):
        super(ACNN_deep, self).__init__()
        self.lead = lead
        self.dila_num = len(dilations)

        self.conv1 = nn.Conv2d(lead, 16, kernel_size=(3, 3), stride=1, padding=(0, 1),
                               dilation=(1, 1), groups=1)

        self.residual_1 = ResidualBlock(16, 16, 3, 1, p=p)

        self.bottleneck_1 = nn.Conv2d(16, 64, kernel_size=1, stride=1,
                                      padding=0, dilation=1, groups=1)

        self.residual_2 = ResidualBlock(64, 64, 3, 2, p=p)

        self.bottleneck_2 = nn.Conv2d(64, 128, kernel_size=1, stride=1,
                                      padding=0, dilation=1, groups=1)

        self.residual_3 = ResidualBlock(128, 128, 3, 1, p=p)

        self.bottleneck_3 = nn.Conv2d(128, 256, kernel_size=1, stride=1,
                                      padding=0, dilation=1, groups=1)

        self.residual_4 = ResidualBlock(256, 256, 3, 2, p=p)

        self.bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()

        self.aspp = ASPP(256, 256, 3, dilations=dilations,
                         use_bn=aspp_bn, use_act=aspp_act)

        self.gap = GAP()

        self.fc = nn.Linear(256 * len(dilations), 4)

    def forward(self, x):

        net = self.conv1(x)

        net = self.residual_1(net)
        net = self.bottleneck_1(net)

        net = self.residual_2(net)
        net = self.bottleneck_2(net)

        net = self.residual_3(net)
        net = self.bottleneck_3(net)

        net = self.residual_4(net)
        net = self.relu(self.bn(net))

        net = self.gap(self.aspp(net)).view(-1, 256 * self.dila_num)

        logits = self.fc(net)

        return net, logits


class ACNN_SE_deep(nn.Module):

    def __init__(self, reduction=16, aspp_bn=True, aspp_act=True,
                 lead=2, p=0.0, dilations=(1, 6, 12, 18)):
        super(ACNN_SE_deep, self).__init__()
        self.lead = lead
        self.dila_num = len(dilations)

        self.conv1 = nn.Conv2d(lead, 16, kernel_size=(3, 3), stride=1, padding=(0, 1),
                               dilation=(1, 1), groups=1)

        self.residual_1 = ResidualBlock(16, 16, 3, 1, p=p)

        self.bottleneck_1 = nn.Conv2d(16, 64, kernel_size=1, stride=1,
                                      padding=0, dilation=1, groups=1)

        self.residual_2 = ResidualBlock(64, 64, 3, 2, p=p)

        self.bottleneck_2 = nn.Conv2d(64, 128, kernel_size=1, stride=1,
                                      padding=0, dilation=1, groups=1)

        self.residual_3 = ResidualBlock(128, 128, 3, 1, p=p)

        self.bottleneck_3 = nn.Conv2d(128, 256, kernel_size=1, stride=1,
                                      padding=0, dilation=1, groups=1)

        self.residual_4 = ResidualBlock(256, 256, 3, 2, p=p)

        self.bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()

        self.aspp = ASPP(256, 256, 3, dilations=dilations,
                         use_bn=aspp_bn, use_act=aspp_act)
        self.se_layer = SELayer(self.dila_num * 256, reduction=reduction)

        self.gap = GAP()

        self.fc = nn.Linear(256 * len(dilations), 4)

    def forward(self, x):
        net = self.conv1(x)

        net = self.residual_1(net)
        net = self.bottleneck_1(net)

        net = self.residual_2(net)
        net = self.bottleneck_2(net)

        net = self.residual_3(net)
        net = self.bottleneck_3(net)

        net = self.residual_4(net)
        net = self.relu(self.bn(net))

        net = self.gap(self.se_layer(self.aspp(net))).view(-1, 256 * self.dila_num)

        logits = self.fc(net)

        return net, logits


class ACNN_LSoftmax(nn.Module):

    def __init__(self, aspp_bn=True, aspp_act=True, margin=4):
        super(ACNN_LSoftmax, self).__init__()

        self.margin = margin

        self.conv1 = nn.Conv2d(2, 16, kernel_size=(3, 3), stride=1, padding=(0, 1),
                               dilation=(1, 1), groups=1)

        self.residual_1 = ResidualBlock(16, 16, 3, 1)

        self.bottleneck = nn.Conv2d(16, 64, kernel_size=1, stride=1,
                                    padding=0, dilation=1, groups=1)

        self.residual_2 = ResidualBlock(64, 64, 3, 2)

        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        self.aspp = ASPP(64, 64, 3, dilations=(1, 6, 12, 18), use_bn=aspp_bn, use_act=aspp_act)

        self.gap = GAP()

        self.fc = LSoftmaxLinear(input_features=256, output_features=4, margin=margin)

    def forward(self, x, target=None):

        net = self.conv1(x)

        net = self.residual_1(net)
        net = self.bottleneck(net)

        net = self.residual_2(net)
        net = self.relu(self.bn(net))

        net = self.gap(self.aspp(net)).view(-1, 256)

        logits = self.fc(net, target=target)

        return net, logits
