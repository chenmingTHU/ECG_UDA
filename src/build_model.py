import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import os.path as osp

from .losses import FocalLoss, ClassBalanceLoss, ClassBalanceFocalLoss, BinClassBalanceLoss, BatchWeightedLoss, \
    WeightedLoss, ExpWeightedLoss, L1Distance, L2Distance, CosineLoss, DynamicLoss
from .data.dataset import *

from .model.ACNN import ACNN, ACNN_deep, ACNN_SE, ACNN_SE_deep, MACNN_SE, MACNN_ATT, MACNN_MATT
from .model.CNN import CNN, CNN_ATT
from .model.MultiPath_LSTM import BiLSTM


def loss_function(loss, dataset='mitdb', num_ew=75000):

    if dataset == 'fmitdb':
        dataset = 'mitdb'

    categories = {'N': 0, 'V': 0, 'S': 0, 'F': 0}

    root_path = '/home/workspace/mingchen/ECG_UDA/data_index'
    dataset_path = osp.join(osp.join(root_path, dataset), 'entire')

    for cate in categories.keys():
        files = os.listdir(osp.join(dataset_path, cate))
        categories[cate] = len(files)

    categories_bin = {'N': 0, 'VS': 0}

    for cate in categories.keys():
        files = os.listdir(osp.join(dataset_path, cate))
        if cate == 'N':
            categories_bin['N'] = len(files)
        else:
            categories_bin['VS'] += len(files)

    loss_dict = {
        'CELoss': nn.CrossEntropyLoss(),
        'FocalLoss': FocalLoss(gamma=2),
        'CBLoss': ClassBalanceLoss(beta=0.999,
                                   n=categories['N'], v=categories['V'],
                                   s=categories['S'], f=categories['F']),
        'CBFocalLoss': ClassBalanceFocalLoss(beta=0.999,
                                             n=categories['N'], v=categories['V'],
                                             s=categories['S'], f=categories['F'],
                                             gamma=2),
        'BinCBLoss': BinClassBalanceLoss(beta=0.999,
                                         n=categories_bin['N'], vs=categories_bin['VS']),
        'BWLoss': BatchWeightedLoss(beta=0.999),
        'WLoss': WeightedLoss(n=categories['N'], v=categories['V'],
                              s=categories['S'], f=categories['F']),
        'EWLoss': ExpWeightedLoss(n=categories['N'], v=categories['V'],
                                  s=categories['S'], f=categories['F'], beta=num_ew),
        'DLoss': DynamicLoss(n=categories['N'], v=categories['V'],
                             s=categories['S'], f=categories['F'], beta=75000)
    }

    return loss_dict[loss]


def build_distance(distance):

    distance_dict = {'L2': L2Distance(),
                     'Cosine': CosineLoss()}

    return distance_dict[distance]


def build_training_records(source):

    incart_records = os.listdir(DENOISE_DATA_DIRS['incartdb'])
    incart_records = [filename.split('.')[0] for filename in incart_records]

    svdb_records = os.listdir(DENOISE_DATA_DIRS['svdb'])
    svdb_records = [filename.split('.')[0] for filename in svdb_records]

    records = {'mitdb': DS1,
               'fmitdb': DS_ENTIRE,
               'svdb': svdb_records,
               'incartdb': incart_records}

    return records[source]


def build_validation_records(target):

    incart_records = os.listdir(DENOISE_DATA_DIRS['incartdb'])
    incart_records = [filename.split('.')[0] for filename in incart_records]

    svdb_records = os.listdir(DENOISE_DATA_DIRS['svdb'])
    svdb_records = [filename.split('.')[0] for filename in svdb_records]

    records = {'mitdb': DS2,
               'fmitdb': DS_ENTIRE,
               'svdb': svdb_records,
               'incartdb': incart_records}

    return records[target]


def build_acnn_models(model, aspp_bn, aspp_act,
                      lead, p, dilations, act_func, f_act_func):

    if model == 'ACNN':
        return ACNN(aspp_bn=aspp_bn, aspp_act=aspp_act,
                    lead=lead, p=p, dilations=dilations)
    elif model == 'ACNN_SE':
        return ACNN_SE(aspp_bn=aspp_bn, aspp_act=aspp_act,
                       lead=lead, p=p, dilations=dilations)
    elif model == 'ACNN_deep':
        return ACNN_deep(aspp_bn=aspp_bn, aspp_act=aspp_act,
                         lead=lead, p=p, dilations=dilations)
    elif model == 'ACNN_SE_deep':
        return ACNN_SE_deep(aspp_bn=aspp_bn, aspp_act=aspp_act,
                            lead=lead, p=p, dilations=dilations)
    elif model == 'MACNN_SE':
        return MACNN_SE(aspp_bn=aspp_bn, aspp_act=aspp_act,
                        lead=lead, p=p, dilations=dilations,
                        act_func=act_func, f_act_func=f_act_func)
    elif model == 'MACNN_ATT':
        return MACNN_ATT(aspp_bn=aspp_bn, aspp_act=aspp_act,
                         lead=lead, p=p, dilations=dilations,
                         act_func=act_func, f_act_func=f_act_func)
    elif model == 'MACNN_MATT':
        return MACNN_MATT(aspp_bn=aspp_bn, aspp_act=aspp_act,
                          lead=lead, p=p, dilations=dilations,
                          act_func=act_func, f_act_func=f_act_func)
    else:
        print('No available network')
        raise ValueError


def build_cnn_models(model, fixed_len, p):

    if model == 'CNN':
        return CNN(fixed_len=fixed_len, p=p)
    elif model == 'CNN_ATT':
        return CNN_ATT(fixed_len=fixed_len, p=p)
    else:
        print('No available network')
        raise ValueError


def build_lstm_models(model, fixed_len):

    if model == 'BiLSTM':
        return BiLSTM(fixed_len=fixed_len)
    else:
        print('No available network')
        raise ValueError
