import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

'''
Evaluation metrics:

Acc = (TP + TN) / (TP + TN + FP + FN)
Se = TP / (TP + FN)
Sp = TN / (TN + FP)
Pp = TP / (TP + FP)

'''


class Eval(object):

    def __init__(self, num_class=4):

        self.num_class = num_class

    def _metrics(self, predictions, labels):

        '''
        :param predictions: A numpy array
        :param labels: A numpy array
        :return: Evaluation results in a dictionary format
        '''

        preds_n = np.where(predictions == 0, 1, 0)
        labels_n = np.where(labels == 0, 1, 0)

        preds_v = np.where(predictions == 1, 1, 0)
        labels_v = np.where(labels == 1, 1, 0)

        preds_s = np.where(predictions == 2, 1, 0)
        labels_s = np.where(labels == 2, 1, 0)

        preds_f = np.where(predictions == 3, 1, 0)
        labels_f = np.where(labels == 3, 1, 0)

        '''VEB'''
        fv = np.sum(np.where((preds_f + labels_v) == 2, 1, 0))

        s_v = preds_v + labels_v
        tn_v = np.sum(np.where(s_v == 0, 1, 0))
        tp_v = np.sum(np.where(s_v == 2, 1, 0))
        m_v = preds_v - labels_v
        fn_v = np.sum(np.where(m_v == 1, 1, 0))
        fp_v = np.sum(np.where(m_v == -1, 1, 0)) - fv

        Se_v = tp_v / (tp_v + fn_v)
        Pp_v = tp_v / (tp_v + fp_v)
        FPR_v = fp_v / (tn_v + fp_v)
        Acc_v = (tp_v + tn_v) / (tp_v + tn_v + fp_v + fn_v)

        '''SVEB'''

        s_s = preds_s + labels_s
        tn_s = np.sum(np.where(s_s == 0, 1, 0))
        tp_s = np.sum(np.where(s_s == 2, 1, 0))
        m_s = preds_s - labels_s
        fn_s = np.sum(np.where(m_s == 1, 1, 0))
        fp_s = np.sum(np.where(m_s == -1, 1, 0))

        Se_s = tp_s / (tp_s + fn_s)
        Pp_s = tp_s / (tp_s + fp_s)
        FPR_s = fp_s / (tn_s + fp_s)
        Acc_s = (tp_s + tn_s) / (tp_s + tn_s + fp_s + fn_s)

        '''Normal'''

        s_n = preds_n + labels_n
        s_f = preds_f + labels_f

        tn_n = np.sum(np.where(s_n == 2, 1, 0))
        tp_f = np.sum(np.where(s_f == 2, 1, 0))

        Sp = tn_n / np.sum(preds_n)

        Se_f = tp_f / np.sum(preds_f)

        Acc = (tn_n + tp_s + tp_v + tp_f) / len(labels)

        eval_results = {
            'VEB': {'Se': Se_v,
                    'Pp': Pp_v,
                    'FPR': FPR_v,
                    'Acc': Acc_v},

            'SVEB': {'Se': Se_s,
                     'Pp': Pp_s,
                     'FPR': FPR_s,
                     'Acc': Acc_s},

            'F': {'Se': Se_f},
            'Sp': Sp,
            'Acc': Acc
        }

        return eval_results

    def _confusion_matrix(self, y_pred, y_label):

        return confusion_matrix(y_true=y_label, y_pred=y_pred)

    def _f1_score(self, y_pred, y_true):

        return f1_score(y_true=y_true, y_pred=y_pred, average=None)

    def _sklean_metrics(self, y_pred, y_label):

        precisions = precision_score(y_pred=y_pred, y_true=y_label, average=None)
        recalls = recall_score(y_pred=y_pred, y_true=y_label, average=None)

        Pp = {'N': precisions[0], 'V': precisions[1], 'S': precisions[2], 'F': precisions[3]}
        Se = {'N': recalls[0], 'V': recalls[1], 'S': recalls[2], 'F': recalls[3]}

        return Pp, Se


class CorrelationAnalysis(object):

    def __init__(self, w):

        self.w = w
        self.omega = self._calculate_omega()
        self.po = self._calculate_correlation_matrix()

    def _calculate_omega(self):

        if not type(self.w) is np.ndarray:
            print('The weight must be numpy array')
            raise ValueError

        d = self.w.shape[0]

        omega = d * np.linalg.inv(np.matmul(self.w.transpose(), self.w))

        return omega

    def _calculate_correlation_matrix(self):

        po = np.zeros(shape=self.omega.shape)

        for i in range(self.omega.shape[0]):
            for j in range(self.omega.shape[1]):

                po[i][j] = -1.0 * self.omega[i][j] / (np.sqrt(self.omega[i][i] * self.omega[j][j]))

        return po


# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1 or classname.find('Transpose') != -1:
#         m.weight.data.normal_(0.0, 0.01)
#         m.bias.data.normal_(0.0, 0.01)
#     elif classname.find('BatchNorm') != -1:
#         m.weight.data.normal_(1.0, 0.01)
#         m.bias.data.fill_(0)
#     elif classname.find('Linear') != -1:
#         m.weight.data.normal_(0.0, 0.01)
#         m.bias.data.normal_(0.0, 0.01)

# def init_weights(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv1d') != -1 or classname.find('Conv2d') != -1:
#         nn.init.kaiming_uniform_(m.weight)
#         nn.init.zeros_(m.bias)
#     elif classname.find('BatchNorm') != -1:
#         nn.init.normal_(m.weight, 1.0, 0.01)
#         nn.init.zeros_(m.bias)
#     elif classname.find('Linear') != -1:
#         nn.init.xavier_normal_(m.weight)
#         nn.init.zeros_(m.bias)

def init_weights(m):

    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.01)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
