from copy import deepcopy
import torch
import torch.nn as nn
from torch.autograd import Variable as Var


class Metrics():
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def pearson(self, predictions, labels):
        x = deepcopy(predictions)
        y = deepcopy(labels)
        x -= x.mean()
        x /= x.std()
        y -= y.mean()
        y /= y.std()
        return torch.mean(torch.mul(x, y))

    def mse(self, predictions, labels):
        x = Var(deepcopy(predictions), volatile=True)  # like 'requires_grad=False'
        y = Var(deepcopy(labels), volatile=True)
        return nn.MSELoss()(x, y).data[0]

    def accuracy(self, predictions, labels):
        x = deepcopy(predictions)
        y = deepcopy(labels)
        count = 0
        for idx in range(torch.numel(x)):
            if x[idx] == y[idx]:
                count += 1
        return count * 1.0 / torch.numel(x)

    def f1(self, predictions, labels):
        x = deepcopy(predictions)
        y = deepcopy(labels)
        tp, fp, fn = 0, 0, 0
        for idx in range(torch.numel(x)):
            # print x[idx], y[idx]
            if x[idx] == 1 and y[idx] == 1:
                tp += 1
            if x[idx] == 1 and y[idx] == 0:
                fp += 1
            if x[idx] == 0 and y[idx] == 1:
                fn += 1
        prec = tp * 1.0 / (tp + fp + 0.0001)
        reca = tp * 1.0 / (tp + fn + 0.0001)
        f1 = 2 * prec * reca / (prec + reca + 0.0001)
        return f1