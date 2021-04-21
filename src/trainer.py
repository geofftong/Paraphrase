import random

import torch
import torch.nn.functional as F
from torch.autograd import Variable as Var
from tqdm import tqdm


class Trainer(object):
    def __init__(self, args, model, criterion, optimizer):
        super(Trainer, self).__init__()
        self.args = args
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.epoch = 0

    def train(self, dataset):
        self.model.train()
        self.optimizer.zero_grad()
        loss, k = 0.0, 0
        indices = torch.randperm(len(dataset))
        for idx in tqdm(xrange(len(dataset)), desc='Training epoch ' + str(self.epoch + 1) + ''):
            ltree, lsent, rtree, rsent, label = dataset[indices[idx]]
            linput, rinput = Var(lsent), Var(rsent)
            # target = Var(map_label_to_target(label, dataset.num_classes))
            target = Var(torch.LongTensor([int(label)]))  # volatile=True
            if self.args.cuda:
                linput, rinput = linput.cuda(), rinput.cuda()
                target = target.cuda()
            output = self.model(ltree, linput, rtree, rinput, plot_flag=False)
            _, pred = torch.max(output.data, 1)
            # print F.softmax(output), pred
            err = self.criterion(output, target)
            loss += err.data
            err.backward()
            k += 1
            if k % self.args.batchsize == 0:
                self.optimizer.step()  # Does the update
                self.optimizer.zero_grad()  # zero the gradient buffers
        self.epoch += 1
        return loss / len(dataset)

    def test(self, dataset, plot_flag):
        self.model.eval()
        loss = 0
        predictions = torch.zeros(len(dataset))
        predict_score = torch.zeros(len(dataset), dataset.num_classes)
        # indices = torch.arange(1, dataset.num_classes + 1)
        if plot_flag:
            plot_num = 10
            for _ in tqdm(xrange(plot_num), desc='Plotting attention  ' + str(self.epoch) + ''):
                ltree, lsent, rtree, rsent, label = dataset[random.randint(0, len(dataset) - 1)]
                linput, rinput = Var(lsent, volatile=True), Var(rsent, volatile=True)
                if self.args.cuda:
                    linput, rinput = linput.cuda(), rinput.cuda()
                _ = self.model(ltree, linput, rtree, rinput, plot_flag=True)
            return
        for idx in tqdm(xrange(len(dataset)), desc='Testing epoch  ' + str(self.epoch) + ''):
            ltree, lsent, rtree, rsent, label = dataset[idx]
            linput, rinput = Var(lsent, volatile=True), Var(rsent, volatile=True)
            # target = Var(map_label_to_target(label, dataset.num_classes), volatile=True)
            target = Var(torch.LongTensor([int(label)]), volatile=True)  # CrossEntropyLoss or NLLLoss()
            if self.args.cuda:
                linput, rinput = linput.cuda(), rinput.cuda()
                target = target.cuda()
            output = self.model(ltree, linput, rtree, rinput, False)
            err = self.criterion(output, target)
            loss += err.data
            # output = torch.squeeze(output, 0)  # pyTorch 0.2: dot() need 1 dim
            # predictions[idx] = torch.dot(indices, torch.exp(output.data.cpu()))  # KL div
            # if predictions[idx] >= 1.5:
            #     predictions[idx] = 1
            # else:
            #     predictions[idx] = 0
            _, pred = torch.max(output.data, 1)
            predictions[idx] = pred[0]
            predict_score[idx] = F.softmax(output).data
            # if idx < 3:
            #     print predictions[idx], F.softmax(output)
        return loss / len(dataset), predictions, predict_score
