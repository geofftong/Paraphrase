from __future__ import print_function

import os
import random

import numpy
import torch
import torch.nn as nn
import torch.optim as optim
from util.dataset import QuoraDataset
from util.eval import remove_dirs
from util.metrics import Metrics
from util.utils import load_word_vectors, build_vocab
from util.vocab import Vocab

from model import *
from config import parse_args
from trainer import Trainer


# MAIN BLOCK
def main():
    global args
    args = parse_args()
    args.input_dim, args.mem_dim = 200, 150
    args.hidden_dim, args.num_classes = 50, 2
    args.cuda = args.cuda and torch.cuda.is_available()
    if args.sparse and args.wd != 0:
        print('Sparsity and weight decay are incompatible!')
        exit()
    print(args)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    numpy.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    word_vectors_path = os.path.join(args.glove, '/data1/qspace/yananlu/embedding/huge.readable')
    train_dir = os.path.join(args.data, 'train/')
    dev_dir = os.path.join(args.data, 'dev/')
    test_dir = os.path.join(args.data, 'test/')

    # write unique words from all token files
    xianer_vocab_file = os.path.join(args.data, 'xianer.vocab')
    if not os.path.isfile(xianer_vocab_file):
        token_files_a = [os.path.join(split, 'a.toks') for split in [train_dir, dev_dir, test_dir]]
        token_files_b = [os.path.join(split, 'b.toks') for split in [train_dir, dev_dir, test_dir]]
        token_files = token_files_a + token_files_b
        xianer_vocab_file = os.path.join(args.data, 'xianer.vocab')
        build_vocab(token_files, xianer_vocab_file)

    # get vocab object from vocab file previously written
    vocab = Vocab(filename=xianer_vocab_file,
                  data=[config.PAD_WORD, config.UNK_WORD, config.BOS_WORD, config.EOS_WORD])
    print('==> Xianer vocabulary size : %d ' % vocab.size())

    # load Xianer dataset splits
    train_file = os.path.join(args.data, 'xianer_train.pth')  # quora_train.pth
    if os.path.isfile(train_file):
        train_dataset = torch.load(train_file)
    else:
        train_dataset = QuoraDataset(train_dir, vocab, args.num_classes)
        torch.save(train_dataset, train_file)
    print('==> Size of train data   : %d ' % len(train_dataset))

    dev_file = os.path.join(args.data, 'xianer_dev.pth')
    if os.path.isfile(dev_file):
        dev_dataset = torch.load(dev_file)
    else:
        dev_dataset = QuoraDataset(dev_dir, vocab, args.num_classes)
        torch.save(dev_dataset, dev_file)
    print('==> Size of dev data     : %d ' % len(dev_dataset))

    test_file = os.path.join(args.data, 'xianer_test.pth')
    if os.path.isfile(test_file):
        test_dataset = torch.load(test_file)
    else:
        test_dataset = QuoraDataset(test_dir, vocab, args.num_classes)
        torch.save(test_dataset, test_file)
    print('==> Size of test data    : %d ' % len(test_dataset))

    # initialize model, criterion/loss_function, optimizer
    model = SimilarityTreeLSTM(
        args.cuda, vocab,  # vocab.size()
        args.input_dim, args.mem_dim,
        args.hidden_dim, args.num_classes,
        args.sparse)
    # criterion = nn.KLDivLoss()
    criterion = nn.CrossEntropyLoss()  # nn.MSELoss()
    if args.cuda:
        model.cuda(), criterion.cuda()
    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd)
    metrics = Metrics(args.num_classes)

    # for words common to dataset vocab and GLOVE, use GLOVE vectors
    # for other words in dataset vocab, use random normal vectors
    emb_file = os.path.join(args.data, 'xianer_embed.pth')
    if os.path.isfile(emb_file):
        emb = torch.load(emb_file)
    else:  # load glove embeddings and vocab
        glove_vocab, glove_emb = load_word_vectors(word_vectors_path)
        print('==> GLOVE vocabulary size: %d ' % glove_vocab.size())
        emb = torch.Tensor(vocab.size(), glove_emb.size(1)).normal_(-0.05, 0.05)
        # zero out the embeddings for padding and other special words if they are absent in vocab
        for idx, item in enumerate([config.PAD_WORD, config.UNK_WORD, config.BOS_WORD, config.EOS_WORD]):
            emb[idx].zero_()
        for word in vocab.labelToIdx.keys():
            if glove_vocab.getIndex(word):
                emb[vocab.getIndex(word)] = glove_emb[glove_vocab.getIndex(word)]
        torch.save(emb, emb_file)
    # plug these into embedding matrix inside model
    if args.cuda:
        emb = emb.cuda()
    model.childsumtreelstm.emb.state_dict()['weight'].copy_(emb)

    # create trainer object for training and testing
    trainer = Trainer(args, model, criterion, optimizer)
    best = -float('inf')
    for epoch in range(args.epochs):
        _ = trainer.train(train_dataset)

        train_loss, train_pred, train_score = trainer.test(train_dataset, plot_flag=False)
        train_pearson = metrics.pearson(train_pred, train_dataset.labels)
        train_mse = metrics.mse(train_pred, train_dataset.labels)
        train_accu = metrics.accuracy(train_pred, train_dataset.labels)
        train_f1 = metrics.f1(train_pred, train_dataset.labels)  # GEOFF
        print(
            '==> Train Loss: {}\tPearson: {}\tMSE: {}\tAccu: {}\tF1: {}'.format(train_loss, train_pearson, train_mse,
                                                                                   train_accu, train_f1))

        dev_loss, dev_pred, dev_score = trainer.test(dev_dataset, plot_flag=False)
        dev_pearson = metrics.pearson(dev_pred, dev_dataset.labels)
        dev_mse = metrics.mse(dev_pred, dev_dataset.labels)
        dev_accu = metrics.accuracy(dev_pred, dev_dataset.labels)
        dev_f1 = metrics.f1(dev_pred, dev_dataset.labels)
        print('==> Dev Loss: {}\tPearson: {}\tMSE: {}\tAccu: {}\tF1: {}'.format(dev_loss, dev_pearson, dev_mse,
                                                                                     dev_accu, dev_f1))

        test_loss, test_pred, test_score = trainer.test(test_dataset, plot_flag=False)
        test_pearson = metrics.pearson(test_pred, test_dataset.labels)
        test_mse = metrics.mse(test_pred, test_dataset.labels)
        test_accu = metrics.accuracy(test_pred, test_dataset.labels)
        test_f1 = metrics.f1(test_pred, test_dataset.labels)
        print('==> Test Loss: {}\tPearson: {}\tMSE: {}\tAccu: {}\tF1: {}'.format(test_loss, test_pearson, test_mse,
                                                                                     test_accu, test_f1))

        if best < dev_f1:  # xianer use dev
            best = dev_f1
            checkpoint = {'model': trainer.model.state_dict(), 'optim': trainer.optimizer,
                          'pearson': dev_pearson, 'mse': dev_mse, 'f1': dev_f1, 'accuracy': dev_accu,
                          'args': args, 'epoch': epoch}
            print('==> New optimum found.')
            torch.save(checkpoint, '%s.pt' % os.path.join(args.save, args.expname + '.pth'))
            remove_dirs()  # clear attention dir
            trainer.test(dev_dataset, plot_flag=True)  # plot attention
            numpy.savetxt(os.path.join(args.data, 'dev/dev.predict'), dev_score.cpu().numpy())  # save predict result
            numpy.savetxt(os.path.join(args.data, 'train/train.predict'), train_score.cpu().numpy())
            numpy.savetxt(os.path.join(args.data, 'test/test.predict'), test_score.cpu().numpy())

if __name__ == "__main__":
    main()
