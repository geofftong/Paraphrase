import argparse

# Constants
PAD, UNK, BOS, EOS = 0, 1, 2, 3
PAD_WORD, UNK_WORD, BOS_WORD, EOS_WORD = '<blank>', '<unk>', '<s>', '</s>'


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch TreeLSTM for Sentence Similarity on Dependency Trees')
    parser.add_argument('--data', default='../data/xianer/',
                        help='path to dataset')
    parser.add_argument('--glove', default='../data/glove/',
                        help='directory with GLOVE embeddings')
    parser.add_argument('--save', default='../checkpoints/',
                        help='directory to save checkpoints in')
    parser.add_argument('--expname', type=str, default='test',
                        help='Name to identify experiment')
    parser.add_argument('--batchsize', default=32, type=int,
                        help='batchsize for optimizer updates')
    parser.add_argument('--epochs', default=15, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--lr', default=0.01, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--wd', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--sparse', action='store_true',
                        help='Enable sparsity for embeddings, \
                              incompatible with weight decay')
    parser.add_argument('--optim', default='adagrad',
                        help='optimizer (default: adagrad)')
    parser.add_argument('--seed', default=123, type=int,
                        help='random seed (default: 123)')
    cuda_parser = parser.add_mutually_exclusive_group(required=False)
    cuda_parser.add_argument('--cuda', dest='cuda', action='store_true')
    cuda_parser.add_argument('--no-cuda', dest='cuda', action='store_false')
    parser.set_defaults(cuda=True)

    args = parser.parse_args()
    return args
