from __future__ import print_function

import os, math
import torch
from vocab import Vocab


# count the dependency tree label
def count_rels(rootdir):
    label_dict = {}
    dirlist = ['train', 'test', 'dev']
    for d in dirlist:
        sub_dir = os.path.join(rootdir, d)
        print(sub_dir)
        for filepath in os.listdir(sub_dir):
            if filepath.endswith('.rels'):
                with open(os.path.join(sub_dir, filepath)) as f1:
                    print (os.path.join(sub_dir, filepath))
                    for line in f1:
                        tokens = line.strip().split(' ')
                        for tok in tokens:
                            if not label_dict.has_key(tok):
                                label_dict[tok] = 1
                            else:
                                label_dict[tok] = label_dict.get(tok) + 1
    label_dict = sorted(label_dict.items(), key=lambda d: d[1], reverse=True)
    return label_dict


# split training data
def split_training_data(path):
    for fpath, dirs, fs in os.walk(path):
        for files in fs:
            print(os.path.join(fpath, files))
            with open(os.path.join(fpath, files)) as f:
                with open(os.path.join(fpath, files).replace("train", "train_small"), "w") as fw:
                    for idx, line in enumerate(f):
                        if idx >= 2000:
                            break
                        fw.write(line)
    # with open('data/quora/train.tsv') as f:
    #     with open('data/train_small.tsv', 'w') as fw:
    #         for idx, line in enumerate(f):
    #             if idx >= 100000:
    #                 break
    #             fw.write(line)


# loading GLOVE word vectors
def load_word_vectors(path):
    if os.path.isfile(path + '.pth') and os.path.isfile(path + '.vocab'):
        print('==> File found, loading to memory')
        vectors = torch.load(path + '.pth')
        vocab = Vocab(filename=path + '.vocab')
        return vocab, vectors
    # saved file not found, read from txt file
    # and create tensors for word vectors
    print('==> File not found, preparing, be patient')
    count = sum(1 for line in open(path))
    with open(path, 'r') as f:
        contents = f.readline().rstrip().split(' ')  # GEOFF: strip('\n').split(' ')
        dim = len(contents[1:])
    words = [None] * count
    print(count, dim)
    vectors = torch.zeros(count, dim)
    with open(path, 'r') as f:
        idx = 0
        for line in f:
            contents = line.rstrip().split(' ')  # rstrip('\n').split(' ')
            words[idx] = contents[0]
            vectors[idx] = torch.Tensor(map(float, contents[1:]))
            idx += 1
    with open(path + '.vocab', 'w') as f:
        for word in words:
            f.write(word + '\n')
    vocab = Vocab(filename=path + '.vocab')
    torch.save(vectors, path + '.pth')
    return vocab, vectors


# write unique words from a set of files to a new file
def build_vocab(filenames, vocabfile):
    vocab = set()
    for filename in filenames:
        with open(filename, 'r') as f:
            for line in f:
                tokens = line.rstrip('\n').split(' ')
                vocab |= set(tokens)
    with open(vocabfile, 'w') as f:
        for token in sorted(vocab):
            f.write(token + '\n')


# mapping from scalar to vector
def map_label_to_target(label, num_classes):
    target = torch.zeros(1, num_classes)
    ceil = int(math.ceil(label))
    floor = int(math.floor(label))
    if ceil == floor:
        target[0][floor] = 1  # floor - 1
    else:
        target[0][floor] = ceil - label
        target[0][ceil] = label - floor
    return target

if __name__ == "__main__":
    # split_training_data('data/quora/train')
    dic = count_rels('data/xianer')  # data/xianer