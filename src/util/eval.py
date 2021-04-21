#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import sys
import time
import random
import shutil
import matplotlib
matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from matplotlib.font_manager import *
# plt.rcParams['font.sans-serif'] = ['YaHei Consolas Hybrid']  # set font family
zh_font = FontProperties(fname='../img/font/yahei_consolas_hybrid.ttf')  # load chinese font for matplob


def remove_dirs(rootdir='../img/'):
    filelist = os.listdir(rootdir)
    for f in filelist:
        filepath = os.path.join(rootdir, f)
        if os.path.isfile(filepath):
            os.remove(filepath)
            # print(filepath + " removed")
        # elif os.path.isdir(filepath):
        #     shutil.rmtree(filepath, True)
            # print("dir " + filepath + " removed")


def plot_attention(data, x_label=None, y_label=None, rootdir='../img/cross_'):
    '''
      Plot the attention model heatmap
      Args:
        data: attn_matrix with shape [ty, tx], cutted before 'PAD'
        x_label: list of size tx, encoder tags
        y_label: list of size ty, decoder tags
    '''
    fig, ax = plt.subplots(figsize=(20, 8))  # set figure size
    heatmap = ax.pcolor(data, cmap=plt.cm.Blues, alpha=0.9)
    # Set axis labels
    if x_label != None and y_label != None:
        x_label = [x.decode('utf-8') for x in x_label]
        y_label = [y.decode('utf-8') for y in y_label]
        xticks = [x + 0.5 for x in range(0, len(x_label))]  # range(0, len(x_label))
        ax.set_xticks(xticks, minor=False)  # major ticks
        ax.set_xticklabels(x_label, minor=False, rotation=90, fontproperties=zh_font)  # labels should be 'unicode'
        yticks = [y + 0.5 for y in range(0, len(y_label))]  # range(0, len(y_label))
        ax.set_yticks(yticks, minor=False)
        ax.set_yticklabels(y_label, minor=False, fontproperties=zh_font)  # labels should be 'unicode'
        # ax.grid(True)
    # Save Figure
    plt.title(u'Cross Attention Heatmap')
    timestamp = int(time.time())
    file_name = rootdir + str(timestamp) + "_" + str(random.randint(0, 1000)) + ".jpg"
    fig.savefig(file_name)  # save the figure to file
    plt.close(fig)  # close the figure


# drop self attention
def plot_attention2(data, x_label=None, rootdir='../img/self_'):
    fig, ax = plt.subplots(figsize=(20, 2))
    heatmap = ax.pcolor(data, cmap=plt.cm.Blues, alpha=0.9)
    if x_label != None:
        x_label = [x.decode('utf-8') for x in x_label]
        xticks = [x + 0.5 for x in range(0, len(x_label))]
        ax.set_xticks(xticks, minor=False)
        ax.set_xticklabels(x_label, minor=False, fontproperties=zh_font)
        ax.set_yticks([0], minor=False)
        ax.set_yticklabels("", minor=False, fontproperties=zh_font)
        # ax.grid(True)
    plt.title(u'Self Attention Heatmap')
    timestamp = int(time.time())
    file_name = rootdir + str(timestamp) + "_" + str(random.randint(0, 1000)) + ".jpg"
    fig.savefig(file_name)  # save the figure to file
    plt.close(fig)  # close the figure

if __name__ == "__main__":
    # matrix = [[4, 2, 4, 3, 1, 3], [3, 6, 4, 4, 1, 1], [3, 6, 4, 4, 1, 2], [3, 6, 4, 4, 1, 2], [3, 6, 4, 4, 1, 2]]
    # query = ["你好", "boys", "hello", "boys", "hello", "Hi"]
    # target = ["Hi", "girls", "girls", "Hi", "boys"]
    # plot_attention(matrix, query, target)
    vector = [[0.1, 0.2, 0.5, 0.05, 0.1, 0.05, 0.1, 0.2, 0.5, 0.05, 0.1, 0.05]]
    query2 = ["你好", "boys", "hello", "boys", "hello", "Hi", "你好", "boys", "hello", "boys", "hello", "Hi"]
    plot_attention2(vector, query2)
    # remove_dirs('img/')