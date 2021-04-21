import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var
from util.eval import plot_attention, plot_attention2
import config


# optional: bi-tree-lstm
# module for childsumtreelstm: use for dependence tree
class ChildSumTreeLSTM(nn.Module):
    def __init__(self, cuda, vocab_size, in_dim, mem_dim, sparsity):
        super(ChildSumTreeLSTM, self).__init__()
        self.cudaFlag = cuda
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.emb = nn.Embedding(vocab_size, in_dim, padding_idx=config.PAD, sparse=sparsity)
        self.ix = nn.Linear(self.in_dim, self.mem_dim)
        self.ih = nn.Linear(self.mem_dim, self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)
        self.ox = nn.Linear(self.in_dim, self.mem_dim)
        self.oh = nn.Linear(self.mem_dim, self.mem_dim)
        self.ux = nn.Linear(self.in_dim, self.mem_dim)
        self.uh = nn.Linear(self.mem_dim, self.mem_dim)

    def node_forward(self, inputs, child_c, child_h):
        child_h_sum = F.torch.sum(torch.squeeze(child_h, 1), 0)
        i = F.sigmoid(self.ix(inputs) + self.ih(child_h_sum))
        o = F.sigmoid(self.ox(inputs) + self.oh(child_h_sum))
        u = F.tanh(self.ux(inputs) + self.uh(child_h_sum))
        # add extra singleton dimension
        fx = F.torch.unsqueeze(self.fx(inputs), 1)
        f = F.torch.cat([self.fh(child_hi) + fx for child_hi in child_h], 0)
        f = F.sigmoid(f)
        # removing extra singleton dimension
        # f = F.torch.unsqueeze(f, 1)  # pyTorch 0.2
        fc = F.torch.squeeze(F.torch.mul(f, child_c), 1)
        c = F.torch.mul(i, u) + F.torch.sum(fc, 0)
        h = F.torch.mul(o, F.tanh(c))
        return c, h

    def forward(self, tree, inputs, outputs, idxs):
        if self.cudaFlag:
            outputs = outputs.cuda()
        embs = F.torch.unsqueeze(self.emb(inputs), 1)
        for idx in xrange(tree.num_children):
            outputs, idxs = self.forward(tree.children[idx], inputs, outputs, idxs)
        child_c, child_h = self.get_child_states(tree)
        tree.state = self.node_forward(embs[tree.idx], child_c, child_h)
        outputs = F.torch.cat((outputs, tree.state[0]), 0)
        idxs.append(tree.idx)
        # print tree.size(), tree.idx, outputs[1:, ].size()
        return outputs, idxs

    def get_child_states(self, tree):
        if tree.num_children == 0:
            child_c = Var(torch.zeros(1, 1, self.mem_dim))
            child_h = Var(torch.zeros(1, 1, self.mem_dim))
            if self.cudaFlag:
                child_c, child_h = child_c.cuda(), child_h.cuda()
        else:
            child_c = Var(torch.Tensor(tree.num_children, 1, self.mem_dim))
            child_h = Var(torch.Tensor(tree.num_children, 1, self.mem_dim))
            if self.cudaFlag:
                child_c, child_h = child_c.cuda(), child_h.cuda()
            for idx in xrange(tree.num_children):
                child_c[idx], child_h[idx] = tree.children[idx].state
        return child_c, child_h


# distance-angle similarity module
class Similarity(nn.Module):
    def __init__(self, cuda, mem_dim, hidden_dim, num_classes):
        super(Similarity, self).__init__()
        self.cudaFlag = cuda
        self.mem_dim = mem_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.wh = nn.Linear(4 * self.mem_dim, self.hidden_dim)  # 6
        self.wp = nn.Linear(self.hidden_dim, self.num_classes)

    # def forward(self, lvec, rvec):  # cosine
    #     # cat_vec = F.torch.cat((lvec, rvec), 1)
    #     mult_dist = F.torch.mul(lvec, rvec)
    #     abs_dist = F.torch.abs(F.torch.add(lvec, -rvec))
    #     vec_dist = F.torch.cat((mult_dist, abs_dist), 1)
    #     out = self.wh(vec_dist)  # F.sigmoid(self.wh(vec_dist))
    #     # out = F.log_softmax(self.wp(out))
    #     # out = F.softmax(self.wp(out))
    #     return out

    def forward(self, vec):  # mlp
        # cat_vec = F.torch.cat((lvec, rvec), 1)
        out = F.relu(self.wh(vec))  # sigmoid
        # out = F.log_softmax(self.wp(out))  # KL loss
        # out = F.softmax(self.wp(out)) # MSE loss
        out = self.wp(out)  # CrossEntropy Loss
        return out


# bilinear similarity module
class Similarity2(nn.Module):
    def __init__(self, cuda, mem_dim, num_classes):
        super(Similarity2, self).__init__()
        self.cuda = cuda
        self.mem_dim = mem_dim
        self.num_classes = num_classes
        self.bilinear = nn.Bilinear(self.mem_dim, self.mem_dim, self.mem_dim)
        self.wp = nn.Linear(self.mem_dim, self.num_classes)

    def forward(self, query_state, target_state):
        output = F.relu(self.bilinear(query_state, target_state))
        output = F.log_softmax(self.wp(output))
        return output


class Attention(nn.Module):
    def __init__(self, cuda, mem_dim):
        super(Attention, self).__init__()
        self.cudaFlag = cuda
        self.mem_dim = mem_dim
        self.WQ = nn.Linear(mem_dim, mem_dim)  # bias=False
        # self.WV = nn.Linear(mem_dim, mem_dim)
        # self.WP = nn.Linear(mem_dim, 1)

    def forward(self, query_h, doc_h):
        # doc_h = torch.unsqueeze(doc_h, 0)
        # ha = F.tanh(self.WQ(query_h) + self.WV(doc_h).expand_as(query_h))  # tan(W1a + W2b)
        # p = F.softmax(self.WP(ha).squeeze())
        # weighted = p.unsqueeze(1).expand_as(
        #     query_h) * query_h
        # v = weighted.sum(dim=0)
        p = F.softmax(torch.transpose(torch.mm(query_h, doc_h.unsqueeze(1)), 0, 1))  # dot
        weighted = torch.transpose(p, 0, 1).expand_as(query_h) * query_h
        v = weighted.sum(dim=0)
        return v, p


class SelfAttention(nn.Module):
    def __init__(self, cuda, mem_dim, attention_unit=300, attention_hops=1):
        super(SelfAttention, self).__init__()
        self.cudaflag = cuda
        self.ws1 = nn.Linear(mem_dim, attention_unit, bias=False)
        self.ws2 = nn.Linear(attention_unit, attention_hops, bias=False)
        self.tanh = nn.Tanh()
        self.attention_hops = attention_hops

    def forward(self, inputs):  # (batch=1)10, 150
        hbar = self.tanh(self.ws1(inputs))  # [10, 300]
        h2 = self.ws2(hbar)
        # alphas = F.softmax(h2)
        alphas = F.softmax(torch.transpose(h2, 0, 1))  # [10, 1]
        # print alphas
        # return torch.mm(torch.transpose(alphas, 0, 1), inputs)
        return alphas


# extract feature of node embedding and cross attention embedding
class Compare(nn.Module):
    def __init__(self, cuda, mem_dim, comp_type="sub_mul_linear"):
        super(Compare, self).__init__()
        self.cudaFlag = cuda
        self.mem_dim = mem_dim
        self.comp_type = comp_type
        self.Wt = nn.Linear(2 * self.mem_dim, self.mem_dim)
        self.bilinear = nn.Bilinear(self.mem_dim, self.mem_dim, self.mem_dim)

    def forward(self, at, ht):  # (mem_dim,)
        if self.comp_type == "linear":
            t = F.torch.cat((at, ht), 0)
            output = F.tanh(self.Wt(t))
        elif self.comp_type == "bilinear":
            at = torch.unsqueeze(at, 0)
            ht = torch.unsqueeze(ht, 0)
            output = F.relu(self.bilinear(at, ht))  # N=1, mem_dim=150
        elif self.comp_type == "sub":
            abs_dist = torch.abs(F.torch.add(at, -ht))
            output = torch.mul(abs_dist, abs_dist)  # mem_dim=150
        elif self.comp_type == "mul":
            output = torch.mul(at, ht)
        else:  # "sub_mul_linear"
            abs_dist = torch.abs(F.torch.add(at, -ht))
            t = F.torch.cat((torch.mul(abs_dist, abs_dist), torch.mul(at, ht)), 0)
            output = F.tanh(self.Wt(t))
        # t = F.torch.cat((At, ht), 1)
        return output


class CnnLayer(nn.Module):
    def __init__(self, cuda, mem_dim, num_filters=50):
        super(CnnLayer, self).__init__()
        self.cudaFlag = cuda
        self.num_filters = num_filters
        self.mem_dim = mem_dim
        self.encoders = []
        self.filter_sizes = [1, 2, 3]  # nn.Sequential()
        self.dropout = nn.Dropout(p=0.5)
        for i, filter_size in enumerate(self.filter_sizes):
            enc_attr_name = "encoder_%d" % i
            self.__setattr__(enc_attr_name,
                             nn.Conv2d(in_channels=1,
                                       out_channels=self.num_filters,
                                       kernel_size=(filter_size, self.mem_dim), stride=1))
            self.encoders.append(self.__getattr__(enc_attr_name))

    def forward(self, inputs):  # TODO
        while inputs.size(2) < 3:  # (N=1, Ci=1, W, D)
            pad = Var(torch.zeros(inputs.size()))
            if self.cudaFlag:
                pad = pad.cuda()
            inputs = F.torch.cat((inputs, pad), 2)
        enc_outs = []
        for encoder in self.encoders:
            enc_ = F.relu(encoder(inputs))  # 1, 50, 10, 1
            enc_ = F.max_pool2d(enc_, kernel_size=(enc_.size()[2], 1))  # 1, 50, 1, 1
            enc_ = enc_.squeeze(3).squeeze(2)  # 1, 50 [N, C]
            enc_outs.append(enc_)
        encoding = torch.cat(enc_outs, 1)
        encoding = self.dropout(encoding)
        return encoding


class SimilarityTreeLSTM(nn.Module):
    def __init__(self, cuda, vocab, in_dim, mem_dim, hidden_dim, num_classes, sparsity):
        super(SimilarityTreeLSTM, self).__init__()
        self.cudaFlag = cuda
        self.mem_dim = mem_dim
        self.vocab = vocab
        self.childsumtreelstm = ChildSumTreeLSTM(cuda, self.vocab.size(), in_dim, mem_dim, sparsity)
        self.attention = Attention(cuda, mem_dim)
        self.self_attention = SelfAttention(cuda, mem_dim)
        self.compare = Compare(cuda, mem_dim)
        self.cnn = CnnLayer(cuda, mem_dim)
        self.similarity = Similarity(cuda, mem_dim, hidden_dim, num_classes)
        # self.similarity2 = Similarity2(cuda, mem_dim, num_classes)
        # self.childsumtreelstm2 = ChildSumTreeLSTM2(cuda, self.vocab.size(), in_dim, mem_dim, sparsity)

    def forward(self, ltree, linputs, rtree, rinputs, plot_flag):
        query_state, query_idx = self.childsumtreelstm(ltree, linputs, Var(torch.zeros(1, self.mem_dim)), [])
        target_state, target_idx = self.childsumtreelstm(rtree, rinputs, Var(torch.zeros(1, self.mem_dim)), [])
        query_state, target_state = query_state[1:, ], target_state[1:, ]
        # print query_state.size(), query_idx
        # query_root_state, target_root_state = query_state[-1, ].unsqueeze(0), target_state[-1, ].unsqueeze(0)

        # sort by tree.idx
        for i in range(len(query_idx)):
            idx = query_idx.index(i)
            query_state = F.torch.cat((query_state, query_state[idx].unsqueeze(0)), 0)
        for i in range(len(target_idx)):
            idx = target_idx.index(i)
            target_state = F.torch.cat((target_state, target_state[idx].unsqueeze(0)), 0)
        query_state, target_state = query_state[len(query_idx):, ], target_state[len(target_idx):, ]

        # self attention
        query_self_alphas = self.self_attention(query_state)
        target_self_alphas = self.self_attention(target_state)
        query_self_attn = torch.mm(query_self_alphas, query_state)  # torch.transpose(query_self_alphas, 0, 1)
        target_self_attn = torch.mm(target_self_alphas, target_state)
        # return self.similarity(query_self_attn, target_self_attn)

        # cross attention
        query_cross_alphas = Var(torch.Tensor(query_state.size(0), target_state.size(0)))
        target_cross_alphas = Var(torch.Tensor(target_state.size(0), query_state.size(0)))
        q_to_t = Var(torch.Tensor(query_state.size(0), self.mem_dim))
        t_to_q = Var(torch.Tensor(target_state.size(0), self.mem_dim))
        if self.cudaFlag:
            q_to_t, t_to_q = q_to_t.cuda(), t_to_q.cuda()
            query_cross_alphas, target_cross_alphas = query_cross_alphas.cuda(), target_cross_alphas.cuda()
        for i in range(query_state.size(0)):
            q_to_t[i], query_cross_alphas[i] = self.attention(target_state, query_state[i, ])
        for i in range(target_state.size(0)):
            t_to_q[i], target_cross_alphas[i] = self.attention(query_state, target_state[i, ])

        # query_cross_sub = torch.abs(query_state - q_to_t)
        # query_cross_mul = query_state * q_to_t
        # target_cross_sub = torch.abs(target_state - t_to_q)
        # target_cross_mul = target_state * t_to_q
        # query_self_alphas2 = self.self_attention(query_cross_sub)
        # target_self_alphas2 = self.self_attention(target_cross_sub)
        # h1 = torch.mm(query_self_alphas2, query_cross_sub)
        # h2 = torch.mm(query_self_alphas2, query_cross_mul)
        # h3 = torch.mm(target_self_alphas2, target_cross_sub)  # share h1 weight ?
        # h4 = torch.mm(target_self_alphas2, target_cross_mul)
        h5 = F.torch.abs(F.torch.add(query_self_attn, - target_self_attn))
        h6 = F.torch.mul(query_self_attn, target_self_attn)

        # h5 = F.torch.abs(F.torch.add(query_root_state, - target_root_state))
        # h6 = F.torch.mul(query_root_state, target_root_state)
        # output = self.similarity(F.torch.cat((h1, h2, h3, h4, h5, h6), 1))

        # compare
        query_com = Var(torch.Tensor(query_state.size(0), self.mem_dim))
        target_com = Var(torch.Tensor(target_state.size(0), self.mem_dim))
        if self.cudaFlag:
            query_com, target_com = query_com.cuda(), target_com.cuda()
        for i in range(query_state.size(0)):
            query_com[i] = self.compare(query_state[i], q_to_t[i])
        for i in range(target_com.size(0)):
            target_com[i] = self.compare(target_state[i], t_to_q[i])

        # cnn
        query_com = torch.unsqueeze(torch.unsqueeze(query_com, 0), 1)  # (N=1, Ci=1, W, D)
        target_com = torch.unsqueeze(torch.unsqueeze(target_com, 0), 1)
        # print query_com.size(), target_com.size()
        query_cnn_output = self.cnn(query_com)
        target_cnn_output = self.cnn(target_com)
        output = self.similarity(F.torch.cat((h5, h6, query_cnn_output, target_cnn_output), 1))

        # tree lstm: return root attention state
        # query_tree_output = self.childsumtreelstm2(ltree, linputs, query_com)[0]
        # target_tree_output = self.childsumtreelstm2(rtree, rinputs, target_com)[0]
        # output = self.similarity(F.torch.cat((h5, h6, query_tree_output, target_tree_output), 1))

        if plot_flag:
            linputs_words = self.vocab.convertToLabels(linputs.data.cpu().numpy(), len(linputs.data.cpu().numpy()))
            rinputs_words = self.vocab.convertToLabels(rinputs.data.cpu().numpy(), len(rinputs.data.cpu().numpy()))
            plot_attention(query_cross_alphas.data.cpu().numpy(), rinputs_words, linputs_words)
            plot_attention(target_cross_alphas.data.cpu().numpy(), linputs_words, rinputs_words)
            plot_attention2(query_self_alphas.data.cpu().numpy(), linputs_words)
            plot_attention2(target_self_alphas.data.cpu().numpy(), rinputs_words)
        return output

