""" Some code borrowed from https://github.com/tkipf/pygcn."""
# 主要用于处理目标之间的关系或场景中的拓扑结构
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from utils.net_util import norm_col_init, weights_init
import scipy.sparse as sp
import numpy as np

from datasets.glove import Glove

from .model_io import ModelOutput


def normalize_adj(adj):    # 函数的目的是将图的邻接矩阵进行对称归一化
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


class GCN(torch.nn.Module):
    def __init__(self, args):
        action_space = args.action_space
        target_embedding_sz = args.glove_dim
        resnet_embedding_sz = 512
        hidden_state_sz = args.hidden_state_sz
        super(GCN, self).__init__()
                                                              # Convolutional and Pooling Layers
        self.conv1 = nn.Conv2d(resnet_embedding_sz, 64, 1)
        self.maxp1 = nn.MaxPool2d(2, 2)                        
                                                                   # Linear Layers for Embedding
        self.embed_glove = nn.Linear(target_embedding_sz, 64)      # 用于处理词向量的线性层
        self.embed_action = nn.Linear(action_space, 10)            # 用于处理动作特征的线性层

        pointwise_in_channels = 138
                                                                     # Pointwise Convolution
        self.pointwise = nn.Conv2d(pointwise_in_channels, 64, 1, 1)

        lstm_input_sz = 7 * 7 * 64 + 512

        self.hidden_state_sz = hidden_state_sz
        self.lstm = nn.LSTMCell(lstm_input_sz, hidden_state_sz)      # 用于处理时序数据的LSTMCell
        num_outputs = action_space
                                                                     # Linear Layers for Actor-Critic
        self.critic_linear = nn.Linear(hidden_state_sz, 1)
        self.actor_linear = nn.Linear(hidden_state_sz, num_outputs)

        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain("relu")
        self.conv1.weight.data.mul_(relu_gain)
        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01
        )
        self.actor_linear.bias.data.fill_(0)                     # 策略线性层的初始化
        self.critic_linear.weight.data = norm_col_init(         
            self.critic_linear.weight.data, 1.0
        )                                             # 价值线性层的初始化
        self.critic_linear.bias.data.fill_(0)
                                            # LSTM的初始化
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.action_predict_linear = nn.Linear(2 * lstm_input_sz, action_space)

        self.dropout = nn.Dropout(p=args.dropout_rate)

        n = 83
        self.n = n
                                                   # 图卷积
        # get and normalize adjacency matrix.
        A_raw = torch.load("./data/gcn/adjmat.dat")         # 加载原始邻接矩阵
        A = normalize_adj(A_raw).tocsr().toarray()          # 规范化邻接矩阵
        self.A = torch.nn.Parameter(torch.Tensor(A))        # 将邻接矩阵作为可训练参数

        # last layer of resnet18.
        resnet18 = models.resnet18(pretrained=True)          # ResNet18 作为特征提取网络被加载
        modules = list(resnet18.children())[-2:]           # 取ResNet18最后两层(移除了最后的全连接层，只保留了卷积层部分)
        self.resnet18 = nn.Sequential(*modules)
        for p in self.resnet18.parameters():
            p.requires_grad = False
                                                    # 词嵌入
        # glove embeddings for all the objs.
        objects = open("./data/gcn/objects.txt").readlines()
        objects = [o.strip() for o in objects]
        all_glove = torch.zeros(n, 300)
        glove = Glove(args.glove_file)
        for i in range(n):
            all_glove[i, :] = torch.Tensor(glove.glove_embeddings[objects[i]][:])

        self.all_glove = nn.Parameter(all_glove)
        self.all_glove.requires_grad = False

        self.get_word_embed = nn.Linear(300, 512)             # 从GloVe嵌入生成词向量的线性层
        self.get_class_embed = nn.Linear(1000, 512)           # 从ResNet特征生成类别嵌入的线性层

        self.W0 = nn.Linear(1024, 1024, bias=False)           # 图卷积网络的线性层
        self.W1 = nn.Linear(1024, 1024, bias=False)
        self.W2 = nn.Linear(1024, 1, bias=False)

        self.final_mapping = nn.Linear(n, 512)               # 最终映射层

    def gcn_embed(self, state):
        x = self.resnet18[0](state)
        x = x.view(x.size(0), -1)
        x = torch.sigmoid(self.resnet18[1](x))
        class_embed = self.get_class_embed(x)
        word_embed = self.get_word_embed(self.all_glove.detach())
        x = torch.cat((class_embed.repeat(self.n, 1), word_embed), dim=1)
        x = torch.mm(self.A, x)
        x = F.relu(self.W0(x))
        x = torch.mm(self.A, x)
        x = F.relu(self.W1(x))
        x = torch.mm(self.A, x)
        x = F.relu(self.W2(x))
        x = x.view(1, self.n)
        x = self.final_mapping(x)
        return x

    def embedding(self, state, target, action_probs):
        action_embedding_input = action_probs
                                                                     # GloVe 嵌入
        glove_embedding = F.relu(self.embed_glove(target))
        glove_reshaped = glove_embedding.view(1, 64, 1, 1).repeat(1, 1, 7, 7)
                                                                           # 动作嵌入
        action_embedding = F.relu(self.embed_action(action_embedding_input))
        action_reshaped = action_embedding.view(1, 10, 1, 1).repeat(1, 1, 7, 7)
                                                             # 图像嵌入
        image_embedding = F.relu(self.conv1(state))
        x = self.dropout(image_embedding)
                                                                    # 将图像特征、GloVe 嵌入和动作嵌入拼接在一起，并通过 1x1 卷积层（self.pointwise）进行处理
        x = torch.cat((x, glove_reshaped, action_reshaped), dim=1)
        x = F.relu(self.pointwise(x))
        x = self.dropout(x)
                                           # Flatten the tensor
        out = x.view(x.size(0), -1)
                                                     # 获得 GCN 嵌入，并与之前的特征拼接
        out = torch.cat((out, self.gcn_embed(state)), dim=1)

        return out, image_embedding                 # out 被用作 LSTM 的输入

    def a3clstm(self, embedding, prev_hidden):
        hx, cx = self.lstm(embedding, prev_hidden)
        x = hx
        actor_out = self.actor_linear(x)
        critic_out = self.critic_linear(x)
        return actor_out, critic_out, (hx, cx)

    def forward(self, model_input, model_options):
        state = model_input.state
        (hx, cx) = model_input.hidden                    # LSTM 的隐藏状态和细胞状态，分别代表 LSTM 的短期记忆和长期记忆

        target = model_input.target_class_embedding           # 目标的类别嵌入（使用 GloVe 嵌入表示）
        action_probs = model_input.action_probs
        x, image_embedding = self.embedding(state, target, action_probs)            # embedding 函数将状态、目标和动作特征结合起来，生成一个用于 LSTM 的嵌入向量 x       
        actor_out, critic_out, (hx, cx) = self.a3clstm(x, (hx, cx))                  # out（即 x）传递给 a3clstm 函数，用于计算策略输出和价值输出

        return ModelOutput(
            value=critic_out,
            logit=actor_out,
            hidden=(hx, cx),
            embedding=image_embedding,
        )
