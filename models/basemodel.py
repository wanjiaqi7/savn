# 通过CNN提取图像特征，结合目标嵌入和动作嵌入，使用LSTM处理时序信息，并最终生成策略（actor）和价值（critic）估计
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.net_util import norm_col_init, weights_init

from .model_io import ModelOutput


class BaseModel(torch.nn.Module):
    def __init__(self, args):
        action_space = args.action_space
        target_embedding_sz = args.glove_dim
        resnet_embedding_sz = args.hidden_state_sz
        hidden_state_sz = args.hidden_state_sz
        super(BaseModel, self).__init__()
                                                                   # 定义了卷积层、池化层和两个线性嵌入层
        self.conv1 = nn.Conv2d(resnet_embedding_sz, 64, 1)
        self.maxp1 = nn.MaxPool2d(2, 2)
        self.embed_glove = nn.Linear(target_embedding_sz, 64)
        self.embed_action = nn.Linear(action_space, 10)
                                                                 # 定义了一个逐点卷积层
        pointwise_in_channels = 138
        self.pointwise = nn.Conv2d(pointwise_in_channels, 64, 1, 1)
                                                                   # 定义了LSTM单元
        lstm_input_sz = 7 * 7 * 64
        self.hidden_state_sz = hidden_state_sz
        self.lstm = nn.LSTMCell(lstm_input_sz, hidden_state_sz)
                                                                     # 定义了两个线性层，一个用于价值估计（critic），另一个用于策略估计（actor）
        num_outputs = action_space
        self.critic_linear = nn.Linear(hidden_state_sz, 1)
        self.actor_linear = nn.Linear(hidden_state_sz, num_outputs)
                                                                     # 初始化各层的权重，特别是卷积层、LSTM和线性层的权重和偏置
        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain("relu")
        self.conv1.weight.data.mul_(relu_gain)
        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01
        )
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0
        )
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)
                                                                                 # 定义了一个线性层用于动作预测和一个dropout层用于正则化
        self.action_predict_linear = nn.Linear(2 * lstm_input_sz, action_space)
        self.dropout = nn.Dropout(p=args.dropout_rate)

    def embedding(self, state, target, action_probs, params):

        action_embedding_input = action_probs

        if params is None:                                          # 在没有给定参数的情况下，计算状态嵌入、目标嵌入和动作嵌入，并通过卷积层和逐点卷积层
            glove_embedding = F.relu(self.embed_glove(target))
            glove_reshaped = glove_embedding.view(1, 64, 1, 1).repeat(1, 1, 7, 7)

            action_embedding = F.relu(self.embed_action(action_embedding_input))
            action_reshaped = action_embedding.view(1, 10, 1, 1).repeat(1, 1, 7, 7)

            image_embedding = F.relu(self.conv1(state))
            x = self.dropout(image_embedding)
            x = torch.cat((x, glove_reshaped, action_reshaped), dim=1)
            x = F.relu(self.pointwise(x))
            x = self.dropout(x)
            out = x.view(x.size(0), -1)

        else:                               # 在给定参数的情况下，使用这些参数计算嵌入              
            glove_embedding = F.relu(
                F.linear(
                    target,
                    weight=params["embed_glove.weight"],
                    bias=params["embed_glove.bias"],
                )
            )

            glove_reshaped = glove_embedding.view(1, 64, 1, 1).repeat(1, 1, 7, 7)

            action_embedding = F.relu(
                F.linear(
                    action_embedding_input,
                    weight=params["embed_action.weight"],
                    bias=params["embed_action.bias"],
                )
            )
            action_reshaped = action_embedding.view(1, 10, 1, 1).repeat(1, 1, 7, 7)

            image_embedding = F.relu(
                F.conv2d(
                    state, weight=params["conv1.weight"], bias=params["conv1.bias"]
                )
            )
            x = self.dropout(image_embedding)
            x = torch.cat((x, glove_reshaped, action_reshaped), dim=1)

            x = F.relu(
                F.conv2d(
                    x, weight=params["pointwise.weight"], bias=params["pointwise.bias"]
                )
            )
            x = self.dropout(x)
            out = x.view(x.size(0), -1)

        return out, image_embedding

    def a3clstm(self, embedding, prev_hidden, params):
        if params is None:                                    # 在没有给定参数的情况下，计算LSTM的输出，并通过线性层得到策略输出和价值输出
            hx, cx = self.lstm(embedding, prev_hidden)
            x = hx
            actor_out = self.actor_linear(x)
            critic_out = self.critic_linear(x)

        else:                                              # 在给定参数的情况下，使用这些参数计算LSTM的输出
            hx, cx = self._backend.LSTMCell(
                embedding,
                prev_hidden,
                params["lstm.weight_ih"],
                params["lstm.weight_hh"],
                params["lstm.bias_ih"],
                params["lstm.bias_hh"],
            )

            # Change for pytorch 1.01
            # hx, cx = nn._VF.lstm_cell(
            #     embedding,
            #     prev_hidden,
            #     params["lstm.weight_ih"],
            #     params["lstm.weight_hh"],
            #     params["lstm.bias_ih"],
            #     params["lstm.bias_hh"],
            # )

            x = hx

            critic_out = F.linear(
                x,
                weight=params["critic_linear.weight"],
                bias=params["critic_linear.bias"],
            )
            actor_out = F.linear(
                x,
                weight=params["actor_linear.weight"],
                bias=params["actor_linear.bias"],
            )

        return actor_out, critic_out, (hx, cx)

    def forward(self, model_input, model_options):

        state = model_input.state
        (hx, cx) = model_input.hidden

        target = model_input.target_class_embedding
        action_probs = model_input.action_probs
        params = model_options.params

        x, image_embedding = self.embedding(state, target, action_probs, params)
        actor_out, critic_out, (hx, cx) = self.a3clstm(x, (hx, cx), params)

        return ModelOutput(
            value=critic_out,
            logit=actor_out,
            hidden=(hx, cx),
            embedding=image_embedding,
        )
