#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time    :   2020/12/16 20:52:36
# @Author  :   Jia Zhang 
# @Contact :   jialrs.z@gmail.com
# @File    :   nn_model.py

    
from utility import GNNConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph_attention import *
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class GATModel(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        self.dropout = dropout
        super(GATModel, self).__init__()
        self.attentions = [SpGraphAttentionLayer(nfeat, 
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads, 
                                             nclass, 
                                             dropout=dropout, 
                                             alpha=alpha, 
                                             concat=False)
        
    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))

        return F.log_softmax(x, dim=1)

class TransformerU2GNN(nn.Module):

    def __init__(self, feature_dim_size, num_classes, config: GNNConfig):
        super(TransformerU2GNN, self).__init__()
        self.feature_dim_size = feature_dim_size
        self.ff_hidden_size = config.ff_hidden_size
        self.num_classes = num_classes
        self.value_head_size = config.value_head_size
        self.num_self_att_layers = config.num_self_att_layers #Each U2GNN layer consists of a number of self-attention layers
        self.num_U2GNN_layers = config.num_U2GNN_layers
        self.dropout = config.dropout
        #
        self.u2gnn_layers = torch.nn.ModuleList()
        for _ in range(self.num_U2GNN_layers):
            encoder_layers = TransformerEncoderLayer(d_model=self.feature_dim_size, nhead=1, dim_feedforward=self.ff_hidden_size, dropout=self.dropout)
            self.u2gnn_layers.append(TransformerEncoder(encoder_layers, self.num_self_att_layers))
        # Linear function
        self.policy_predictions = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()
        self.value_dense_layers = torch.nn.ModuleList()
        self.rectifier = nn.ReLU()
        self.value_prediction = nn.Linear(self.value_head_size, 1)
        # self.predictions.append(nn.Linear(feature_dim_size, num_classes)) # For including feature vectors to predict graph labels???
        for _ in range(self.num_U2GNN_layers):
            self.policy_predictions.append(nn.Linear(self.feature_dim_size, self.num_classes))
            self.value_dense_layers.append(nn.Linear(self.feature_dim_size, self.value_head_size))
            self.dropouts.append(nn.Dropout(self.dropout))

    def forward(self, input_x, graph_pool, X_concat):
        policy_prediction_scores = 0
        value_hidden_vector = 0
        # print(input_x.size())
        # print(X_concat.size())
        # print(input_x)
        input_Tr = F.embedding(input_x, X_concat)
     
        for layer_idx in range(self.num_U2GNN_layers):
            #print(input_Tr.shape)
            output_Tr = self.u2gnn_layers[layer_idx](input_Tr)
            output_Tr = torch.split(output_Tr, split_size_or_sections=1, dim=1)[0]
            output_Tr = torch.squeeze(output_Tr, dim=1)
            #new input for next layer
            input_Tr = F.embedding(input_x, output_Tr)
            #sum pooling
            graph_embeddings = torch.spmm(graph_pool, output_Tr)
            graph_embeddings = self.dropouts[layer_idx](graph_embeddings)
            # Produce the final scores
            policy_prediction_scores += self.policy_predictions[layer_idx](graph_embeddings)
            value_hidden_vector += self.value_dense_layers[layer_idx](graph_embeddings)
        
        value_prediction_scores = self.value_prediction(self.rectifier(value_hidden_vector))
       
        return policy_prediction_scores, value_prediction_scores


class FixedGraphAttentionLayer(nn.Module):
    """
    GAT layer with fixed size and degree, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(FixedGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
                
        self.a = nn.Parameter(torch.zeros(size=(2*out_features,)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.softmax = nn.Softmax(dim = 2)

    def forward(self, input):
        #dv = 'cuda' if input.is_cuda else 'cpu'
         
        bs, N, D, _ = input.size()
        #bs: batch_size, N: number of nodes, degree: max degree of the node with self-loop 
        h_prime = torch.matmul(input, self.W)
        #h_prime: bs x N x D x out 
        h_node = torch.repeat_interleave(h_prime[:,:,0:1,:], D, dim = 2)
        h_edge = torch.cat([h_prime, h_node], dim = 3)
        #h_prime: bs x N x D x out 
        edge_e = self.softmax(self.leakyrelu(torch.matmul(h_edge, self.a)))
        edge_e = self.dropout(edge_e[:,:,None,:])
        #edge: bs x N x 1 x D
        h_prime = torch.matmul(edge_e, h_prime).squeeze(dim = 2)
        #h_prime: bs x N x out

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class DGATModel(nn.Module):
    def __init__(self, nfeat, nclass, config: GNNConfig ):
        """Fixed degree and fixed size graph attention """
        nlayer, nghid, nglayer, dropout, alpha, nheads, ngheads, nhid =  \
            config.nlayer, config.nghid, config.dropout, config.alpha, config.nheads, config.ngheads, config.nhid 
    
        super(GATModel, self).__init__()
        self.graph_attentions = torch.nn.ModuleList()
        for _ in range(nlayer):
            self.graph_attentions.append([FixedGraphAttentionLayer(nfeat, 
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True) for _ in range(nheads)])

        for i, attention in enumerate(self.graph_attentions):
            for j, head in enumerate(attention):
                self.add_module('attention_head_{}'.format(i,j), head)

        encoder_layers = TransformerEncoderLayer(d_model = nhid, nhead = ngheads, dim_feedforward = nghid, dropout=dropout)
        self.encoder = TransformerEncoder(encoder_layers, nglayer)
        self.policy_predictions = nn.Linear(nghid, nclass)
        self.log_softmax = nn.LogSoftmax(dim = 1)

    def forward(self, adj, embedding):
        policy_prediction_scores = 0
        value_hidden_vector = 0
        
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))

        return F.log_softmax(x, dim=1)


class SequenceModel(nn.Module):
    def __init__(self, n_qubits , n_class, config: GNNConfig):
        super(SequenceModel, self).__init__()
        d_model, n_head, n_layer, dim_feedforward, dropout, n_encoder = config.d_model, config.n_head, config.n_layer, config.dim_feedforward, config.dropout, config.n_encoder
        self.embedding = nn.Embedding(num_embeddings = n_qubits+2, embedding_dim = d_model, padding_idx = 1)
        #nn.init.xavier_normal_(self.embedding.weight, gain=1.414)

        torch.nn.ModuleList()

        encoder_layers = TransformerEncoderLayer(d_model = d_model, nhead = n_head, dim_feedforward = dim_feedforward, dropout = dropout)
        self.encoder = TransformerEncoder(encoder_layers, n_layer)

        self.policy_prediction = nn.Linear(d_model, n_class)
        self.dropout = nn.Dropout(dropout)
        self.rectifier = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim = 1)
        
    
    def forward(self, x):
        input_x = self.embedding(x)
        # batch_size x num_of_gates x 2 x embedding_dim
        input_x = input_x[:,:,0,:] + input_x[:,:,1,:]
        # batch_size x num_of_gates x embedding_dim

        output_Tr = self.encoder(input_x)
        output_Tr = torch.split(output_Tr, split_size_or_sections=1, dim=1)[0]
        output_Tr = torch.squeeze(output_Tr, dim=1)

        policy_prediction_score = self.policy_prediction(self.dropout(self.rectifier(output_Tr)))
        
        return policy_prediction_score 

