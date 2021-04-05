#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time    :   2020/12/16 20:52:36
# @Author  :   Jia Zhang 
# @Contact :   jialrs.z@gmail.com
# @File    :   nn_model.py

    
from utility import GNNConfig
import math
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
        input_x = torch.transpose(input_x, 0 ,1)
        # print(input_x.size())
        # print(X_concat.size())
        # print(input_x)
        input_Tr = F.embedding(input_x, X_concat)
     
        for layer_idx in range(self.num_U2GNN_layers):
            #print(input_Tr.shape)
            output_Tr = self.u2gnn_layers[layer_idx](input_Tr)
            output_Tr = torch.split(output_Tr, split_size_or_sections=1, dim=0)[0]
            output_Tr = torch.squeeze(output_Tr, dim=0)
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
                
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.softmax = nn.Softmax(dim = 2)

    def forward(self, x, adj):
        #dv = 'cuda' if input.is_cuda else 'cpu'
          
        bs, L, D  = adj.size()
        input = torch.zeros(size = (bs, L, D, self.in_features), dtype = x.dtype, device = x.device)
        # print(bs)
        # print(adj.size())
        # print(x.size())
        # print(input.size())
        for i in range(bs):
            input[i,:,:,:] = x[i, adj[i,:,:],:]
        #input :  bs x N x D x in
        #bs: batch_size, N: number of nodes, degree: max degree of the node with self-loop 
        h_prime = torch.matmul(input, self.W)
        #print(h_prime.size())
        #h_prime: bs x N x D x out 
        h_node = torch.repeat_interleave(h_prime[:,:,0:1,:], D, dim = 2)
        #print(h_node.size())
        h_edge = torch.cat([h_prime, h_node], dim = 3)
        #print(h_edge.size())
        #h_prime: bs x N x D x (2*out) 
        edge_e = self.softmax(self.leakyrelu(torch.matmul(h_edge, self.a)))
        edge_e = self.dropout(edge_e.transpose(2,3))
        #print(edge_e.size())
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
        d_embed, d_model, n_head, n_layer, dim_feedforward, dropout, n_encoder, hid_dim = config.d_embed, config.d_model, config.n_head, config.n_layer, \
                                                                                    config.dim_feedforward, config.dropout, config.n_encoder, config.hid_dim
        self.n_gat = config.n_gat
        self.gat = config.gat
        self.device = config.device
        self.n_encoder = n_encoder

        self.embedding = nn.Embedding(num_embeddings = n_qubits+3, embedding_dim = d_embed, padding_idx = 2)
        nn.init.xavier_normal_(self.embedding.weight, gain=1.414)

        self.gat_layer = torch.nn.ModuleList()
        for _ in range(self.n_gat): 
            self.gat_layer.append(FixedGraphAttentionLayer(in_features = d_embed, out_features = d_model, dropout = dropout ,alpha = 0.01))
    
        
       
        
        self.encoder = torch.nn.ModuleList()
        self.policy_prediction = torch.nn.ModuleList()
        self.value_hid_layer = torch.nn.ModuleList()
        
        self.norm = torch.nn.ModuleList()
         
    
        for _ in range(n_encoder):
            encoder_layers = TransformerEncoderLayer(d_model = d_model, nhead = n_head, dim_feedforward = dim_feedforward, dropout = dropout, activation='relu')
            self.encoder.append(TransformerEncoder(encoder_layers, n_layer))
            self.policy_prediction.append(nn.Linear(d_model, n_class))
            self.value_hid_layer.append(nn.Linear(d_model, hid_dim))
        # encoder_layers = TransformerEncoderLayer(d_model = d_model, nhead = n_head, dim_feedforward = dim_feedforward, dropout = dropout, activation='relu')
        # self.encoder = TransformerEncoder(encoder_layers, n_layer)
       
        self.rectifier = nn.ReLU() 
        self.value_prediction = nn.Linear(hid_dim, 1)
        self.value_output = nn.Softplus()
        
        
    
    def forward(self, src, padding_mask, adj):
        padding_mask = torch.cat([torch.zeros(size =(padding_mask.size()[0], 2), dtype = torch.uint8, device = self.device) , padding_mask], dim = 1)
        src = torch.cat([torch.zeros(size = (src.size()[0], 1 ,2), dtype = torch.long, device = self.device) , torch.ones(size = (src.size()[0], 1 ,2), dtype = torch.long, device = self.device) ,src + 3], dim = 1)
        adj = torch.cat([torch.zeros(size = (adj.size()[0], 1 ,adj.size()[2]), dtype = torch.long, device = self.device) , torch.ones(size = (adj.size()[0], 1 ,adj.size()[2]), dtype = torch.long, device = self.device) ,adj + 2], dim = 1)
        policy_prediction_score = 0.0
        value_prediction_score = 0.0
        value_hidden = 0.0

        input_x = self.embedding(src)
        input_x = input_x[:,:,0,:] + input_x[:,:,1,:]
        input_x  = self.rectifier(input_x)
        if self.gat:
            for i in range(self.n_gat):
                input_x = self.gat_layer[i](self.rectifier(input_x), adj)
        input_x.transpose_(0,1)
        # num_of_gates x batch_size x embedding_dim  
        length, batch_size, d_model = input_x.size() 
        pe =  self._positionalencoding1d(batch_size,  d_model, length)
        #print(pe.size())
        input_x = input_x + pe
        # num_of_gates x  batch_size xembedding_dim
        # print(input_x[0,:])
        # print(input_x[1,:])
        #print(input_x.size())
        #print(padding_mask.size())
        for i in range(self.n_encoder):
            output_temp = self.encoder[i](input_x,  src_key_padding_mask =  padding_mask)
           
            policy_output = torch.split(output_temp, split_size_or_sections=1, dim=0)[0]
            policy_output = torch.squeeze(policy_output, dim=0)

            value_output = torch.split(output_temp, split_size_or_sections=1, dim=0)[1]
            value_output = torch.squeeze(value_output, dim=0)

            policy_prediction_score += self.policy_prediction[i](policy_output)
            value_hidden += self.value_hid_layer[i](value_output)
            
            input_x = output_temp

        # print(output_Tr.size())
        # print(output_Tr[0,0,:])
        # print(output_Tr[1,0,:])

   
        # print(output_Tr.size())
        # print(output_Tr[0,:])
        # print(output_Tr[1,:])

        # print(policy_prediction_score[0])
        # print(policy_prediction_score[1])
        value_prediction_score = self.value_output(self.value_prediction(self.rectifier(value_hidden)))
        return policy_prediction_score, value_prediction_score 


    def _positionalencoding1d(self, batch_size, d_model, length):
       
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """

        if d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                            "odd dim (got dim={:d})".format(d_model))
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                            -(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        bpe = torch.zeros(size = (batch_size, length ,d_model))
        bpe[:,:,:] = pe


        return torch.transpose(bpe, 0, 1).to(self.device)

