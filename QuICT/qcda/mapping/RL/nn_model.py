#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time    :   2020/12/16 20:52:36
# @Author  :   Jia Zhang 
# @Contact :   jialrs.z@gmail.com
# @File    :   nn_model.py

    
from QuICT.qcda.mapping.utility.utility import RLConfig
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import TransformerEncoder, TransformerEncoderLayer




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

    def forward(self, x: torch.Tensor, adj):
        #dv = 'cuda' if input.is_cuda else 'cpu'
        x = torch.cat([x, torch.ones(size = (x.size()[0], 1 ,x.size()[2]), dtype = torch.long, device = x.device)], dim = 1)

        bs, L, D  = adj.size()
        input = torch.zeros(size = (bs, L, D, self.in_features), dtype = x.dtype, device = x.device)

        for i in range(bs):
            input[i,:,:,:] = x[i, adj[i,:,:],:]
        #input :  bs x N x D x in
        #bs: batch_size, N: number of nodes, degree: max degree of the node with self-loop 
        h_prime = torch.matmul(input, self.W)
        #h_prime: bs x N x D x out 
        h_node = torch.repeat_interleave(h_prime[:,:,0:1,:], D, dim = 2)
        h_edge = torch.cat([h_prime, h_node], dim = 3)
        #h_prime: bs x N x D x (2*out) 
        edge_e = self.softmax(self.leakyrelu(torch.matmul(h_edge, self.a)))
        edge_e = self.dropout(edge_e.transpose(2,3))
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




class SequenceModel(nn.Module):
    def __init__(self, n_qubits , n_class, config: RLConfig):
        
        super(SequenceModel, self).__init__()
        d_embed, d_model, n_head, n_layer, dim_feedforward, dropout, n_encoder, hid_dim = config.d_embed, config.d_model, config.n_head, config.n_layer, \
                                                                                    config.dim_feedforward, config.dropout, config.n_encoder, config.hid_dim
        self.n_gat = config.n_gat
        self.gat = config.gat
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
       
        self.rectifier = nn.ReLU() 
        self.value_prediction = nn.Linear(hid_dim, 1)
        self.value_output = nn.Softplus()
        
        
    
    def forward(self, src, padding_mask, adj):

        padding_mask = torch.cat([torch.zeros(size =(padding_mask.size()[0], 2), dtype = torch.bool, device = padding_mask.device) , padding_mask], dim = 1)
        src = torch.cat([torch.zeros(size = (src.size()[0], 1 ,2), dtype = torch.long, device = src.device) , torch.ones(size = (src.size()[0], 1 ,2), dtype = torch.long, device = src.device), src + 3], dim = 1)
        adj = torch.cat([torch.zeros(size = (adj.size()[0], 1 ,adj.size()[2]), dtype = torch.long, device = adj.device) , torch.ones(size = (adj.size()[0], 1 ,adj.size()[2]), dtype = torch.long, device = adj.device) ,adj + 2], dim = 1)
        
        policy_prediction_score = 0.0
        value_prediction_score = 0.0
        value_hidden = 0.0

        input_x = self.embedding(src)
        input_x = input_x[:,:,0,:] + input_x[:,:,1,:]
        input_x  = self.rectifier(input_x)
        if self.gat:
            for gat_layer in self.gat_layer:
                input_x = gat_layer(self.rectifier(input_x), adj)

                
        input_x.transpose_(0,1)
        length, batch_size, d_model = input_x.size() 
        pe =  self._positionalencoding1d(batch_size,  d_model, length, input_x.device)
        input_x = input_x + pe
        for encoder, policy_prediction, value_hid_layer in zip(self.encoder, self.policy_prediction, self.value_hid_layer):
            output_temp = encoder(input_x,  src_key_padding_mask =  padding_mask)
           
            policy_output = torch.split(output_temp, 1, dim=0)[0]
            policy_output = torch.squeeze(policy_output, dim=0)

            value_output = torch.split(output_temp, 1, dim=0)[1]
            value_output = torch.squeeze(value_output, dim=0)

            policy_prediction_score += policy_prediction(policy_output)
            value_hidden += value_hid_layer(value_output)
            
            input_x = output_temp
        value_prediction_score = self.value_output(self.value_prediction(self.rectifier(value_hidden)))
        return policy_prediction_score, value_prediction_score 


    def _positionalencoding1d(self, batch_size: int, d_model: int, length: int, device: torch.device):
       
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """

        if d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                            "odd dim (got dim={:d})".format(d_model))

        pe = torch.zeros(size = [int(length), int(d_model)], device = device)
        position = torch.arange(0, length, device = device).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float, device = device) *
                            -(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        bpe = torch.zeros(size = [int(batch_size), int(length), int(d_model)], device = device)
        bpe[:,:,:] = pe


        return torch.transpose(bpe, 0, 1)

