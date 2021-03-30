from typing import List, Dict, Tuple



import torch
import queue
import numpy as np

from torch.multiprocessing import Queue, Pipe

from .nn_model import *
from QuICT.qcda.mapping.utility import *

class Inference(object):
    def __init__(self, graph_name: str , config: GNNConfig, model_path: str, data_queue: Queue, data_connection_list, timeout: int = 5):
        self._timeout = timeout
        self._num_of_connection = len(data_connection_list)
        self._data_queue = data_queue
        self._data_connection_list = data_connection_list
        self._device = config.device
        self._coupling_graph = get_coupling_graph(graph_name = graph_name)
        self._config = config
        
        self._model = SequenceModel(n_qubits = self._coupling_graph.size, n_class = self._coupling_graph.num_of_edge, config = self._config).to(config.device).float()
        self._model.load_state_dict(torch.load(model_path))
        self._model.eval()



    def __call__(self, num_of_process: int = 16):
        self._num_of_live_process = num_of_process
        batch_size = num_of_process

        self._adj_list =  torch.zeros(size = (batch_size, self._config.num_of_nodes, 5), dtype = torch.long)
        self._qubits_list = torch.zeros(size = (batch_size, self._config.num_of_nodes, 2), dtype = torch.long)
        self._padding_mask_list = torch.ones(size = (batch_size, self._config.num_of_nodes), dtype = torch.uint8)
        self._id_list = torch.zeros(batch_size, dtype = torch.long) -1

        while self._num_of_live_process > 0 :
            num_of_samples  = self._get_batch_data()
            policy_scores, value_scores = self._model(self._qubits_list.to(self._device), self._padding_mask_list.to(self._device), self._adj_list.to(self._device))
            #print(policy_scores.size())
    
            #print(value_scores.size())
            for i in range(num_of_samples):
                self._data_connection_list[self._id_list[i]].send((policy_scores[i, :].detach().to(torch.device('cpu')), value_scores[i].detach().to(torch.device('cpu'))))


    def _get_batch_data(self):
        idx = 0
        while idx < self._num_of_live_process:
            try:
                id, qubits, padding_mask, adj = self._data_queue.get(timeout = self._timeout) 
                self._adj_list[idx, :, :] =  adj
                self._qubits_list[idx, :, :] = qubits
                self._padding_mask_list[idx, :] = padding_mask
                self._id_list[idx] = id
                idx += 1
            except queue.Empty:
                self._num_of_live_process -= 1

        return idx
        




        
        