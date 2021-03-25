from typing import List, Dict, Tuple

from multiprocessing import Queue, Pipe
from multiprocessing.connection import Connection

import torch
import numpy as np

from .nn_model import *
from QuICT.qcda.mapping.utility import *

class Inference(object):
    def __init__(self, feature_dim: int, num_of_class: int, config: GNNConfig, device: torch.device, timeout: int, num_of_connection: int, data_queue: Queue, data_connection_list: List[Connection]):
        self._timeout = timeout
        self._num_of_connection = num_of_connection
        self._data_queue = data_queue
        self._data_connection_list = data_connection_list
        self._device = device
        self._feature_dim = feature_dim
        self._num_of_class = num_of_class
        self._config = config
        self._model = TransformerU2GNN(feature_dim_size = self._feature_dim, 
                            num_classes = self._num_of_class, 
                           config = self._config).to(config.device).float()



    def __call__(self, batch_size: int = 8):
        id_list, adj_list, feature_list = self._get_batch_data(batch_size = batch_size)
        adj, graph_pool, feature = transform_batch(batch_data = (adj_list, feature_list), device = self._device)
        policy_scores, value_scores = self._model(adj, graph_pool, feature)
        for i in range(batch_size):
            self._data_connection_list[id_list[i]].send((policy_scores[i], value_scores[i]))


    def _get_batch_data(self, batch_size: int = 8):
        adj_list = []
        feature_list = []
        id_list = []
        for _ in range(batch_size):
            id, feature, adj = self._data_queue.get(timeout=self._timeout) 
            adj_list.append(adj)
            feature_list.append(feature)
            id_list.append(id)
        adj_list = np.array(adj_list)
        feature_list = np.array(feature_list)
        id_list = np.array(id_list)
        return id_list, adj_list, feature_list

    def _load_model(self, model_path: str): 
        self._model.load_state_dict(torch.load(model_path))
        self._model.eval()

        
        