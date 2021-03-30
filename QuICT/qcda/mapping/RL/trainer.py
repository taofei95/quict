from table_based_mcts import MCTSMode
from typing import List, Dict, Tuple

from multiprocessing import Queue, Pipe
from multiprocessing.connection import Connection
from tensorboardX import SummaryWriter


import time 
import torch
import numpy as np

from .dataloader import *
from .nn_model import *
from .rl_based_mcts import *
from QuICT.qcda.mapping.utility import *
from QuICT.qcda.mapping.coupling_graph import *
from QuICT.qcda.mapping.random_circuit_generator import *

class EvaluateMode(enumerate):
    SEARCH = 0
    EXTENDED_PROB = 1
    PROB = 2

class Benchmark(enumerate):
    REVLIB = 0
    RANDOM = 1


class Trainer(object):
    def __init__(self, coupling_graph: str = None, config: GNNConfig = None, log_path: str = None, tb_path: str = None):
        self._graph_name = coupling_graph
        self._coupling_graph = get_coupling_graph(coupling_graph)
        self._config = config
        self._device = config.device
        self._model = SequenceModel(n_qubits = self._coupling_graph.size, n_class = self._coupling_graph.num_of_edge,  config = self._config).to(config.device).float()
        self._writer = SummaryWriter(tb_path)
        
        logging.config.dictConfig({"version":1,"disable_existing_loggers": False})
        self._logger = logging.getLogger("train")
        # 第二步，创建一个handler，用于写入日志文件
        self._logger.setLevel(logging.DEBUG)
        self._logger.info("start up")
        formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
        if log_path is not None:
            print(log_path)
            fh = logging.FileHandler(log_path, mode='w')
            fh.setLevel(logging.INFO)  # 输出到file的log等级的开关
            # 第三步，定义handler的输出格式
            fh.setFormatter(formatter)
            self._logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO) 
        ch.setFormatter(formatter)
        self._logger.addHandler(ch)

    def run(self, model_path: str = None, input_path: str = None, quota: float = 0.1):
        dataloader = DataLoader(max_capacity = self._config.maximum_capacity, num_of_nodes = self._config.num_of_nodes, num_of_class = self._coupling_graph.num_of_edge)
        dataloader.load_data(file_path = input_path)
        dataloader.split_data(quota = quota)
       
        self._model.load_state_dict(torch.load(model_path))
        
        self._experience_pool = dataloader 
        num_params = 0
        for param in self._model.parameters():
            num_params += param.numel()
        self._logger.info(num_params)
        
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr = self._config.learning_rate)
        num_batches_per_epoch = int((self._experience_pool.train_set_size() - 1) / self._config.batch_size) + 1

        mini_loss = 1e9

        cost_loss = []
        for epoch in range(1, self._config.num_of_epochs + 1):
            epoch_start_time = time.time()
            train_loss = self._train(num_batches_per_epoch = num_batches_per_epoch, batch_size = self._config.batch_size)
            cost_loss.append(train_loss)
            test_value_loss, test_policy_loss = self._evaluate(batch_size = self._config.batch_size)
            
            self._logger.info("| epoch %3d | time: %5.2f s | loss %5.2f | test value loss %5.2f |  test policy loss %5.2f |"%(
                        epoch, (time.time() - epoch_start_time), train_loss, test_value_loss, test_policy_loss))
            
            if epoch > 5 and cost_loss[-1] > np.mean(cost_loss[-6:-1]):
                self._scheduler.step()
            if test_value_loss + test_policy_loss < mini_loss:
                mini_loss = test_value_loss + test_value_loss
                torch.save(self._model.state_dict(), model_path) 

        return 0

    def _train(self, num_batches_per_epoch: int, batch_size: int):
        self._model.train() # Turn on the train mode
        total_loss = 0.0
        for _ in range(num_batches_per_epoch):
            input_x, padding_mask, adj, value, policy, _ = self._transform_batch(self._experience_pool.get_batch_data(batch_size = batch_size))
            self._optimizer.zero_grad()
            policy_prediction_scores, value_prediction_scores = self._model(input_x, padding_mask, adj)
            # loss = criterion(prediction_scores, graph_labels)
            loss = self._cross_entropy_and_MSE_loss(policy_prediction_scores.squeeze(), value_prediction_scores.squeeze(), policy, value)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), 0.5) # prevent the exploding gradient problem
            self._optimizer.step()
            total_loss += loss.item()
        return total_loss

    def _evaluate(self, batch_size: int):
        """
        """
        self._model.eval()
        value_loss = 0.0
        policy_loss = 0.0
        num = self._experience_pool.evaluate_set_size()
        with torch.no_grad():
            # evaluating
            for i in range(0, num, batch_size):
                test_input_x, test_padding_mask, adj,  value, policy, _ = self._transform_batch(self._experience_pool.get_evaluate_data(start = i, end = i + batch_size))
                policy_prediction_scores, value_prediction_scores = self._model(test_input_x, test_padding_mask, adj)
                value_loss += self._MSE_loss(arr = value_prediction_scores.squeeze() , target = value)
                policy_loss += self._cross_entropy_loss(arr = policy_prediction_scores.squeeze(), target = policy)
                
            acc_value_test = value_loss / float(num)
            acc_policy_test = policy_loss /float(num)    
            return acc_value_test, acc_policy_test  
               
  
    def _transform_batch(self, batch_data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,np.ndarray,np.ndarray])->Tuple[torch.LongTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor ]:
        qubits_list, padding_mask_list, adj_list, value_list, action_probability_list, label_list = batch_data
        
        # padding_mask_list = np.concatenate([np.zeros(shape = (padding_mask_list.shape[0], 1), dtype = np.int32) , padding_mask_list], axis = 1)
        # qubits_list = np.concatenate([np.zeros(shape = (qubits_list.shape[0], 1 ,2), dtype = np.uint8) , qubits_list+2], axis = 1)
        
        adj = torch.from_numpy(adj_list).to(self._device).to(torch.long)
        padding_mask = torch.from_numpy(padding_mask_list).to(self._device).to(torch.uint8)
        qubits = torch.from_numpy(qubits_list).to(self._device).long()
        value = torch.from_numpy(value_list).to(self._device).float()
        action_probability = torch.from_numpy(action_probability_list).to(self._device).float()
        label = torch.from_numpy(label_list).to(self._device).long()

        return  qubits, padding_mask, adj, value, action_probability, label 

    def _cross_entropy_and_MSE_loss(self, policy_prediction, value_prediction, policy, value):
            logsoftmax = nn.LogSoftmax(dim = 1)
            mse_loss = nn.MSELoss(reduction='mean')
            return mse_loss(value_prediction.squeeze(), value) + self._config.loss_c * torch.mean(torch.sum(- policy * logsoftmax(policy_prediction), 1))
    
    def _NLL_loss(self, arr, target):
        return torch.sum(torch.sum( -target * torch.log(arr), dim = 1))

    def _cross_entropy_loss(self, arr, target):
        logsoftmax = nn.LogSoftmax(dim = 1)
        # print(target.size())
        # print(arr.size())
        return torch.sum(torch.sum( -target * logsoftmax(arr), dim = 1))

    def _MSE_loss(self, arr, target):
        mse_loss = nn.MSELoss(reduction='sum')
        return mse_loss(target, arr)
    