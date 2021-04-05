import os
import sys
import time
import cProfile
import numpy as np
import logging

import torch.multiprocessing as mp
from torch.multiprocessing import Pool
from tensorboardX import SummaryWriter

from multiprocessing import shared_memory, Lock
from multiprocessing.managers import BaseManager, BaseProxy
from multiprocessing.sharedctypes import Value

import torch
import torch.nn as nn
import torch.nn.functional as F


from typing import List, Sequence, Tuple, Optional, Union, Dict
from QuICT.tools.interface import *
from QuICT.core.circuit import *
from QuICT.core.layout import *


from utility import *
from table_based_mcts import *
from random_circuit_generator import *
from RL.experience_pool_v4 import *
from RL.nn_model import *
from RL.dataloader import *

exp_name = "test_overfitting_swap_gates"


def mcts_process(qubit_mapping: List[int], coupling_graph: str,  minimum_circuit: int = 50, maximum_circuit: int = 1000, min_num_of_qubits: int = 5, max_num_of_qubits: int = 20, seed: int = 0):
    random_circuit_generator = RandomCircuitGenerator(minimum = minimum_circuit, maximum = maximum_circuit, min_num_of_qubits = min_num_of_qubits, max_num_of_qubits = max_num_of_qubits, seed = seed)
    qc = random_circuit_generator()
    mcts = TableBasedMCTS(mode = MCTSMode.TRAIN, coupling_graph = coupling_graph, experience_pool = global_experience_pool, gamma = config.gamma, 
                        num_of_swap_gates = config.num_of_swap_gates, sim_method = config.sim_method, rl = RLMode.WARMUP)   
    #cProfile.runctx('mcts.search(logical_circuit = qc, init_mapping = qubit_mapping)', globals(), locals())
    res = mcts.test_random_simulation(logical_circuit = qc, init_mapping = qubit_mapping ,num_of_gates = -1)
    return res

def init(ep: ExperiencePool):
    global global_experience_pool
    global_experience_pool = ep

 
def mcts_callback(result):
    print(result)

def mcts_error_callback(result):
    print("error:", end = ' ')
    print(result)


class PolicyTrainer(object):
    def __init__(self, graph_name: str, num_of_qubits: int = 20, log_path: str = None, tb_path: str = None):
        self._num_of_qubits = num_of_qubits
        self._graph_name = graph_name
        self._coupling_graph = get_coupling_graph(graph_name)

        self._writer = SummaryWriter(tb_path)
        self._logger = logging.getLogger("policy_train")
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
    

    def preprocess(self, init_mapping:List[int],config: GNNConfig, new_data: bool = True, input_dir_path: str = None, num_of_circuits: int = 50):
        self._new_data= new_data
        self._num_of_circuits = num_of_circuits
        self._init_mapping = init_mapping
        
        self._config = config
        self._num_of_process = config.num_of_process
        self._device = config.device
        self._minimum_circuit = config.minimum_circuit
        self._maximum_circuit = config.maximum_circuit 
        self._maximum_capacity = config.maximum_capacity
        self._num_of_gates = config.num_of_gates
        self._feature_update = config.feature_update
        # adj = np.zeros(shape = (config.maximum_capacity, config.num_of_gates, 4), dtype = np.int32)
        # feature = np.zeros(shape = (config.maximum_capacity, config.num_of_gates, self._coupling_graph.node_feature.shape[1]), dtype = np.float)
        # action_probability = np.zeros(shape = (config.maximum_capacity, self._coupling_graph.size), dtype = np.float)
        # value = np.zeros(shape = (config.maximum_capacity,), dtype = np.float)
        self._logger.info(config)       
        if new_data:
            self._experience_pool = ExperiencePool(max_capacity = self._maximum_capacity, num_of_nodes = self._num_of_gates, num_of_class =  self._coupling_graph.num_of_edge)
        else:
            self._experience_pool = DataLoader(max_capacity = self._maximum_capacity, num_of_nodes = self._num_of_gates, num_of_class =  self._coupling_graph.num_of_edge)
        self._construct_training_sample_set(input_dir_path)

    def run(self, output_dir_path: str = None):
    
        #print(self._experience_pool.feature_dim)
        self._model = SequenceModel(n_qubits = self._coupling_graph.size, n_class = self._coupling_graph.num_of_edge,  config = self._config).to(self._device).float()
        num_params = 0
        for param in self._model.parameters():
            num_params += param.numel()
        self._logger.info(num_params)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr = self._config.learning_rate, weight_decay = config.weight_decay)
        num_batches_per_epoch = int((self._experience_pool.train_set_size() - 1) / self._config.batch_size) + 1
        self._scheduler = torch.optim.lr_scheduler.StepLR(self._optimizer, step_size = num_batches_per_epoch, gamma=0.1)

        print("Writing to {}\n".format(output_dir_path))
        # Checkpoint directory
        checkpoint_dir = os.path.abspath(os.path.join(output_dir_path, "checkpoints"))
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_prefix = os.path.join(checkpoint_dir, "warm_up_model")
        mini_loss = 1e9

        #self._model.load_state_dict(torch.load(f"{checkpoint_dir}/policy_model_state_dict_gat"))
        with open(checkpoint_prefix + '_acc.txt', 'w') as write_acc:
            cost_loss = []
            for epoch in range(1, self._config.num_of_epochs + 1):
                epoch_start_time = time.time()
                train_loss = self._train(num_batches_per_epoch = num_batches_per_epoch, batch_size = self._config.batch_size)
                cost_loss.append(train_loss)
                acc_value_test, acc_policy_test = self._evaluate(batch_size = self._config.batch_size)
                self._logger.info("| epoch %3d | time: %5.2f s | loss %5.2f | test value loss %5.2f |  test policy loss %5.2f |"%(
                            epoch, (time.time() - epoch_start_time), train_loss, acc_value_test, acc_policy_test))
                print("| epoch %3d | time: %5.2f s | loss %5.2f | test value loss %5.2f |  test policy loss %5.2f |"%(
                            epoch, (time.time() - epoch_start_time), train_loss, acc_value_test, acc_policy_test))
                if epoch > 5 and cost_loss[-1] > np.mean(cost_loss[-6:-1]):
                    self._scheduler.step()
                if acc_value_test + acc_policy_test < mini_loss:
                    mini_loss = acc_value_test + acc_policy_test
                    with open(f"{checkpoint_dir}/model_state_dict_{exp_name}",'w') as f:
                        torch.save(self._model.state_dict(), f"{checkpoint_dir}/model_state_dict_{exp_name}" ) 

            write_acc.write('epoch ' + str(epoch) + 'test value loss' + str(acc_value_test) + 'test policy loss' + str(acc_policy_test) + '\n')


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
                test_input_x, test_padding_mask, adj, value, policy, _ = self._transform_batch(self._experience_pool.get_evaluate_data(start = i, end = i + batch_size))
                policy_prediction_scores, value_prediction_scores = self._model(test_input_x, test_padding_mask, adj)
                value_loss += self._MSE_loss(arr = value_prediction_scores.squeeze() , target = value)
                policy_loss += self._cross_entropy_loss(arr = policy_prediction_scores.squeeze(), target = policy)
                
            acc_value_test = value_loss / float(num)
            acc_policy_test = policy_loss /float(num)    
            return acc_value_test, acc_policy_test  
                                                      

    def run_policy(self, output_dir_path: str = None):
        self._model = SequenceModel(n_qubits = self._coupling_graph.size, n_class = self._coupling_graph.num_of_edge,  config = self._config).to(self._device).float()
        #self._writer.add_graph(self._model, input_to_model=None, verbose=False)
        num_params = 0
        for param in self._model.parameters():
            num_params += param.numel()
        self._logger.info(num_params)

        self._optimizer = torch.optim.Adam(self._model.parameters(), lr = self._config.learning_rate)
        num_batches_per_epoch = int((self._experience_pool.train_set_size() - 1) / self._config.batch_size) + 1
        #self._scheduler = torch.optim.lr_scheduler.MultiStepLR(self._optimizer, milestones=[100, 200], gamma=0.1)

        print("Writing to {}\n".format(output_dir_path))
        # Checkpoint diretory
        checkpoint_dir = os.path.abspath(os.path.join(output_dir_path, "checkpoints"))
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_prefix = os.path.join(checkpoint_dir, "warm_up_model")
        mini_loss = 1e9
        with open(checkpoint_prefix + '_acc_policy.txt', 'w') as write_acc:
            cost_loss = []
            for epoch in range(1, self._config.num_of_epochs + 1):
                epoch_start_time = time.time()
                train_loss, train_lb = self._policy_train(num_batches_per_epoch = num_batches_per_epoch, batch_size = self._config.batch_size)
                cost_loss.append(train_loss)
                acc_policy_test, test_lb = self._policy_evaluate(batch_size = self._config.batch_size)
                self._logger.info("| epoch %3d | time: %5.2f s | loss %5.2f  | lower bound %5.2f | test policy loss %5.2f | test lower bound %5.2f |"%(
                            epoch, (time.time() - epoch_start_time), train_loss, train_lb, acc_policy_test, test_lb))
                if  acc_policy_test < mini_loss:
                    mini_loss = acc_policy_test
                    with open(f"{checkpoint_dir}/policy_model_state_dict_ctexm",'w') as f:
                        torch.save(self._model.state_dict(), f"{checkpoint_dir}/policy_model_state_dict_ctexm" ) 
                self._writer.add_scalar('traning_loss', train_loss , epoch)
                self._writer.add_scalar('testing_loss', acc_policy_test, epoch)
            write_acc.write('epoch ' + str(epoch) + 'test policy loss' + str(acc_policy_test) + '\n')
            # softmax = nn.Softmax(dim = 1)
            # input_x, _, policy, _ = self._transform_batch(self._experience_pool.get_evaluate_data(start = 0, end =  self._config.batch_size))
            # policy = policy + 1e-7
            # policy_prediction_scores = self._model(input_x)
            # print(input_x)
            # print(policy)
            # print(policy_prediction_scores)


    def _policy_train(self, num_batches_per_epoch: int, batch_size: int):
        self._model.train() # Turn on the train mode
        total_loss = 0.0
        total_lb = 0.0  
        log_softmax = nn.LogSoftmax(dim = 1)
        kl_loss = nn.KLDivLoss(reduction = 'batchmean')
        num = self._experience_pool.train_set_size()
        for i in range(num_batches_per_epoch):
            input_x, padding_mask, adj, _, policy, _ = self._transform_batch(self._experience_pool.get_batch_data(batch_size = batch_size))
            policy = policy + 1e-7
            self._optimizer.zero_grad()
            policy_prediction_scores, _ = self._model(input_x, padding_mask, adj)
            # print(policy_prediction_scores.size())
            # print(softmax(policy_prediction_scores)[0])
            # print(torch.sum(softmax(policy_prediction_scores), dim = 1))
            # loss = criterion(prediction_scores, graph_labels)
            # p = log_softmax(policy_prediction_scores)
            # print(p.size())
            # loss = kl_loss(p, policy)
            loss = self._cross_entropy_loss(policy_prediction_scores, policy)
            lb = self._NLL_loss(policy, policy)
            loss.backward()

            # for name, param in  self._model.named_parameters():
            #      self._writer.add_histogram(name , param.grad, i)
            #torch.nn.utils.clip_grad_norm_(self._model.parameters(), 0.5) # prevent the exploding gradient problem
            
            self._optimizer.step()
            total_loss += loss.item()
            total_lb += lb.item()
        return total_loss, total_lb

    def _policy_evaluate(self, batch_size: int):
        """
        """
        self._model.eval()
        policy_loss = 0.0
        policy_lb = 0.0
        num = self._experience_pool.evaluate_set_size()
        with torch.no_grad():
            # evaluating
            for i in range(0, num, batch_size):
                test_input_x, padding_mask, adj, _, policy, _ = self._transform_batch(self._experience_pool.get_evaluate_data(start = i, end = i + batch_size))
                policy = policy + 1e-7
                policy_prediction_scores,_ = self._model(test_input_x, padding_mask, adj)
                policy_loss += self._cross_entropy_loss(arr = policy_prediction_scores, target = policy)
                policy_lb += self._NLL_loss(arr = policy, target = policy)
                #print(policy_loss)
            acc_policy_test = policy_loss
            lb_polict_test = policy_lb    
            return  acc_policy_test, lb_polict_test   



    def run_label(self, output_dir_path: str = None):
        self._model = SequenceModel(n_qubits = self._coupling_graph.size, n_class = self._coupling_graph.num_of_edge,  config = self._config).to(self._device).float()
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr = self._config.learning_rate)
        num_batches_per_epoch = int((self._experience_pool.train_set_size() - 1) / self._config.batch_size) + 1
        self._scheduler = torch.optim.lr_scheduler.StepLR(self._optimizer, step_size = num_batches_per_epoch, gamma=0.1)

        print("Writing to {}\n".format(output_dir_path))
        # Checkpoint diretory
        checkpoint_dir = os.path.abspath(os.path.join(output_dir_path, "checkpoints"))
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_prefix = os.path.join(checkpoint_dir, "warm_up_model")
        mini_loss = 1e9
        with open(checkpoint_prefix + '_acc_label.txt', 'w') as write_acc:
            cost_loss = []
            for epoch in range(1, self._config.num_of_epochs + 1):
                epoch_start_time = time.time()
                train_loss = self._label_train(num_batches_per_epoch = num_batches_per_epoch, batch_size = self._config.batch_size)
                cost_loss.append(train_loss)
                acc_label_test = self._label_evaluate(batch_size = self._config.batch_size)
                self._logger.info("| epoch %3d | time: %5.2f s | loss %5.2f  |  test label loss %5.2f |"%(
                            epoch, (time.time() - epoch_start_time), train_loss, acc_label_test))
                if epoch > 5 and cost_loss[-1] > np.mean(cost_loss[-6:-1]):
                    self._scheduler.step()
                if  acc_label_test < mini_loss:
                    mini_loss = acc_label_test
                    with open(f"{checkpoint_dir}/label_model_state_dict_dag",'w') as f:
                        torch.save(self._model.state_dict(), f"{checkpoint_dir}/label_model_state_dict_dag" ) 

            write_acc.write('epoch ' + str(epoch) + 'test label loss' + str(acc_label_test) + '\n')


    def _label_train(self, num_batches_per_epoch: int, batch_size: int):
        self._model.train() # Turn on the train mode
        total_loss = 0.0
        num = self._experience_pool.train_set_size()
        for i in range(0, num, batch_size):
            input_x, padding_mask, _, _, label_index = self._transform_batch(self._experience_pool.get_train_data(start = i, end = i + batch_size))
            label = torch.zeros(size = (label_index.shape[0], self._coupling_graph.num_of_edge), dtype = torch.float32, device = self._device)
            for i, idx in enumerate(label_index):
                label[i,idx] = 1
            
            self._optimizer.zero_grad()
            policy_prediction_scores,_ = self._model(input_x, padding_mask)
            # loss = criterion(prediction_scores, graph_labels)
            loss = self._cross_entropy_loss(policy_prediction_scores, label)
            loss.backward()
            # for name,param in self._model.named_parameters():
            #     self._writer.add(name, param.grad, i )

            #torch.nn.utils.clip_grad_norm_(self._model.parameters(), 0.5) # prevent the exploding gradient problem
            self._optimizer.step()
            total_loss += loss.item()
        return total_loss

    def _label_evaluate(self, batch_size: int):
        """
        """
        self._model.eval()
        policy_loss = 0.0
        num = self._experience_pool.evaluate_set_size()
        softmax = nn.Softmax(dim = 1)
        with torch.no_grad():
            # evaluating
            for i in range(0, num, batch_size):
                test_input_x, test_padding_mask, _, _, label = self._transform_batch(self._experience_pool.get_evaluate_data(start = i, end = i + batch_size))
                policy_prediction_scores,_ = self._model(test_input_x, test_padding_mask)
                predict_label =  torch.argmax(softmax(policy_prediction_scores), dim = 1)
                policy_loss +=  torch.sum(predict_label != label) 
                #print(policy_loss)
            acc_policy_test = policy_loss / float(num)  
            return  acc_policy_test    


    def run_value(self, output_dir_path: str = None):
        self._model = SequenceModel(n_qubits = self._coupling_graph.size, n_class = self._coupling_graph.num_of_edge,  config = self._config).to(self._device).float()
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr = self._config.learning_rate)
        num_batches_per_epoch = int((self._experience_pool.train_set_size() - 1) / self._config.batch_size) + 1
        self._scheduler = torch.optim.lr_scheduler.StepLR(self._optimizer, step_size = num_batches_per_epoch, gamma=0.1)

        print("Writing to {}\n".format(output_dir_path))
        # Checkpoint diretory
        checkpoint_dir = os.path.abspath(os.path.join(output_dir_path, "checkpoints"))
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_prefix = os.path.join(checkpoint_dir, "warm_up_model")
        mini_loss = 1e9
        with open(checkpoint_prefix + '_acc_label.txt', 'w') as write_acc:
            cost_loss = []
            for epoch in range(1, self._config.num_of_epochs + 1):
                epoch_start_time = time.time()
                train_loss = self._value_train(num_batches_per_epoch = num_batches_per_epoch, batch_size = self._config.batch_size)
                cost_loss.append(train_loss)
                acc_value_test = self._value_evaluate(batch_size = self._config.batch_size)
                self._logger.info("| epoch %3d | time: %5.2f s | loss %5.2f  |  test value loss %5.2f |"%(
                            epoch, (time.time() - epoch_start_time), train_loss, acc_value_test))
                if epoch > 5 and cost_loss[-1] > np.mean(cost_loss[-6:-1]):
                    self._scheduler.step()
                if  acc_value_test < mini_loss:
                    mini_loss = acc_value_test
                    with open(f"{checkpoint_dir}/value_model_state_dict",'w') as f:
                        torch.save(self._model.state_dict(), f"{checkpoint_dir}/value_model_state_dict" ) 

            write_acc.write('epoch ' + str(epoch) + 'test value loss' + str(acc_value_test) + '\n')


    def _value_train(self, num_batches_per_epoch: int, batch_size: int):
        self._model.train() # Turn on the train mode
        total_loss = 0.0
        num = self._experience_pool.train_set_size()
        for i in range(0, num, batch_size):
            input_x, padding_mask, value, _, _ = self._transform_batch(self._experience_pool.get_train_data(start = i, end = i + batch_size))
            
            self._optimizer.zero_grad()
            _, value_prediction_scores = self._model(input_x, padding_mask)
            # loss = criterion(prediction_scores, graph_labels)
            loss = self._MSE_loss(value_prediction_scores.squeeze(), value)
            loss.backward()
            # for name,param in self._model.named_parameters():
            #     self._writer.add(name, param.grad, i )

            #torch.nn.utils.clip_grad_norm_(self._model.parameters(), 0.5) # prevent the exploding gradient problem
            self._optimizer.step()
            total_loss += loss.item()
        return total_loss

    def _value_evaluate(self, batch_size: int):
        """
        """
        self._model.eval()
        value_loss = 0.0
        num = self._experience_pool.evaluate_set_size()
        with torch.no_grad():
            # evaluating
            for i in range(0, num, batch_size):
                test_input_x, test_padding_mask, value, _, _ = self._transform_batch(self._experience_pool.get_evaluate_data(start = i, end = i + batch_size))
                _,test_value_prediction_scores = self._model(test_input_x, test_padding_mask)
                value_loss += self._MSE_loss(test_value_prediction_scores.squeeze(), value)
                #print(policy_loss)
            acc_value_test = value_loss / float(num)  
            return  acc_value_test    




    def close(self):
        self._writer.close()
        if self._new_data:
            self._experience_pool.close()
            self._experience_pool.unlink()
            

    def _construct_training_sample_set(self, input_dir_path: str = None):
        if self._new_data is True:
            prg = np.random.default_rng(12345)
            start_time = time.time()
            seed = prg.integers(low = 0, high = 10000, size = self._num_of_circuits)
            with Pool(processes = self._num_of_process, initializer = init, initargs = (self._experience_pool,) ) as pool:
                for i in range(self._num_of_circuits):   
                    pool.apply_async(mcts_process, args = (self._init_mapping, self._graph_name, self._minimum_circuit, self._maximum_circuit, 5, self._num_of_qubits,  seed[i]), callback = mcts_callback, error_callback = mcts_error_callback)       
                pool.close()
                pool.join() 
            self._experience_pool.save_data(input_dir_path)  
            time_cost = time.time() - start_time
            self._logger.info("The process of data generation cost {:5.2f}s".format(time_cost))
        else:
            self._experience_pool.load_data(input_dir_path)

        self._experience_pool.split_data(quota = 0.1)

    def _cross_entropy_and_MSE_loss(self, policy_prediction, value_prediction, policy, value):
            logsoftmax = nn.LogSoftmax(dim = 1)
            mse_loss = nn.MSELoss(reduction='mean')
            return mse_loss(value_prediction.squeeze(), value) + self._config.loss_c * torch.mean(torch.sum(- policy * logsoftmax(policy_prediction), 1))
    
    def _NLL_loss(self, arr, target):
        return torch.mean(torch.sum( -target * torch.log(arr), dim = 1))

    def _cross_entropy_loss(self, arr, target):
        logsoftmax = nn.LogSoftmax(dim = 1)
        # print(target.size())
        # print(arr.size())
        return torch.sum(torch.sum( -target * logsoftmax(arr), dim = 1))

    def _MSE_loss(self, arr, target):
        mse_loss = nn.MSELoss(reduction='sum')
        return mse_loss(target, arr)

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

    def _get_adj_list(self, adj_list: List[np.ndarray])->np.ndarray:
        idx_bias = 0
        # print(adj_list)
        for i in range(len(adj_list)):
            adj_list[i] = adj_list[i] + idx_bias 
            idx_bias +=  adj_list[i].shape[0]
        
        #print(adj_list)
        return adj_list

    def _get_graph_pool(self, batch_graph: List[np.ndarray], device: torch.device)-> torch.FloatTensor:
        """
        """
        num_of_graph = len(batch_graph)
        num_of_nodes = 0
        bias = [0]
        for i in range(num_of_graph):
            num_of_nodes += batch_graph[i].shape[0]
            bias.append(num_of_nodes)

        elem = torch.ones(num_of_nodes, dtype = torch.float)
        idx = torch.zeros([2, num_of_nodes ], dtype = torch.long)
        for i in range(num_of_graph):
            v = torch.arange(start = 0, end = batch_graph[i].shape[0], dtype = torch.long)
            idx[0, bias[i] : bias[i+1]] = i
            idx[1, bias[i] : bias[i+1]] = v

        graph_pool = torch.sparse.FloatTensor(idx, elem, torch.Size([num_of_graph, num_of_nodes])).to(device)

        return graph_pool

if __name__ == "__main__":
    file_path = os.path.realpath(__file__)
    dir_path, file_name = os.path.split(file_path)
    torch.cuda.set_device(2)
    policy_dir_path = os.path.abspath(os.path.join(dir_path, "policy"))
    if os.path.exists(policy_dir_path) is not True:
        os.makedirs(policy_dir_path)
        print(policy_dir_path)
    
    input_dir_path = os.path.abspath(os.path.join(policy_dir_path, "input"))
    if os.path.exists(input_dir_path) is not True:
        os.makedirs(input_dir_path)
        print(input_dir_path)

    output_dir_path = os.path.abspath(os.path.join(policy_dir_path, "output"))
    if os.path.exists(output_dir_path) is not True:
        os.makedirs(output_dir_path)
        print(output_dir_path)
    
    log_dir_path = os.path.abspath(os.path.join(policy_dir_path, "log"))
    if os.path.exists(log_dir_path) is not True:
        os.makedirs(log_dir_path)
        print(log_dir_path)
    
    tb_dir_path = os.path.abspath(os.path.join(policy_dir_path, "tensorboard"))
    if os.path.exists(tb_dir_path) is not True:
        os.makedirs(tb_dir_path)
        print(tb_dir_path)
    
    warmup_config = GNNConfig(maximum_capacity = 600000, num_of_gates = 150, maximum_circuit = 1500, minimum_circuit = 200, batch_size = 256, ff_hidden_size = 128, num_self_att_layers=4, dropout = 0.5, value_head_size = 128, gamma = 0.7, 
                      num_U2GNN_layers=2, learning_rate = 0.001, weight_decay = 1e-4, num_of_epochs = 1000, device = torch.device( "cuda"), graph_name = 'ibmq20',num_of_process = 10, feature_update = True, gat = False, n_gat = 2, mcts_c = 20, loss_c = 10,
                      num_of_swap_gates = 15, sim_method = 2)
    config = warmup_config
    new_data = True
    input_path = f"{input_dir_path}/swap_gates"
    num_of_qubits=20
    init_mapping=[i for i in range(0,20)]
    num_of_circuits = 500
   
    warmup_process =  PolicyTrainer(graph_name = "ibmq20", num_of_qubits = num_of_qubits, log_path = f"{log_dir_path}/{exp_name}", tb_path = f"{tb_dir_path}/policy/{exp_name}")
    
    try:
        warmup_process.preprocess(init_mapping = init_mapping, config = config, new_data = new_data, num_of_circuits = num_of_circuits, input_dir_path = input_path)
        warmup_process.run(output_dir_path = output_dir_path )
    finally:
        warmup_process.close()
    

