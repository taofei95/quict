import multiprocessing
import os
import sys
import time
import cProfile
import numpy as np
import logging
from shutil import copyfile



import torch
import torch.nn as nn
import torch.nn.functional as F


from typing import List, Sequence, Tuple, Optional, Union, Dict
from QuICT.tools.interface import *
from QuICT.core.circuit import *
from QuICT.core.layout import *




from table_based_mcts import *
from random_circuit_generator import *
from RL.nn_model import *
from RL.trainer import *
from RL.dataloader import *
from RL.trainer import *
from RL.inference import *
from utility import *

from tensorboardX import SummaryWriter
from torch.multiprocessing import Value, Pipe, Queue, SimpleQueue ,set_start_method

from multiprocessing import Pool, Process
from RL.experience_pool_v3 import *
def inference_process(graph_name: str, config: GNNConfig, model_path: str = None, q: Queue = None, conn = None):
    inference = Inference(graph_name = graph_name, config = config, model_path = model_path, data_queue = q, data_connection_list = conn, timeout = 10)
    print("inference")
    inference(num_of_process = config.num_of_process)


def trainer_process(graph_name: str, config: GNNConfig, log_path: str = None, tb_path: str = None, model_path: str = None, input_path: str = None):
    coupling_graph = get_coupling_graph(graph_name = graph_name)
    trainer = Trainer(coupling_graph = graph_name , config = config, log_path = log_path, tb_path = tb_path)
    return trainer.run(model_path =  model_path, input_path = input_path, pre_train = True) 

def mcts_process(qubit_mapping: List[int], graph_name: str,  minimum_circuit: int = 50, maximum_circuit: int = 1000, min_num_of_qubits: int = 5, max_num_of_qubits: int = 20, seed: int = 0, log_path: str = None, gpu_device: int = 0, model_path: str = None, config: GNNConfig = None):
    #print("rcg")
    global_experience_pool.create()
    random_circuit_generator = RandomCircuitGenerator(minimum = minimum_circuit, maximum = maximum_circuit, min_num_of_qubits = min_num_of_qubits, max_num_of_qubits = max_num_of_qubits, seed = seed)
    qc = random_circuit_generator()
    mcts = RLBasedMCTS(mode = MCTSMode.TRAIN, rl = RLMode.SELFPALY, coupling_graph = graph_name, experience_pool = global_experience_pool, log_path = log_path, 
                     c = config.mcts_c, size_threshold = config.num_of_nodes, device = config.device, input = gloabl_queue, output = global_conn, id = global_id, gamma = config.gamma, extended = True )   
    #cProfile.runctx('mcts.search(logical_circuit = qc, init_mapping = qubit_mapping)', globals(), locals())
    #print("mcts")
    res = mcts.search(logical_circuit = qc, init_mapping = qubit_mapping)
    global_experience_pool.close()
    return res

def init(ep: ExperiencePool, q: Queue, conn, idx: Value ):
    global global_experience_pool, gloabl_queue, global_conn, global_id
    global_experience_pool = ep
    gloabl_queue = q
   
    # curp = multiprocessing.current_process()
    # print(curp._identity)
    with idx.get_lock():
        global_conn = conn[idx.value]
        global_id = idx.value
        print(idx.value)
        idx.value += 1
        

 
def callback(result):
    print(result)

def error_callback(result):
    print("error:", end = ' ')
    print(result)



class AlphaQuts(object):

    def __init__(self, graph_name: str, config: GNNConfig, log_path: str = None, train_log_path: str = None, train_tb_path: str = None, mcts_log_path: str = None, model_path: str = None):
        self._graph_name = graph_name
        self._coupling_graph = get_coupling_graph(graph_name = graph_name)
        self._config = config
        
        self._train_log_path = train_log_path
        self._train_tb_path = train_tb_path
        self._mcts_log_path = mcts_log_path
        self._model_path = model_path
        
        logging.config.dictConfig({"version":1,"disable_existing_loggers": False})
        self._logger = logging.getLogger("AlphaQut")
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
        
        self._logger.info(config)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO) 
        ch.setFormatter(formatter)
        self._logger.addHandler(ch)

        max_capacity = config.maximum_capacity
        num_of_nodes = config.num_of_nodes
        num_of_neighbours = 5
        num_of_class = self._coupling_graph.num_of_edge

        self._label_shm = shared_memory.SharedMemory(create = True, size =max_capacity * 4)
        self._num_shm = shared_memory.SharedMemory(create = True, size =max_capacity * 4)
        self._adj_shm = shared_memory.SharedMemory(create = True, size = max_capacity * num_of_nodes * num_of_neighbours * 4)
        self._qubits_shm = shared_memory.SharedMemory(create = True, size = max_capacity * num_of_nodes * 2 * 4)
        self._action_probability_shm = shared_memory.SharedMemory(create = True, size = max_capacity * num_of_class * 8)
        self._value_shm = shared_memory.SharedMemory(create = True, size = max_capacity * 8)
        
        self._shm_name = SharedMemoryName(label_name = self._label_shm.name, 
                                        num_name =  self._num_shm.name,
                                        adj_name = self._adj_shm.name,
                                        qubits_name = self._qubits_shm.name,
                                        action_probability_name = self._action_probability_shm.name,
                                        value_name = self._value_shm.name )

        self._experience_pool = ExperiencePool(max_capacity = config.maximum_capacity, num_of_nodes = config.num_of_nodes, num_of_class = self._coupling_graph.num_of_edge, shm_name = self._shm_name) 
        self._experience_pool.create()
        

    def __call__(self, num_of_iterations: int = 5, num_of_circuits: int = 100, input_path: str = None):
        #self._experience_pool.load_data(input_path)
        prg = np.random.default_rng(12345*int(time.time())) 
        init_mapping = [i for i in range(self._coupling_graph.size)]
        idx = Value('i')
        
        q = Queue()
        send_conn = []
        recv_conn = []
        for _ in range(self._config.num_of_process):
            s_conn, r_conn = Pipe()
            send_conn.append(s_conn)
            recv_conn.append(r_conn)

        idx.value = 0
        start_time = time.time()
        seed = prg.integers(low = 0, high = 10000, size = num_of_circuits)
        p_inference = Process(target = inference_process, args = (self._graph_name, self._config, self._model_path, q, send_conn))
        
        with Pool(processes = self._config.num_of_process, initializer = init, initargs = (self._experience_pool, q, recv_conn, idx)) as pool:
            p_inference.start()
            for i in range(num_of_circuits):
                pool.apply_async(mcts_process, 
                            args = (init_mapping, self._graph_name, self._config.minimum_circuit, self._config.maximum_circuit, 5, self._coupling_graph.size,  seed[i], self._mcts_log_path, i, self._model_path, self._config),
                                callback = callback, error_callback = error_callback)   
            pool.close()
            pool.join()

        self._experience_pool.save_data(input_path)  
        time_cost = time.time() - start_time
        p_inference.terminate()
        self._logger.info("The process of data generation cost {:5.2f}s".format(time_cost))   
        p_train = Process(target = trainer_process, args = (self._graph_name, self._config, self._train_log_path, self._train_tb_path, self._model_path, input_path))
        p_train.start()
        p_train.join()
        p_train.close()

    
    def close(self):
        self._experience_pool.close()
        self._experience_pool.unlink()

if __name__ == "__main__":
    file_path = os.path.realpath(__file__)
    dir_path, file_name = os.path.split(file_path)
    torch.cuda.set_device(2)

    work_dir_path = os.path.abspath(os.path.join(dir_path, "alphaQ"))
    input_dir_path = os.path.abspath(os.path.join(work_dir_path, "input"))
    output_dir_path = os.path.abspath(os.path.join(work_dir_path, "output"))
    log_dir_path = os.path.abspath(os.path.join(work_dir_path, "log"))
    tb_dir_path = os.path.abspath(os.path.join(work_dir_path, "tensorboard"))

    exp = "test_small_data_extended"
    train_log_path = f"{log_dir_path}/train/train_{exp}.log"
    train_tb_path = f"{tb_dir_path}/train/train_{exp}"
    mcts_log_path = f"{log_dir_path}/mcts/mcts_{exp}.log"
    log_path = f"{log_dir_path}/{exp}.log"
    initial_model_path = f"{output_dir_path}/checkpoint/model_state_dict_initial"
    model_path = f"{output_dir_path}/checkpoint/model_state_dict_{exp}"
    input_path = f"{input_dir_path}/test"

    copyfile(initial_model_path, model_path)
    
    graph_name = "ibmq20"
    num_of_iterations = 5
    num_of_circuits = 300
    set_start_method("forkserver")

    alpha_config = GNNConfig(maximum_capacity = 200000, num_of_gates = 150, maximum_circuit = 1500, minimum_circuit = 200, batch_size = 128, ff_hidden_size = 128, num_self_att_layers=4, dropout = 0.5, value_head_size = 128,
                       num_U2GNN_layers=2, learning_rate = 0.001, weight_decay = 1e-4, num_of_epochs = 50, device = torch.device("cuda"), graph_name = 'ibmq20',num_of_process = 64, feature_update = True, gat = False, 
                       mcts_c = 20, loss_c = 10, n_gat = 2)
    try:
        alphaQ = AlphaQuts(graph_name = graph_name, config = alpha_config, log_path = log_path, 
                      train_log_path = train_log_path, train_tb_path = train_tb_path, 
                     mcts_log_path = mcts_log_path, model_path = model_path)
    
        alphaQ(num_of_iterations = num_of_iterations, num_of_circuits = num_of_circuits, input_path = input_path)
    finally:
        alphaQ.close()

    #trainer_process(graph_name = graph_name, config = alpha_config , log_path = train_log_path, tb_path = train_tb_path, model_path = model_path, input_path = input_path)
