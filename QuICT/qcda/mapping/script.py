import numpy as np
import importlib as ib

WORK_DIR = "./warmup/input/mcts_cpp"

qubits = np.load(f"{WORK_DIR}/qubits_list.npy", allow_pickle = True)
adj = np.load(f"{WORK_DIR}/adj_list.npy", allow_pickle = True)
value = np.load(f"{WORK_DIR}/value_list.npy", allow_pickle = True)
num = np.load(f"{WORK_DIR}/num_list.npy", allow_pickle = True)
ap = np.load(f"{WORK_DIR}/action_probability_list.npy", allow_pickle = True)
sg = np.load(f"{WORK_DIR}/label_list.npy", allow_pickle = True)