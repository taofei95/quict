import time
from RL.experience_pool import ExperiencePool
from typing import List, Dict, Tuple

from multiprocessing import Queue, Pipe
from multiprocessing.connection import Connection

import torch
import numpy as np

from .nn_model import *
from QuICT.qcda.mapping.utility import *


class Trainner(object):
    def __init__(self, experience_pool: ExperiencePool, config: GNNConfig):
        self._experience_pool = experience_pool
        self._config = config
        self._model = TransformerU2GNN(feature_dim_size = self._experience_pool.feature_dim(), num_classes = self._experience_pool.num_of_class(),
                            config = self._config).to(config.device).float()
        
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr = config.learning_rate)
       


    def __call__(self, output_dir_path: str):
        num_batches_per_epoch = int((self._experience_pool.train_set_size() - 1) / self._config.batch_size) + 1
        self._scheduler = torch.optim.lr_scheduler.StepLR(self._optimizer, step_size = num_batches_per_epoch, gamma=0.1)
        
        print("Writing to {}\n".format(output_dir_path))
        # Checkpoint directory
        checkpoint_dir = os.path.abspath(os.path.join(output_dir_path, "checkpoints"))
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_prefix = os.path.join(checkpoint_dir, "warm_up_model")

        with open(checkpoint_prefix + '_acc.txt', 'w') as write_acc:
            cost_loss = []
            for epoch in range(1, self._config.num_of_epochs + 1):
                epoch_start_time = time.time()
                train_loss = self._train(num_batches_per_epoch = num_batches_per_epoch, batch_size = self._config.batch_size)
                cost_loss.append(train_loss)
                print('| epoch {:3d} | time: {:5.2f}s | loss {:5.2f} | test value loss {:5.2f} |  test policy loss {:5.2f}  |'.format(
                            epoch, (time.time() - epoch_start_time), train_loss))

                if epoch > 5 and cost_loss[-1] > np.mean(cost_loss[-6:-1]):
                    self._scheduler.step()
            write_acc.write('epoch ' + str(epoch) + '\n')
    
    def _train(self, batch_size: int, num_batches_per_epoch: int):
        self._model.train() # Turn on the train mode
        total_loss = 0.0
        for _ in range(num_batches_per_epoch):
            input_x, graph_pool, X_concat, value, policy = transform_batch(self._experience_pool.get_batch_data(batch_size = batch_size))
            self._optimizer.zero_grad()
            policy_prediction_scores, value_prediction_scores = self._model(input_x, graph_pool, X_concat)
            # loss = criterion(prediction_scores, graph_labels)
            loss = self._cross_entropy_and_MSE_loss(policy_prediction_scores, value_prediction_scores, policy, value)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), 0.5) # prevent the exploding gradient problem
            self._optimizer.step()
            total_loss += loss.item()
        return total_loss


    def _cross_entropy_and_MSE_loss(self, policy_prediction, value_prediction, policy, value):
        logsoftmax = nn.LogSoftmax(dim = 1)
        mse_loss = nn.MSELoss(reduction='mean')
        return mse_loss(value_prediction.squeeze(), value) + torch.mean(torch.sum(- policy * logsoftmax(policy_prediction), 1))

    def _cross_entropy_loss(self, arr, target):
        logsoftmax = nn.LogSoftmax(dim = 1)
        return torch.sum(torch.sum( -target * logsoftmax(arr), dim = 1))

    def _MSE_loss(self, arr, target):
        mse_loss = nn.MSELoss(reduction='sum')
        return mse_loss(target, arr)
    