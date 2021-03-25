import torch
from RL.nn_model import *
from coupling_graph import *
model =  TransformerU2GNN(feature_dim_size = coupling_graph.node_feature.shape[1]*2, 
                                        num_classes = coupling_graph.num_of_edge, 
                                        config = config).to(config.device).float()


