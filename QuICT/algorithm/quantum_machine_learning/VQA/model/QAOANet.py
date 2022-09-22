import torch.nn

from QuICT.algorithm.quantum_machine_learning.VQA.model.VQANet import VQANet

class QAOANet(VQANet):
    def __init__(self):
        self.network = Ansatz()

    
    def forward(self):
        state = self.network()
        return state