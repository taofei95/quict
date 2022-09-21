import torch.nn

class VQANet():
    def __init__(self):
        self.network = Ansatz()

    def forward(self, params):
        state = self.network(params)
        return state

