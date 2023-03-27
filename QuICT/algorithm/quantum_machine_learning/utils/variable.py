import numpy as np


class Variable:
    @property
    def pargs(self):
        return self._pargs

    @property
    def gradient(self):
        return self._gradient

    @pargs.setter
    def pargs(self, parg, index):
        self._pargs[index] = parg

    @gradient.setter
    def gradient(self, grad, index):
        self._gradient[index] = grad

    def __init__(self, array: np.ndarray):
        self._pargs = array
        self._gradient = np.zeros(self.pargs.shape, dtype=np.float32)

    # def __setitem__(self, index, value):
        
    
    def __getitem__(self, index):
        return self.pargs[index]


if __name__ == "__main__":
    pargs = np.array([[0, 1, 2], [0.1, 0.2, 0.3]])
    a = Variable(pargs)
    a.pargs(200, [1, 1])
    print(a.pargs)
