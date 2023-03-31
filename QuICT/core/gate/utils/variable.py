import numpy as np


class Variable:
    @property
    def pargs(self):
        return self._pargs

    @property
    def grads(self):
        return self._grads

    @property
    def shape(self):
        return self._shape
    
    @pargs.setter
    def pargs(self, pargs):
        self._pargs = pargs
        
    @grads.setter
    def grads(self, grads):
        self._grads = grads

    def __init__(self, pargs, grads=None):
        self._pargs = pargs
        if grads is None:
            self._grads = np.zeros(pargs.shape, dtype=np.float64)
        else:
            assert grads.shape == pargs.shape
            self._grads = grads
        self._shape = self._pargs.shape

    def __getitem__(self, index):
        return Variable(pargs=self.pargs[index], grads=self.grads[index])

    def __eq__(self, other):
        if (self.pargs == other.pargs).all() and (self.grads == other.grads).all():
            return True
        else:
            return False


if __name__ == "__main__":
    pargs = Variable(np.array([[0, 1, 2], [0.1, 0.2, 0.3]]))

