try:
    import torch
except:
    raise Exception("AI-based mapping algorithm need PyTorch to run!")

from .rl_mapping import RlMapping
