# This code is part of numpy_ml.
#                     GNU GENERAL PUBLIC LICENSE
#                        Version 3, 29 June 2007
#
#  Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
#  Everyone is permitted to copy and distribute verbatim copies
#  of this license document, but changing it is not allowed.

from copy import deepcopy
from abc import ABC, abstractmethod

import numpy as np

from math import erf


def gaussian_cdf(x, mean, var):
    """Compute the probability that a random draw from a 1D Gaussian with mean
    `mean` and variance `var` is less than or equal to `x`.
    """
    eps = np.finfo(float).eps
    x_scaled = (x - mean) / np.sqrt(var + eps)
    return (1 + erf(x_scaled / np.sqrt(2))) / 2


class SchedulerBase(ABC):
    def __init__(self):
        """Abstract base class for all Scheduler objects."""
        self.hyperparameters = {}

    def __call__(self, step=None, cur_loss=None):
        return self.learning_rate(step=step, cur_loss=cur_loss)

    def copy(self):
        """Return a copy of the current object."""
        return deepcopy(self)

    def set_params(self, hparam_dict):
        """Set the scheduler hyperparameters from a dictionary."""
        if hparam_dict is not None:
            for k, v in hparam_dict.items():
                if k in self.hyperparameters:
                    self.hyperparameters[k] = v
        return self

    @abstractmethod
    def learning_rate(self, step=None):
        raise NotImplementedError


class ConstantScheduler(SchedulerBase):
    """Returns a fixed learning rate, regardless of the current step.

    Args:
        lr (float, optional): The learning rate. Defaults to 0.01.
    """

    def __init__(self, lr=0.01, **kwargs):
        """Initialize a ConstantScheduler instance."""
        super().__init__()
        self.lr = lr
        self.hyperparameters = {"id": "ConstantScheduler", "lr": self.lr}

    def __str__(self):
        return "ConstantScheduler(lr={})".format(self.lr)

    def learning_rate(self, **kwargs):
        """Return the current learning rate.

        Returns:
            float: The learning rate.
        """
        return self.lr


class ExponentialScheduler(SchedulerBase):
    """An exponential learning rate scheduler.

    Note:
        The exponential scheduler decays the learning rate by `decay` every
        `stage_length` steps, starting from `initial_lr`:

        ``` python
        learning_rate = initial_lr * decay ** curr_stage
        ```

        where:

        ``` python
        curr_stage = step / stage_length          if staircase = False
        curr_stage = floor(step / stage_length)   if staircase = True
        ```

    Args:
        initial_lr (float, optional): The learning rate at the first step. Defaults to 0.01.
        stage_length (int, optional): The length of each stage, in steps. Defaults to 500.
        staircase (bool, optional): If True, only adjusts the learning rate at the stage transitions,
            producing a step-like decay schedule. If False, adjusts the
            learning rate after each step, creating a smooth decay schedule. Defaults to False.
        decay (float, optional): The amount to decay the learning rate at each new stage.
            Defaults to 0.1.
    """

    def __init__(
        self, initial_lr=0.01, stage_length=500, staircase=False, decay=0.1, **kwargs
    ):
        """Initialize an ExponentialScheduler instance."""
        super().__init__()
        self.decay = decay
        self.staircase = staircase
        self.initial_lr = initial_lr
        self.stage_length = stage_length
        self.hyperparameters = {
            "id": "StepScheduler",
            "decay": self.decay,
            "staircase": self.staircase,
            "initial_lr": self.initial_lr,
            "stage_length": self.stage_length,
        }

    def __str__(self):
        return "ExponentialScheduler(initial_lr={}, stage_length={}, staircase={}, decay={})".format(
            self.initial_lr, self.stage_length, self.staircase, self.decay
        )

    def learning_rate(self, step, **kwargs):
        """Return the current learning rate as a function of `step`.

        Args:
            step (int): The current step number.

        Returns:
            float: The learning rate for the current step.
        """
        cur_stage = step / self.stage_length
        if self.staircase:
            cur_stage = np.floor(cur_stage)
        return self.initial_lr * self.decay**cur_stage


class NoamScheduler(SchedulerBase):
    """The Noam learning rate scheduler, originally used in conjunction with
    the Adam optimizer in [1].

    Note:
        The Noam scheduler increases the learning rate linearly for the first
        `warmup_steps` steps, and decreases it thereafter proportionally to the
        inverse square root of the step number:

        ``` python
        lr = scale_factor * ( (model_dim ** (-0.5)) * adj_step )
        adj_step = min(step_num ** (-0.5), step_num * warmup_steps ** (-1.5))
        ```

    References:
        [1] Vaswani et al. (2017) "Attention is all you need". *31st
           Conference on Neural Information Processing Systems*,
           <https://arxiv.org/pdf/1706.03762.pdf>

    Args:
        model_dim (int, optional): The number of units in the layer output.
            Defaults to 512.
        scale_factor (int, optional): A fixed coefficient for rescaling the final
            learning rate. Defaults to 1.
        warmup_steps (int, optional): The number of steps in the warmup stage
            of training. Defaults to 4000.
    """

    def __init__(self, model_dim=512, scale_factor=1, warmup_steps=4000, **kwargs):
        """Initialize a NoamScheduler instance."""
        super().__init__()
        self.model_dim = model_dim
        self.scale_factor = scale_factor
        self.warmup_steps = warmup_steps
        self.hyperparameters = {
            "id": "NoamScheduler",
            "model_dim": self.model_dim,
            "scale_factor": self.scale_factor,
            "warmup_steps": self.warmup_steps,
        }

    def __str__(self):
        return "NoamScheduler(model_dim={}, scale_factor={}, warmup_steps={})".format(
            self.model_dim, self.scale_factor, self.warmup_steps
        )

    def learning_rate(self, step, **kwargs):
        warmup, d_model = self.warmup_steps, self.model_dim
        new_lr = d_model ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
        return self.scale_factor * new_lr


class KingScheduler(SchedulerBase):
    """The Davis King / DLib learning rate scheduler.

    Note:
        The KingScheduler computes the probability that the slope of the OLS
        fit to the loss history is negative. If the probability that it is
        negative is less than 51% over the last `patience` steps, the scheduler
        exponentially decreases the current learning rate by `decay`.

    References:
        King, D. (2018). "Automatic learning rate scheduling that really works".
        <http://blog.dlib.net/2018/02/automatic-learning-rate-scheduling-that.html>

    Args:
        initial_lr (float, optional): The learning rate to begin at. Defaults to 0.01.
        patience (int, optional): Amount of time to maintain the current learning
            rate without a decrease in loss before adjustment. Defaults to 1000.
        decay (float, optional): The amount to decay the learning rate at each
            new stage. Defaults to 0.99.
    """

    def __init__(self, initial_lr=0.01, patience=1000, decay=0.99, **kwargs):
        """Initialize a KingScheduler instance."""
        super().__init__()
        self.decay = decay
        self.patience = patience
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.max_history = np.ceil(1.1 * (patience + 1)).astype(int)

        self.loss_history = []
        self.hyperparameters = {
            "id": "KingScheduler",
            "decay": self.decay,
            "patience": self.patience,
            "initial_lr": self.initial_lr,
        }

    def __str__(self):
        return "KingScheduler(initial_lr={}, patience={}, decay={})".format(
            self.initial_lr, self.patience, self.decay
        )

    def _steps_without_decrease(self, robust=False, check_all=False):
        """Returns the maximum number of timesteps for which `P(loss is decreasing)
        < 0.51`.

        Args:
            robust (bool, optional): If `robust=True`, first filter out the largest
                10% of the loss values to remove transient spikes in the loss due to,
                e.g., a few bad minibatches. Defaults to False.
            check_all (bool, optional): If False, returns the maximum number of
                timesteps for which P(loss is decreasing) < 0.51. If True, only checks
                whether the number of timesteps for which P(loss is decreasing) < 0.51
                is equal to `self.patience``. The former provides more information but
                is significantly more computationally expensive. Defaults to False.

        Returns:
            int: The maximum number of steps back in loss_history for which P(loss
                is decreasing) < 0.51.
        """
        lh = np.array(self.loss_history)

        # drop top 10% of loss values to filter out large loss spikes
        if robust:
            thresh = np.quantile(lh, 0.9)
            lh = np.array([i for i in lh if i <= thresh])

        N = len(lh)
        steps_without_decrease = 0
        if check_all:
            for i in reversed(range(N - 2)):
                if self._p_decreasing(lh, i) < 0.51:
                    steps_without_decrease = N - i
        else:
            i = max(0, N - self.patience - 1)
            if self._p_decreasing(lh, i) < 0.51:
                steps_without_decrease = N - i
        return steps_without_decrease

    def _p_decreasing(self, loss_history, i):
        """Compute the probability that the slope of the OLS fit to the loss
        history is negative.

        Args:
            loss_history (np.ndarray): The sequence of loss values for the previous
                `N` minibatches.
            i (int): Compute P(Slope < 0) beginning at index i in `history`.

        Returns:
            float: The probability that the slope of the OLS fit to loss_history is
                less than or equal to 0.
        """
        loss = loss_history[i:]
        N = len(loss)

        # perform OLS on the loss entries to calc the slope mean
        X = np.c_[np.ones(N), np.arange(i, len(loss_history))]
        intercept, s_mean = np.linalg.inv(X.T @ X) @ X.T @ loss
        loss_pred = s_mean * X[:, 1] + intercept

        # compute the variance of our loss predictions and use this to compute
        # the (unbiased) estimate of the slope variance
        loss_var = 1 / (N - 2) * np.sum((loss - loss_pred) ** 2)
        s_var = (12 * loss_var) / (N**3 - N)

        # compute the probability that a random sample from a Gaussian
        # parameterized by s_mean and s_var is less than or equal to 0
        p_decreasing = gaussian_cdf(0, s_mean, s_var)
        return p_decreasing

    def learning_rate(self, step, cur_loss):
        """Compute the updated learning rate for the current step and loss.

        Args:
            step (int): The current step number. Unused.
            cur_loss (float): The loss at the current step.

        Returns:
            float: The learning rate for the current step.
        """
        if cur_loss is None:
            raise ValueError("cur_loss must be a float, but got {}".format(cur_loss))

        # this happens if we initialize the scheduler from a string / dict
        if not hasattr(self, "max_history"):
            self.max_history = np.ceil(1.1 * (self.patience + 1)).astype(int)
        patience, max_history = self.patience, self.max_history

        self.loss_history.append(cur_loss)
        if len(self.loss_history) < patience:
            return self.current_lr
        self.loss_history = self.loss_history[-max_history:]

        # if the loss has not decreased for `patience` timesteps, drop the
        # learning rate
        if (
            self._steps_without_decrease() > patience
            and self._steps_without_decrease(robust=True) > patience
        ):
            self.current_lr *= self.decay

        return self.current_lr
