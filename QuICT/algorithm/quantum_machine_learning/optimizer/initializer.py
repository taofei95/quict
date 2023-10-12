# This code is part of numpy_ml.
#                     GNU GENERAL PUBLIC LICENSE
#                        Version 3, 29 June 2007
#
#  Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
#  Everyone is permitted to copy and distribute verbatim copies
#  of this license document, but changing it is not allowed.

import re
from ast import literal_eval as eval

from .optimizer import OptimizerBase, SGD, AdaGrad, RMSProp, Adam
from .scheduler import (
    SchedulerBase,
    ConstantScheduler,
    ExponentialScheduler,
    NoamScheduler,
    KingScheduler,
)


class SchedulerInitializer(object):
    """A class for initializing learning rate schedulers. Valid inputs are:
        (a) __str__ representations of `SchedulerBase` instances
        (b) `SchedulerBase` instances
        (c) Parameter dicts (e.g., as produced via the `summary` method in
            `LayerBase` instances)

    Note:
        If `param` is `None`, return the ConstantScheduler with learning rate
        equal to `lr`.
    """

    def __init__(self, param=None, lr=None):
        if all([lr is None, param is None]):
            raise ValueError("lr and param cannot both be `None`")

        self.lr = lr
        self.param = param

    def __call__(self):
        param = self.param
        if param is None:
            scheduler = ConstantScheduler(self.lr)
        elif isinstance(param, SchedulerBase):
            scheduler = param
        elif isinstance(param, str):
            scheduler = self.init_from_str()
        elif isinstance(param, dict):
            scheduler = self.init_from_dict()
        return scheduler

    def init_from_str(self):
        r = r"([a-zA-Z]*)=([^,)]*)"
        sch_str = self.param.lower()
        kwargs = dict([(i, eval(j)) for (i, j) in re.findall(r, sch_str)])

        if "constant" in sch_str:
            scheduler = ConstantScheduler(**kwargs)
        elif "exponential" in sch_str:
            scheduler = ExponentialScheduler(**kwargs)
        elif "noam" in sch_str:
            scheduler = NoamScheduler(**kwargs)
        elif "king" in sch_str:
            scheduler = KingScheduler(**kwargs)
        else:
            raise NotImplementedError("{}".format(sch_str))
        return scheduler

    def init_from_dict(self):
        S = self.param
        sc = S["hyperparameters"] if "hyperparameters" in S else None

        if sc is None:
            raise ValueError("Must have `hyperparameters` key: {}".format(S))

        if sc and sc["id"] == "ConstantScheduler":
            scheduler = ConstantScheduler().set_params(sc)
        elif sc and sc["id"] == "ExponentialScheduler":
            scheduler = ExponentialScheduler().set_params(sc)
        elif sc and sc["id"] == "NoamScheduler":
            scheduler = NoamScheduler().set_params(sc)
        elif sc:
            raise NotImplementedError("{}".format(sc["id"]))
        return scheduler


class OptimizerInitializer(object):
    """A class for initializing optimizers. Valid inputs are:
        (a) __str__ representations of `OptimizerBase` instances
        (b) `OptimizerBase` instances
        (c) Parameter dicts (e.g., as produced via the `summary` method in
            `LayerBase` instances)

    Note:
        If `param` is `None`, return the SGD optimizer with default parameters.
    """

    def __init__(self, param=None):
        self.param = param

    def __call__(self):
        param = self.param
        if param is None:
            opt = SGD()
        elif isinstance(param, OptimizerBase):
            opt = param
        elif isinstance(param, str):
            opt = self.init_from_str()
        elif isinstance(param, dict):
            opt = self.init_from_dict()
        return opt

    def init_from_str(self):
        r = r"([a-zA-Z]*)=([^,)]*)"
        opt_str = self.param.lower()
        kwargs = dict([(i, eval(j)) for (i, j) in re.findall(r, opt_str)])
        if "sgd" in opt_str:
            optimizer = SGD(**kwargs)
        elif "adagrad" in opt_str:
            optimizer = AdaGrad(**kwargs)
        elif "rmsprop" in opt_str:
            optimizer = RMSProp(**kwargs)
        elif "adam" in opt_str:
            optimizer = Adam(**kwargs)
        else:
            raise NotImplementedError("{}".format(opt_str))
        return optimizer

    def init_from_dict(self):
        O = self.param
        cc = O["cache"] if "cache" in O else None
        op = O["hyperparameters"] if "hyperparameters" in O else None

        if op is None:
            raise ValueError("Must have `hyperparemeters` key: {}".format(O))

        if op and op["id"] == "SGD":
            optimizer = SGD().set_params(op, cc)
        elif op and op["id"] == "RMSProp":
            optimizer = RMSProp().set_params(op, cc)
        elif op and op["id"] == "AdaGrad":
            optimizer = AdaGrad().set_params(op, cc)
        elif op and op["id"] == "Adam":
            optimizer = Adam().set_params(op, cc)
        elif op:
            raise NotImplementedError("{}".format(op["id"]))
        return optimizer
