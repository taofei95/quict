from enum import Enum


class JobOperatorType(Enum):
    restart = "RES"
    stop = "STP"
    delete = "DEL"
    user_delete = "UDL"


class ResourceOp(Enum):
    Allocation = "ALC"
    Release = "RES"


class JobState(Enum):
    Pending = "pending"
    Running = "running"
    Finish = "finish"

    # Un-normal State
    Error = "error"
    Stop = "stop"