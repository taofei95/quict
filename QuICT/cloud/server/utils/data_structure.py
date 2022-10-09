from enum import Enum


class JobOperatorType(Enum):
    restart = "RES",
    stop = "STP",
    delete = "DEL"
