from enum import Enum


class CircuitReprEnum(Enum):
    DAG = 1
    MAT_SEQ = 2

CIRCUIT_REPR_API = CircuitReprEnum.DAG
