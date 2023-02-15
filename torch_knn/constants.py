from enum import Enum


class Metric(Enum):
    L2 = "L2"
    IP = "IP"
    COS = "cos"


class CentroidsInit(Enum):
    RANDOM = "random"
    KMEANSPP = "kmeanspp"
