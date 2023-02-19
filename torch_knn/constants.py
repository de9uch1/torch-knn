from enum import Enum
from typing import Tuple

import torch


class CentroidsInit(Enum):
    RANDOM = "random"
    KMEANSPP = "kmeanspp"
