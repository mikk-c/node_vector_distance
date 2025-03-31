"""
Node Vector Distance
========

A package to calculate node attribute distances and statistics over
a complex network.

See [TODO] for complete documentation.
"""

import torch

__version__ = "0.0.20"
__device__ = "cuda" if torch.cuda.is_available() else "cpu"

from .utils import *
from .variance import *
from .distances import *
from .correlation import *
from .data_preparation import *