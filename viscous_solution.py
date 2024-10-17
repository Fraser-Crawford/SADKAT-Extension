from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from solution import Solution


@dataclass
class ViscousSolution(Solution):
    diffusion: Callable[[float|np.array, float], float]