from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from solution import Solution
from scipy.constants import k

@dataclass
class ViscousSolution(Solution):
    diffusion: Callable[[float|npt.NDArray[np.float_], float], float|npt.NDArray[np.float_]]

    def viscosity(self,mfs, temperature):
        return self.diffusion(mfs,temperature)/(k*temperature)