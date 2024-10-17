from dataclasses import dataclass
from typing import Self

from numpy import typing as npt

from droplet import Droplet
from solution import Solution


@dataclass
class UniformDroplet(Droplet):
    solution: Solution
    mass_solute: float
    mass_solvent: float
    def state(self) -> npt.NDArray:
        pass

    def set_state(self, state):
        pass

    def dxdt(self):
        pass

    def dmdt(self):
        pass

    def mass_solute(self) -> float:
        pass

    def mass_solvent(self) -> float:
        pass

    def volume(self) -> float:
        pass

    def surface_solvent_activity(self) -> float:
        pass

    def virtual_droplet(self, x) -> Self:
        pass