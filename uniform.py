from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy import typing as npt
from droplet import Droplet
from solution import Solution


@dataclass
class UniformDroplet(Droplet):
    solution: Solution
    mass_solvent: float
    mass_solute: float
    def state(self) -> npt.NDArray:
        return np.hstack((self.mass_solvent,self.temperature,self.velocity,self.position))

    def set_state(self, state):
        self.mass_solvent, self.temperature, self.velocity, self.position = state[0], state[1], state[2:5], state[5:]

    def dxdt(self):
        return np.hstack((self.dmdt, self.dTdt, self.dvdt, self.drdt))

    def mass_solute(self) -> float:
        return self.mass_solute

    def mass_solvent(self) -> float:
        return self.mass_solvent

    def volume(self) -> float:
        return self.mass / self.density

    def surface_solvent_activity(self) -> float:
        return self.solution.solvent_activity_from_mass_fraction_solute(self.mass_fraction_solute)

    def virtual_droplet(self, x) -> Self:
        x = (x[0], x[1], x[2:5], x[5:])
        return UniformDroplet(self.solution, self.environment,self.gravity,x[1],x[2],x[3],x[0],self.mass_solute)