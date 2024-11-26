from dataclasses import dataclass

from fluids.constants import gas_constant
from scipy.integrate import solve_ivp
import scipy
from typing_extensions import Self

import numpy as np
from numpy import typing as npt
from droplet import Droplet

@dataclass
class UniformDroplet(Droplet):

    def refractive_index(self):
        return self.solution.refractive_index(self.mass_fraction_solute,self.temperature)

    @property
    def volume(self) -> float:
        return self.mass/self.density

    def extra_results(self):
        return dict()

    float_mass_solvent: float
    float_mass_solute: float

    def convert(self, mass_water):
        return UniformDroplet(self.solution, self.environment, self.gravity, self.stationary,self.environment.temperature, self.velocity,
                       self.position, mass_water, self.mass_solute())

    def solver(self, dxdt, time_range, first_step, rtol, events):
        return solve_ivp(dxdt, time_range, self.state(), first_step=first_step, rtol=rtol, events=events)

    @staticmethod
    def from_mfs(solution, environment, gravity,
                 radius, mass_fraction_solute, temperature,
                 velocity=np.zeros(3), position=np.zeros(3), stationary=True):
        """Create a droplet from experimental conditions.

        Args:
            solution: parameters describing solvent+solute
            environment: parameters of gas surrounding droplet
            gravity: body acceleration in metres/second^2 (3-dimensional vector)
            radius: in metres
            mass_fraction_solute: (MFS) (unitless)
            temperature: in K
            velocity: in metres/second (3-dimensional vector)
            position: in metres (3-dimensional vector)
        """
        mass = 4 * np.pi / 3 * radius ** 3 * solution.density(mass_fraction_solute)
        mass_solvent = (1 - mass_fraction_solute) * mass
        mass_solute = mass_fraction_solute * mass
        return UniformDroplet(solution, environment,gravity,stationary,temperature,velocity,position,mass_solvent,mass_solute)

    def state(self) -> npt.NDArray:
        return np.hstack((self.float_mass_solvent, self.temperature, self.velocity, self.position))

    def set_state(self, state:npt.NDArray[np.float_]):
        self.float_mass_solvent, self.temperature, self.velocity, self.position = state[0], state[1], state[2:5], state[5:]

    def dxdt(self,time)->npt.NDArray[np.float_]:
        return np.hstack((self.dmdt(), self.dTdt(), self.dvdt(), self.drdt()))

    def mass_solute(self) -> float:
        return self.float_mass_solute

    def mass_solvent(self) -> float:
        return self.float_mass_solvent

    def surface_solvent_activity(self) -> float:
        return self.solution.activity(self.mass_fraction_solute)

    def virtual_droplet(self, x) -> Self:
        x = (x[0], x[1], x[2:5], x[5:])
        return UniformDroplet(self.solution, self.environment, self.gravity, self.stationary,x[1], x[2], x[3], x[0], self.float_mass_solute)