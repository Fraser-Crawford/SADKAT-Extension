from dataclasses import dataclass

import numpy as np
from numpy import typing as npt
from scipy.integrate import solve_ivp
from typing_extensions import Self
from droplet import Droplet
from uniform import UniformDroplet
from viscous_solution import ViscousSolution

layer_inertia = 1.0
stiffness = 1.0
damping = 2*np.sqrt(stiffness*layer_inertia)

@dataclass
class RadialDroplet(Droplet):
    solution: ViscousSolution
    total_mass_solvent: float
    cell_boundaries: npt.NDArray[np.float_]
    cell_velocities: npt.NDArray[np.float_]
    layer_mass_solute: npt.NDArray[np.float_]

    @property
    def layers(self):
        return len(self.layer_mass_solute)

    @staticmethod
    def from_mfs(solution, environment, gravity,
                 radius, mass_fraction_solute, temperature, layers=10,
                 velocity=np.zeros(3), position=np.zeros(3)):
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
        layer_solvent_masses = np.full(mass_solvent/layers,layers)
        cell_boundaries = radius/np.array([i for i in range(layers,0,-1)])
        concentrations = solution.concentration(mass_solute/(mass_solute+layer_solvent_masses))
        return RadialDroplet(solution, environment, gravity, temperature, velocity, position, mass_solvent,
                              mass_solute,cell_boundaries,concentrations)

    def state(self) -> npt.NDArray[np.float_]:
        return np.hstack((self.cell_boundaries, self.cell_velocities, self.layer_mass_solute, self.total_mass_solvent, self.temperature, self.velocity, self.position))

    def split_state(self, state: npt.NDArray[np.float_]):
        cell_boundaries = state[:len(self.cell_boundaries)]
        cell_velocities = state[len(self.cell_boundaries):len(self.cell_velocities)+len(self.cell_boundaries)]
        n = len(self.cell_velocities) + len(self.cell_boundaries) + self.layers
        mass_solute = state[len(self.cell_velocities)+len(self.cell_boundaries):n]
        mass_solvent = state[n]
        temp = state[n+1]
        velocity = state[n+2:n+5]
        position = state[n+5:]
        return cell_boundaries, cell_velocities, mass_solute, mass_solvent, temp, velocity, position

    def set_state(self, state: npt.NDArray[np.float_]):
        self.cell_boundaries, self.cell_velocities, self.layer_mass_solute, self.total_mass_solvent, self.temperature, self.velocity, self.position = self.split_state(state)

    def dxdt(self) -> npt.NDArray[np.float_]:
        return np.hstack((self.boundary_correction(),self.boundary_acceleration(),self.change_in_solute_mass(),self.dmdt(),self.dTdt,self.dvdt(),self.drdt()))

    @property
    def deviation(self):
        return self.cell_boundaries - self.radius/np.array([i for i in range(self.layers,0,-1)])

    def boundary_acceleration(self):
        return (self.cell_velocities*damping-stiffness*self.deviation/self.radius)/layer_inertia

    def boundary_correction(self):
        return self.cell_velocities

    @property
    def layer_volume(self):
        true_boundaries = np.concatenate(([0],self.cell_boundaries,[self.radius]))
        return np.array([4/3*np.pi*(r1**3-r0**3) for r0,r1 in zip(true_boundaries,true_boundaries[1:])])

    @property
    def layer_concentration(self):
        return self.layer_mass_solute/self.layer_volume

    def change_in_solute_mass(self):
        sign = np.sign(self.cell_velocities)
        volume_corrections = 4/3*np.pi*(self.cell_boundaries**2*self.cell_velocities)
        concentrations = self.layer_concentration
        result = np.zeros(self.layers)

        for i in range(len(volume_corrections)):
            if sign[i] < 0:
                value = volume_corrections[i]*concentrations[i]
                result[i] += value
                result[i+1] -= value
            else:
                value = volume_corrections[i] * concentrations[i+1]
                result[i] += value
                result[i+1] -= value

        return result

    def mass_solute(self):
        return np.sum(self.layer_mass_solute)

    def mass_solvent(self) -> float:
        return self.total_mass_solvent

    def surface_solvent_activity(self) -> float:
        return self.solution.activity(self.solution.concentration_to_solute_mass_fraction(self.layer_concentration[-1]))

    def virtual_droplet(self, x) -> Self:
        cell_boundaries, cell_velocities, layer_mass_solute, total_mass_solvent, temperature, velocity, position = self.split_state(x)
        return RadialDroplet(self.solution,self.environment,self.gravity,temperature,velocity,position,total_mass_solvent,cell_boundaries,cell_velocities,layer_mass_solute)

    def convert(self, mass_water):
        return UniformDroplet(self.solution, self.environment, self.gravity, self.environment.temperature, self.velocity,
                       self.position, mass_water, self.mass_solute())

    def solver(self, dxdt, time_range, first_step, rtol, events):
        unstable = lambda time, x: self.virtual_droplet(x).radius - self.virtual_droplet(x).cell_boundaries[-1]
        unstable.terminating = True
        events.append(unstable)
        return solve_ivp(dxdt, time_range, self.state(), first_step=first_step, rtol=rtol, events=events, method="Radau")