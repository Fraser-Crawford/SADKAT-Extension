from fit import kelvin_effect, beta
from radial import RadialDroplet
from solution_definitions import aqueous_NaCl
from suspension import Suspension
from dataclasses import dataclass

import numpy as np
from numpy import typing as npt
from scipy.integrate import solve_ivp
from typing_extensions import Self
from droplet import Droplet
from uniform import UniformDroplet

layer_inertia = 1
stiffness = 100
damping = 2.0*np.sqrt(stiffness*layer_inertia)

@dataclass
class SuspensionDroplet(Droplet):
    
    @property
    def volume(self) -> float:
        return self.mass_particles()/self.solution.particle_density+self.total_mass_solvent/self.solution.solvent.density(self.temperature)

    solution: Suspension
    total_mass_solvent: float
    cell_boundaries: npt.NDArray[np.float_]
    cell_velocities: npt.NDArray[np.float_]
    log_mass_particles: npt.NDArray[np.float_]

    def extra_results(self):
        return dict(
            layer_mass_particles=self.log_mass_particles[:],
            layer_concentration=self.layer_concentration,
            layer_boundaries=self.cell_boundaries,
            surface_volume_fraction=np.max(self.layer_volume_fraction()),
            diffusion = self.solution.diffusion(self.temperature),
            peclet = self.peclet,
            predicted_enrichment = self.predicted_enrichment,
            real_enrichment = np.max(self.layer_volume_fraction())/((self.mass_particles()/self.solution.particle_density)/self.volume),
            predicted_surface_concentration = self.predicted_enrichment*(self.mass_particles()/self.solution.particle_mass)/self.volume,
            layer_positions =  np.append(self.cell_boundaries,self.radius),
            all_positions =  np.concatenate(([0],self.cell_boundaries,[self.radius])),
            surface_particle_volume_fraction = self.layer_volume_fraction()[-1],
            average_particle_volume_fraction = (self.mass_particles()/self.solution.particle_density)/self.volume,
            average_particle_concentration = (self.mass_particles()/self.solution.particle_mass)/self.volume,
            surface_particle_concentration = (self.layer_mass_particles[-1]/self.solution.particle_mass)/self.layer_volume[-1],
        )

    @property
    def predicted_enrichment(self):
        Pe = self.peclet
        return np.exp(Pe/2)/(3*beta(Pe))

    @property
    def diffusion(self):
        return self.solution.diffusion(self.temperature)

    @property
    def peclet(self):
        return -self.dmdt()/(self.diffusion*4*np.pi*self.radius*self.solution.solvent.density(self.temperature))

    @property
    def layers(self):
        return len(self.log_mass_particles)

    @staticmethod
    def from_mfp(solution:Suspension, environment, gravity,
                 radius, mass_fraction_particles, temperature, layers=10,
                 velocity=np.zeros(3), position=np.zeros(3)):
        """Create a droplet from experimental conditions.

        Args:
            solution: parameters describing solvent+particles
            environment: parameters of gas surrounding droplet
            gravity: body acceleration in metres/second^2 (3-dimensional vector)
            radius: in metres
            mass_fraction_particles: (MFS) (unitless)
            temperature: in K
            velocity: in metres/second (3-dimensional vector)
            position: in metres (3-dimensional vector)
        """
        volume = 4 * np.pi / 3 * radius ** 3
        mass_particles = mass_fraction_particles*solution.particle_density*volume
        mass_solvent = (1-mass_fraction_particles)*solution.solvent.density(temperature)*volume

        cell_boundaries = radius * np.array([i / layers for i in range(1, layers)])
        concentration = mass_particles / volume
        real_boundaries = np.concatenate(([0], cell_boundaries, [radius]))
        log_mass_solute = np.log(np.array([4 / 3 * np.pi * (r1 ** 3 - r0 ** 3) * concentration for r0, r1 in
                                           zip(real_boundaries, real_boundaries[1:])]))
        cell_velocities = np.zeros(len(cell_boundaries))
        return SuspensionDroplet(solution, environment, gravity, temperature, velocity, position, mass_solvent,
                             cell_boundaries, cell_velocities, log_mass_solute)

    def state(self) -> npt.NDArray[np.float_]:
        return np.hstack((self.cell_boundaries, self.cell_velocities, self.log_mass_particles, self.total_mass_solvent, self.temperature, self.velocity, self.position))

    def split_state(self, state: npt.NDArray[np.float_]):
        cell_boundaries = state[:len(self.cell_boundaries)]
        cell_velocities = state[len(self.cell_boundaries):len(self.cell_velocities)+len(self.cell_boundaries)]
        n = len(self.cell_velocities) + len(self.cell_boundaries) + self.layers
        log_mass_particles = state[len(self.cell_velocities)+len(self.cell_boundaries):n]
        mass_solvent = state[n]
        temp = state[n+1]
        velocity = state[n+2:n+5]
        position = state[n+5:]
        return cell_boundaries, cell_velocities, log_mass_particles, mass_solvent, temp, velocity, position

    def set_state(self, state: npt.NDArray[np.float_]):
        self.cell_boundaries, self.cell_velocities, self.log_mass_particles, self.total_mass_solvent, self.temperature, self.velocity, self.position = self.split_state(state)

    def dxdt(self) -> npt.NDArray[np.float_]:
        return np.hstack((self.boundary_correction(),self.boundary_acceleration(),self.change_in_particles_mass(),self.dmdt(),self.dTdt(),self.dvdt(),self.drdt()))

    @property
    def deviation(self):
        return self.cell_boundaries - self.radius*np.array([i/self.layers for i in range(1,self.layers)])

    def boundary_acceleration(self):
        return (-self.cell_velocities*damping-stiffness*self.deviation/self.radius)/layer_inertia

    def boundary_correction(self):
        return self.cell_velocities

    @property
    def layer_volume(self):
        true_boundaries = np.concatenate(([0], self.cell_boundaries, [self.radius]))
        return np.array([4 / 3 * np.pi * (r1 ** 3 - r0 ** 3) for r0, r1 in zip(true_boundaries, true_boundaries[1:])])

    @property
    def layer_concentration(self):
        return self.layer_mass_particles/self.layer_volume

    @property
    def density(self):
        """Droplet density in kg/m^3."""
        return self.mass/self.volume

    @property
    def refractive_index(self) -> float:
        """Returns the refractive index of the droplet based on a mass fraction/density correction."""
        return self.solution.solvent.refractive_index

    @property
    def mole_fraction_solute(self):
        """Mole fraction of solute (i.e. the non-volatile component).
        NB: Should be zero for pure solvent."""
        return 0.0

    def redistribute(self):
        sign = np.sign(self.cell_velocities)
        volume_corrections = 4*np.pi*(self.cell_boundaries**2*self.cell_velocities)
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

    @property
    def layer_mass_particles(self):
        return np.exp(self.log_mass_particles)

    def get_gradients(self,normalised_boundaries):
        concentrations = self.layer_concentration
        delta_rs = [r1-r0 for r0,r1 in zip(normalised_boundaries,normalised_boundaries[1:])]
        return [(c1-c0)/delta_r for c0,c1,delta_r in zip(concentrations,concentrations[1:],delta_rs)]

    def change_in_particles_mass(self):
        radius = self.radius
        result = self.redistribute()
        full_boundaries = np.concatenate(([0],self.cell_boundaries,[radius]))
        diffusion_coefficient = self.solution.diffusion(self.temperature)
        normalised_boundaries = full_boundaries/radius
        gradients = self.get_gradients(normalised_boundaries)
        diffusion = np.zeros(self.layers)
        for i in range(self.layers-1):
            value = 4*np.pi*radius*diffusion_coefficient*gradients[i]*normalised_boundaries[i+1]**2
            diffusion[i] += value
            diffusion[i+1] -= value
        return (result + diffusion)/self.layer_mass_particles

    def mass_particles(self):
        return np.sum(self.layer_mass_particles)

    def mass_solvent(self) -> float:
        return self.total_mass_solvent

    def surface_solvent_activity(self) -> float:
        return 1.0

    def virtual_droplet(self, x) -> Self:
        cell_boundaries, cell_velocities, layer_mass_particles, total_mass_solvent, temperature, velocity, position = self.split_state(x)
        return SuspensionDroplet(self.solution,self.environment,self.gravity,temperature,velocity,position,total_mass_solvent,cell_boundaries,cell_velocities,layer_mass_particles)

    def convert(self, mass_solvent):
        return UniformDroplet(aqueous_NaCl, self.environment, self.gravity, self.environment.temperature, self.velocity,
                       self.position, mass_solvent, 0.0)

    def mass_solute(self) -> float:
        return 0.0

    def check_for_solidification(self,time,state):
        return self.solution.critical_volume_fraction - np.max(self.virtual_droplet(state).layer_volume_fraction())

    def solver(self, dxdt, time_range, first_step, rtol, events):
        shell_formation = lambda time, x: self.virtual_droplet(x).check_for_solidification(time,x)
        shell_formation.terminal = True
        events.append(shell_formation)
        return solve_ivp(dxdt, time_range, self.state(), first_step=first_step, rtol=rtol, events=events, method="Radau")

    @property
    def concentration(self):
        return self.mass_particles() / self.volume

    @property
    def mass(self):
        return self.mass_solvent() + self.mass_particles()

    def layer_volume_fraction(self):
        volumes = self.layer_mass_particles/self.solution.particle_density
        return volumes/self.layer_volume

    def equilibrium_droplet(self)->UniformDroplet:
        """Given the current state of the droplet and environment, find and return the equilibrium droplet"""
        return self.convert(0.0)