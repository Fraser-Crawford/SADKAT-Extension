from fit import beta
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
damping = 2.0 * np.sqrt(stiffness * layer_inertia)


@dataclass
class SuspensionDroplet(Droplet):

    @property
    def volume(self) -> float:
        return self.mass_particles() / self.solution.particle_density + self.total_mass_solvent / self.solution.solvent.density(
            self.temperature)

    solution: Suspension
    total_mass_solvent: float
    cell_boundaries: npt.NDArray[np.float_]
    cell_velocities: npt.NDArray[np.float_]
    log_mass_particles: npt.NDArray[np.float_]

    def extra_results(self):
        return dict(
            layer_mass_particles=self.log_mass_particles[:],
            average_layer_concentrations=self.average_layer_concentrations,
            layer_boundaries=self.cell_boundaries,
            surface_volume_fraction=np.max(self.layer_volume_fraction()),
            layer_concentrations=self.linear_layer_concentrations(),
            diffusion=self.solution.diffusion(self.temperature),
            peclet=self.peclet,
            predicted_enrichment=self.predicted_enrichment,
            real_enrichment=np.max(self.layer_volume_fraction()) / (
                    (self.mass_particles() / self.solution.particle_density) / self.volume),
            predicted_surface_concentration=self.predicted_enrichment * (
                    self.mass_particles() / self.solution.particle_mass) / self.volume,
            layer_positions=np.append(self.cell_boundaries, self.radius),
            all_positions=np.concatenate(([0], self.cell_boundaries, [self.radius])),
            surface_particle_volume_fraction=self.layer_volume_fraction()[-1],
            average_particle_volume_fraction=(self.mass_particles() / self.solution.particle_density) / self.volume,
            average_particle_concentration=(self.mass_particles() / self.solution.particle_mass) / self.volume,
            surface_particle_concentration=(self.layer_mass_particles[-1] / self.solution.particle_mass) /
                                           self.layer_volume[-1],
            mass_particles=self.mass_particles(),
        )

    @property
    def predicted_enrichment(self):
        Pe = self.peclet
        return np.exp(Pe / 2) / (3 * beta(Pe))

    @property
    def diffusion(self):
        return self.solution.diffusion(self.temperature)

    @property
    def peclet(self):
        return -self.dmdt() / (
                self.diffusion * 4 * np.pi * self.radius * self.solution.solvent.density(self.temperature))

    @property
    def layers(self):
        return len(self.log_mass_particles)

    @staticmethod
    def from_mfp(solution: Suspension, environment, gravity,
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
        mass_particles = mass_fraction_particles * solution.particle_density * volume
        mass_solvent = (1 - mass_fraction_particles) * solution.solvent.density(temperature) * volume

        cell_boundaries = radius * np.array([i / layers for i in range(1, layers)])
        concentration = mass_particles / volume
        real_boundaries = np.concatenate(([0], cell_boundaries, [radius]))
        log_mass_solute = np.log(np.array([4 / 3 * np.pi * (r1 ** 3 - r0 ** 3) * concentration for r0, r1 in
                                           zip(real_boundaries, real_boundaries[1:])]))
        cell_velocities = np.zeros(len(cell_boundaries))
        return SuspensionDroplet(solution, environment, gravity, temperature, velocity, position, mass_solvent,
                                 cell_boundaries, cell_velocities, log_mass_solute)

    def state(self) -> npt.NDArray[np.float_]:
        return np.hstack((self.cell_boundaries, self.cell_velocities, self.log_mass_particles, self.total_mass_solvent,
                          self.temperature, self.velocity, self.position))

    def linear_layer_concentrations(self):
        rs = np.concatenate((self.cell_boundaries, [self.radius]))
        r0s = rs[:-1]
        r1s = rs[1:]
        r03s = r0s ** 3
        r04s = r0s ** 4
        r13s = r1s ** 3
        r14s = r1s ** 4
        masses = self.layer_mass_particles
        c0 = 3 * masses[0] / (4 * np.pi * r03s[0])
        c = [c0, c0]

        for r03, r04, r13, r14, mass, r0, r1 in zip(r03s, r04s, r13s, r14s, masses[1:], r0s, r1s):
            numerator = mass / np.pi + 4 / 3 * c[-1] * (r03 - r13)
            denominator = r14 - r04 + 4 / 3 * r0 * (r03 - r13)
            gradient = numerator / denominator
            c.append(gradient * (r1 - r0) + c[-1])

        return c

    def split_state(self, state: npt.NDArray[np.float_]):
        cell_boundaries = state[:len(self.cell_boundaries)]
        cell_velocities = state[len(self.cell_boundaries):len(self.cell_velocities) + len(self.cell_boundaries)]
        n = len(self.cell_velocities) + len(self.cell_boundaries) + self.layers
        log_mass_particles = state[len(self.cell_velocities) + len(self.cell_boundaries):n]
        mass_solvent = state[n]
        temp = state[n + 1]
        velocity = state[n + 2:n + 5]
        position = state[n + 5:]
        return cell_boundaries, cell_velocities, log_mass_particles, mass_solvent, temp, velocity, position

    def set_state(self, state: npt.NDArray[np.float_]):
        self.cell_boundaries, self.cell_velocities, self.log_mass_particles, self.total_mass_solvent, self.temperature, self.velocity, self.position = self.split_state(
            state)

    def dxdt(self, time) -> npt.NDArray[np.float_]:
        return np.hstack((self.boundary_correction(), self.boundary_acceleration(), self.change_in_particles_mass(),
                          self.dmdt(), self.dTdt(), self.dvdt(), self.drdt()))

    @property
    def deviation(self):
        return self.cell_boundaries - self.radius * np.array([i / self.layers for i in range(1, self.layers)])

    def boundary_acceleration(self):
        return (-self.cell_velocities * damping - stiffness * self.deviation / self.radius) / layer_inertia

    def boundary_correction(self):
        return self.cell_velocities

    @property
    def layer_volume(self):
        true_boundaries = np.concatenate(([0], self.cell_boundaries, [self.radius]))
        return np.array([4 / 3 * np.pi * (r1 ** 3 - r0 ** 3) for r0, r1 in zip(true_boundaries, true_boundaries[1:])])

    @property
    def average_layer_concentrations(self):
        return self.layer_mass_particles / self.layer_volume

    @property
    def density(self):
        """Droplet density in kg/m^3."""
        return self.mass / self.volume

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
        volume_corrections = 4 * np.pi * (self.cell_boundaries ** 2 * self.cell_velocities)
        concentrations = self.linear_layer_concentrations()[1:]
        result = np.zeros(self.layers)

        for i in range(len(volume_corrections)):
            if sign[i] < 0:
                value = volume_corrections[i] * concentrations[i]
                result[i] += value
                result[i + 1] -= value
            else:
                value = volume_corrections[i] * concentrations[i + 1]
                result[i] += value
                result[i + 1] -= value

        return result

    @property
    def settle(self):
        """
        Due to the relatively massive nature of the nano-particles suspended, gravitational mixing
        shouldn't be ignored. This scheme estimates the terminal velocity of the particles in the fluid
        and fluxes them to the layer below. Note that this can cause concentration for very large
        nano-particles and extra mixing for small particles.
        """
        delta_rho = self.solution.particle_density - self.solution.solvent.density(self.temperature)
        terminal_velocity = (2 * 9.81 * self.solution.particle_radius ** 2 * delta_rho
                             / (9 * self.solution.viscosity(self.temperature)))
        particle_mass = self.solution.particle_mass
        areas = 4 * np.pi * self.cell_boundaries ** 2
        concentrations = self.linear_layer_concentrations()
        top_half = terminal_velocity/particle_mass*concentrations[1:-1]*areas
        bottom_half = terminal_velocity / particle_mass * concentrations[:-2] * areas
        return  bottom_half-top_half

    @property
    def layer_mass_particles(self):
        return np.exp(self.log_mass_particles)

    def get_gradients(self, normalised_boundaries):
        concentrations = self.linear_layer_concentrations()
        return np.array([(c2 - c0) / (r2 - r0) for r0, r2, c0, c2 in
                         zip(normalised_boundaries[:-2], normalised_boundaries[2:], concentrations[:-2],
                             concentrations[2:])])

    def change_in_particles_mass(self):
        radius = self.radius
        full_boundaries = np.concatenate(([0], self.cell_boundaries, [radius]))
        diffusion_coefficient = self.solution.diffusion(self.temperature)
        normalised_boundaries = full_boundaries / radius
        gradients = self.get_gradients(normalised_boundaries)
        diffusion = np.zeros(self.layers)
        for i in range(self.layers - 1):
            value = 4 * np.pi * radius * diffusion_coefficient * gradients[i] * normalised_boundaries[i + 1] ** 2
            diffusion[i] += value
            diffusion[i + 1] -= value
        return (self.redistribute() + diffusion + self.settle) / self.layer_mass_particles

    def mass_particles(self):
        return np.sum(self.layer_mass_particles)

    def mass_solvent(self) -> float:
        return self.total_mass_solvent

    def surface_solvent_activity(self) -> float:
        return 1.0

    def virtual_droplet(self, x) -> Self:
        cell_boundaries, cell_velocities, layer_mass_particles, total_mass_solvent, temperature, velocity, position = self.split_state(
            x)
        return SuspensionDroplet(self.solution, self.environment, self.gravity, temperature, velocity, position,
                                 total_mass_solvent, cell_boundaries, cell_velocities, layer_mass_particles)

    def convert(self, mass_solvent):
        return UniformDroplet(aqueous_NaCl, self.environment, self.gravity, self.environment.temperature, self.velocity,
                              self.position, mass_solvent, 0.0)

    def mass_solute(self) -> float:
        return 0.0

    def check_for_solidification(self):
        return self.solution.critical_volume_fraction - self.linear_layer_concentrations()[
            -1] / self.solution.particle_density

    def solver(self, dxdt, time_range, first_step, rtol, events):
        shell_formation = lambda time, x: self.virtual_droplet(x).check_for_solidification()
        shell_formation.terminal = True
        events.append(shell_formation)
        return solve_ivp(dxdt, time_range, self.state(), first_step=first_step, rtol=rtol, events=events,
                         method="Radau")

    @property
    def concentration(self):
        return self.mass_particles() / self.volume

    @property
    def mass(self):
        return self.mass_solvent() + self.mass_particles()

    @property
    def surface_volume_fraction(self):
        return self.linear_layer_concentrations()[-1] / self.solution.particle_density

    def layer_volume_fraction(self):
        volumes = self.layer_mass_particles / self.solution.particle_density
        return volumes / self.layer_volume

    def equilibrium_droplet(self) -> UniformDroplet:
        """Given the current state of the droplet and environment, find and return the equilibrium droplet"""
        return self.convert(0.0)
