import numpy as np
from attr import dataclass
from scipy.integrate import solve_ivp
from droplet import Droplet
from environment import Environment
from viscous_solution import ViscousSolution


@dataclass
class RadialDroplet(Droplet):
    """Class for describing a droplet with a non-uniform but radially symmetric composition"""
    solution: ViscousSolution
    environment: Environment
    gravity: np.array  # m/s^2
    temperature: float  # K
    velocity: np.array
    position: np.array
    equilibrium_solvent_mass: np.array
    float_mass_solvent: float  # kg
    log_mass_solute: np.array  # kg

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
            layers: number of droplet layers for diffusion
            velocity: in metres/second (3-dimensional vector)
            position: in metres (3-dimensional vector)
        """
        delta_r = radius / layers
        radii = [(1 + i) * delta_r for i in range(layers)]
        volumes = np.array([4.0 / 3.0 * np.pi * (R ** 3 - r ** 3) for r, R in zip([0] + radii, radii)])
        droplet_volume = 4.0 / 3.0 * np.pi * radius ** 3
        mass = droplet_volume * solution.density(mass_fraction_solute)
        mass_solvent = (1 - mass_fraction_solute) * mass
        log_mass_solute = np.log(mass_fraction_solute * solution.density(mass_fraction_solute) * volumes)
        equilibrium_solvent_mass = (1 - mass_fraction_solute) * solution.density(mass_fraction_solute) * volumes
        return RadialDroplet(solution, environment, gravity, temperature, velocity, position, equilibrium_solvent_mass,
                             mass_solvent, log_mass_solute)

    def set_state(self, state):
        self.float_mass_solvent, self.log_mass_solute, self.temperature, self.velocity, self.position = state[0], state[1:self.initial_layers + 1], \
            state[1 + self.initial_layers], state[2 + self.initial_layers:5 + self.initial_layers], state[5 + self.initial_layers:]

    @property
    def layer_mass_solute(self):
        return np.exp(self.log_mass_solute)

    def mass_solute(self) -> float:
        return np.sum(self.layer_mass_solute)

    def mass_solvent(self) -> float:
        return self.float_mass_solvent

    def volume(self) -> float:
        return np.sum(self.layer_volumes)

    def surface_solvent_activity(self) -> float:
        return self.solution.activity(
            self.layer_mass_fraction_solute[self.outer_layer_index])

    @property
    def outer_layer_solvent_mass(self):
        interior_solvent = np.sum(self.equilibrium_solvent_mass[:self.outer_layer_index])
        return self.float_mass_solvent - interior_solvent

    @property
    def layer_radii(self):
        cumulative_layer_volumes = np.cumsum(self.layer_volumes)
        return np.cbrt(3 * cumulative_layer_volumes / (4 * np.pi))

    def concentration_gradients(self, boundaries):
        concentrations = self.concentrations
        delta_rs = [boundary1-boundary0 for boundary0,boundary1 in zip(boundaries,boundaries[1:])]
        delta_rs = [boundaries[0]] + delta_rs
        return [2.0 * (c1 - c0) / (dr1 + dr2) for c0, c1, dr1, dr2 in
                zip(concentrations, concentrations[1:], delta_rs, delta_rs[1:])]

    @property
    def layer_widths(self):
        radii = self.layer_radii
        result = [self.layer_radii[0]]
        for r0, r1 in zip(radii, radii[1:]):
            result.append(r1 - r0)
        return result

    def correct_derivative(self, mass_solute_derivatives):
        return np.array(
            [0 if mass_solute <= 0 else mass_solute_derivative / mass_solute for mass_solute, mass_solute_derivative in
             zip(self.layer_mass_solute, mass_solute_derivatives)])

    def dCdt(self):
        diffusion = self.diffusion_coefficients
        boundaries = self.layer_radii
        radius = self.radius
        normalised_boundaries = boundaries / radius
        d_plus = np.array([(d1 + d0) / 2.0 for d0, d1 in zip(diffusion, diffusion[1:])])
        values = d_plus * self.concentration_gradients(normalised_boundaries) * normalised_boundaries[:-1] ** 2
        result = np.zeros(self.initial_layers)
        for i in range(self.outer_layer_index):
            result[i] += values[i]
            if i != self.outer_layer_index:
                result[i + 1] -= values[i]
        result *= 4.0 * np.pi * radius
        return self.correct_derivative(result)

    @property
    def outer_layer_index(self):
        accumulator = 0
        for index, solvent_mass in enumerate(self.equilibrium_solvent_mass):
            accumulator += solvent_mass
            if self.float_mass_solvent <= accumulator + solvent_mass * 1e-5:
                return index
        return self.initial_layers - 1

    @property
    def layer_mass_fraction_solute(self):
        result = np.zeros(self.initial_layers)
        solute_mass = self.layer_mass_solute
        for i in range(self.outer_layer_index):
            result[i] += solute_mass[i] / (self.equilibrium_solvent_mass[i] + solute_mass[i])
        result[self.outer_layer_index] = solute_mass[self.outer_layer_index] / (
                self.outer_layer_solvent_mass + solute_mass[self.outer_layer_index])
        return result

    @property
    def outer_layer_number(self):
        return self.outer_layer_index + 1

    @property
    def concentrations(self):
        layer_mfs = self.layer_mass_fraction_solute
        return layer_mfs * self.solution.density(layer_mfs)

    @property
    def initial_layers(self):
        return len(self.log_mass_solute)

    @property
    def diffusion_coefficients(self) -> np.array:
        return self.solution.diffusion(self.layer_mass_fraction_solute, self.temperature)

    @property
    def layer_volumes(self):
        result = np.zeros(self.initial_layers)
        solute_mass = self.layer_mass_solute
        concentrations = self.concentrations
        for i in range(self.outer_layer_number):
            result[i] += solute_mass[i] / concentrations[i]
        return result

    def dxdt(self):
        return np.hstack((self.dmdt(), self.dCdt(), self.dTdt(), self.dvdt(), self.drdt()))

    def virtual_droplet(self, x):
        x = (x[0], x[1:1 + self.initial_layers], x[1 + self.initial_layers],
             x[2 + self.initial_layers:5 + self.initial_layers], x[5 + self.initial_layers:])
        return RadialDroplet(self.solution, self.environment, self.gravity, x[2], x[3], x[4],
                             self.equilibrium_solvent_mass, x[0], x[1])

    def solver(self, dxdt, time_range, first_step, rtol, events):
        return solve_ivp(dxdt, time_range, self.state(), first_step=first_step, rtol=rtol, events=events, method="Radau")

    def state(self) -> np.array:
        return np.hstack((self.float_mass_solvent, self.log_mass_solute, self.temperature, self.velocity, self.position))
