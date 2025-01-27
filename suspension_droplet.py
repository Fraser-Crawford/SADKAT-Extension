from fit import beta, correct_radius
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


def reverse_planck(x, a, b, c, d):
    X = x - 1
    return a * (-X) ** b / (np.exp(-c * X) - 1) + d * X ** 2


def crossing_rate(normalised_boundaries: npt.NDArray[np.float_], radius: float) -> npt.NDArray[np.float_]:
    a_poly = np.poly1d([2.19783107e-04, -3.51635586e-02, 2.38920835e+00, -8.93283977e+01,
                        1.98888346e+03, -2.66112346e+04, 2.05816461e+05, -8.23708581e+05,
                        1.31175102e+06])
    b_poly = np.poly1d([2.53448443e-09, -3.85291781e-07, 2.47342925e-05, -8.69224399e-04,
                        1.81006090e-02, -2.25349160e-01, 1.60144207e+00, -5.63447673e+00,
                        1.01377002e+01])
    c_poly = np.poly1d([1.05469823e-08, -1.56729633e-06, 9.80793232e-05, -3.34821449e-03,
                        6.74374840e-02, -8.08136859e-01, 5.51197720e+00, -1.88149952e+01,
                        3.49201721e+01])
    d_poly = np.poly1d([4.02218255e-09, -5.59484149e-07, 3.25011115e-05, -1.02161469e-03,
                        1.88471472e-02, -2.07924035e-01, 1.35272436e+00, -5.00837063e+00,
                        1.01290991e+01])
    return reverse_planck(normalised_boundaries, a_poly(radius * 1e6), b_poly(radius * 1e6), c_poly(radius * 1e6),
                          d_poly(radius * 1e6)) / 2


@dataclass
class SuspensionDroplet(Droplet):

    def measured_radius(self) -> float:
        return correct_radius(self.radius, self.solution.solvent.refractive_index, 1.335)

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
            diffusion=self.diffusion_coefficient,
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
    def diffusion_coefficient(self):
        return self.solution.diffusion(self.temperature)  # *self.particle_sherwood_number

    @property
    def particle_sherwood_number(self):
        Sc = self.solution.viscosity(self.temperature) / (
                self.solution.solvent.density(self.temperature) * self.solution.diffusion(self.temperature))
        Pe = self.peclet
        Re = Pe / Sc
        return 1 + 0.3 * np.sqrt(Re) * np.cbrt(Sc)

    @property
    def peclet(self):
        return -self.dmdt() / (
                self.solution.diffusion(self.temperature) * 4 * np.pi * self.radius * self.solution.solvent.density(
            self.temperature))

    @property
    def layers(self):
        return len(self.log_mass_particles)

    @staticmethod
    def from_mfp(solution: Suspension, environment, gravity,
                 radius, mass_fraction_particles, temperature, layers=10,
                 velocity=np.zeros(3), position=np.zeros(3), stationary=True):
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
        cell_boundaries = radius * np.arange(1,layers)/layers
        concentration = mass_particles / volume
        real_boundaries = np.concatenate(([0], cell_boundaries, [radius]))
        log_mass_solute = np.log(4/3*np.pi*concentration*(real_boundaries[1:]**3-real_boundaries[:-1]**3))
        cell_velocities = np.zeros(len(cell_boundaries))
        return SuspensionDroplet(solution, environment, gravity, stationary, temperature, velocity, position,
                                 mass_solvent,
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
        return np.array(c)

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
        return self.cell_boundaries - self.radius * np.arange(1,self.layers)/self.layers

    def boundary_acceleration(self):
        return (-self.cell_velocities * damping - stiffness * self.deviation / self.radius) / layer_inertia

    def boundary_correction(self):
        return self.cell_velocities

    @property
    def layer_volume(self):
        true_boundaries = np.concatenate(([0], self.cell_boundaries, [self.radius]))
        return 4/3*np.pi*(true_boundaries[1:]**3-true_boundaries[:-1]**3)

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

    @property
    def redistribute(self):
        sign = np.sign(self.cell_velocities)
        volume_corrections = 4 * np.pi * self.cell_boundaries ** 2 * self.cell_velocities
        volume_corrections = np.append(volume_corrections, [volume_corrections[-1] * self.layers / (self.layers - 1)])
        concentrations = self.linear_layer_concentrations()[1:]
        result = np.zeros(self.layers)
        fullness = (self.solution.rigidity *
                np.clip(concentrations[1:] - self.solution.critical_volume_fraction * self.solution.particle_density,a_min=0,a_max=None)**2 /
                    (self.solution.max_volume_fraction*self.solution.particle_density - self.solution.critical_volume_fraction * self.solution.particle_density)
        )
        for i in range(len(volume_corrections) - 1):
            if sign[i] < 0:
                value = volume_corrections[i] * (concentrations[i]) - volume_corrections[i + 1] * fullness[i]
                result[i] += value
                result[i + 1] -= value
            else:
                value = volume_corrections[i] * concentrations[i + 1] - volume_corrections[i + 1] * fullness[i]
                result[i] += value
                result[i + 1] -= value

        return result

    @property
    def corrected_crossing_rate(self):
        radius = self.radius
        R = self.cell_boundaries / radius
        viscosity_ratio = self.solution.viscosity(self.temperature) / self.environment.dynamic_viscosity
        return crossing_rate(R, radius) * (self.relative_speed / 0.02) * (1 + 1e-3 / 1.81e-5) / (
                1 + viscosity_ratio) * (self.layers / 100)

    @property
    def circulate(self):
        crossing_rates = self.corrected_crossing_rate
        result = np.zeros(self.layers)
        for index, (m0, m1, rate) in enumerate(
                zip(self.layer_mass_particles, self.layer_mass_particles[1:], crossing_rates)):
            value = self.layers * rate * (m0 - m1) / self.radius
            result[index] -= value
            result[index + 1] += value
        return result

    @property
    def layer_mass_particles(self):
        return np.exp(self.log_mass_particles)

    def get_gradients(self, normalised_boundaries):
        concentrations = self.linear_layer_concentrations()
        return (concentrations[2:] - concentrations[:-2]) / (normalised_boundaries[2:] - normalised_boundaries[:-2])

    @property
    def diffuse(self):
        radius = self.radius
        full_boundaries = np.concatenate(([0], self.cell_boundaries, [radius]))
        diffusion_coefficient = self.diffusion_coefficient
        normalised_boundaries = full_boundaries / radius
        gradients = self.get_gradients(normalised_boundaries)
        diffusion = np.zeros(self.layers)
        for i in range(self.layers - 1):
            value = 4 * np.pi * radius * diffusion_coefficient * gradients[i] * normalised_boundaries[i + 1] ** 2
            diffusion[i] += value
            diffusion[i + 1] -= value
        return diffusion

    def change_in_particles_mass(self):
        return (self.redistribute + self.diffuse) / self.layer_mass_particles

    def mass_particles(self):
        return np.sum(self.layer_mass_particles)

    def mass_solvent(self) -> float:
        return self.total_mass_solvent

    def surface_solvent_activity(self) -> float:
        return 1.0

    def virtual_droplet(self, x) -> Self:
        cell_boundaries, cell_velocities, layer_mass_particles, total_mass_solvent, temperature, velocity, position = self.split_state(
            x)
        return SuspensionDroplet(self.solution, self.environment, self.gravity, self.stationary, temperature, velocity,
                                 position,
                                 total_mass_solvent, cell_boundaries, cell_velocities, layer_mass_particles)

    def convert(self, mass_solvent):
        return UniformDroplet(aqueous_NaCl, self.environment, self.gravity, self.stationary,
                              self.environment.temperature, self.velocity,
                              self.position, mass_solvent, 0.0)

    def mass_solute(self) -> float:
        return 0.0

    def check_for_solidification(self):
        return self.solution.critical_volume_fraction - self.probe_concentration / self.solution.particle_density

    @property
    def probe_concentration(self):
        linear_concentrations = self.linear_layer_concentrations()
        radial_position = self.radius - self.solution.critical_shell_thickness * self.solution.particle_radius
        positions = np.concatenate(([0], self.cell_boundaries, [self.radius]))
        outer_index = np.max([np.argmax(positions >= radial_position), 1])
        m = (linear_concentrations[outer_index] - linear_concentrations[outer_index - 1]) / (
                positions[outer_index] - positions[outer_index - 1])
        return linear_concentrations[outer_index] + (radial_position - positions[outer_index]) * m

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

    def layer_volume_fraction(self):
        volumes = self.layer_mass_particles / self.solution.particle_density
        return volumes / self.layer_volume

    def equilibrium_droplet(self) -> UniformDroplet:
        """Given the current state of the droplet and environment, find and return the equilibrium droplet"""
        return self.convert(0.0)
