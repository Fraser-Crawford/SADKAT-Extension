import numpy as np
from attr import dataclass

from droplet import Droplet
from viscous_solution import ViscousSolution


@dataclass
class RadialDroplet(Droplet):
    """Class for describing a droplet with a non-uniform but radially symmetric composition"""
    solution: ViscousSolution
    equilibrium_solvent_mass: np.array
    mass_solvent: float # kg
    log_mass_solute: np.array # kg
    @staticmethod
    def from_mfs(solution, environment, gravity,
                 radius, mass_fraction_solute, temperature, layers=100,
                 velocity=np.zeros(3), position=np.zeros(3)):
        """Create a droplet from experimental conditions.

        Args:
            solution: parameters describing solvent+solute
            environment: parameters of gas surrounding droplet
            body acceleration in metres/second^2 (3-dimensional vector)
            radius in metres
            mass_fraction_solute (MFS) (unitless)
            temperature in K
            velocity in metres/second (3-dimensional vector)
            position in metres (3-dimensional vector)
        """
        delta_r = radius / layers
        radii = [(1 + i) * delta_r for i in range(layers)]
        volumes = np.array([4.0 / 3.0 * np.pi * (R ** 3 - r ** 3) for r, R in zip([0] + radii, radii)])
        droplet_volume = 4.0 / 3.0 * np.pi * radius ** 3
        mass = droplet_volume * solution.density(mass_fraction_solute)
        mass_solvent = (1 - mass_fraction_solute) * mass
        log_mass_solute = np.log(mass_fraction_solute * solution.density(mass_fraction_solute) * volumes)
        equilibrium_solvent_mass = (1 - mass_fraction_solute) * solution.density(mass_fraction_solute) * volumes
        return RadialDroplet(solution,environment,gravity,temperature,velocity,position,equilibrium_solvent_mass,mass_solvent,log_mass_solute)

    def set_state(self, state):
        try:
            self.mass_solvent, self.temperature, self.concentration, self.velocity, self.position = state
        except:
            x = state
            self.mass_solvent, self.log_mass_solute, self.temperature, self.velocity, self.position = x[0], x[
                                                                                                        1:self.initial_layers + 1], \
        x[1 + self.initial_layers], x[2 + self.initial_layers:5 + self.initial_layers], x[5 + self.initial_layers:]

    @property
    def layer_mass_solute(self):
        return np.exp(self.log_mass_solute)

    def mass_solute(self) -> float:
        return np.sum(self.layer_mass_solute)

    def mass_solvent(self) -> float:
        return self.mass_solvent

    def volume(self) -> float:
        return np.sum(self.layer_volumes)

    def surface_solvent_activity(self) -> float:
        return self.solution.solvent_activity_from_mass_fraction_solute(self.layer_mass_fraction_solute[self.outer_layer_index])

    @property
    def outer_layer_solvent_mass(self):
        interior_solvent = np.sum(self.equilibrium_solvent_mass[:self.outer_layer_index])
        return self.mass_solvent - interior_solvent

    @property
    def layer_radii(self):
        cumulative_layer_volumes = np.cumsum(self.layer_volumes)
        return np.cbrt(3 * cumulative_layer_volumes / (4 * np.pi))

    @property
    def concentration_gradients(self):
        concentrations = self.concentrations
        delta_rs = self.layer_widths
        return [2.0 * (c1 - c0) / (dr1 + dr2) for c0, c1, dr1, dr2 in
                zip(concentrations, concentrations[1:], delta_rs, delta_rs[1:])]

    @property
    def layer_widths(self):
        radii = self.layer_radii
        result = [self.layer_radii[0]]
        for r0, r1 in zip(radii, radii[1:]):
            result.append(r1 - r0)
        return result

    def correct_derivative(self, dmdts):
        return np.array(
            [0 if mass_solute <= 0 else dmdt / mass_solute for mass_solute, dmdt in zip(self.layer_mass_solute, dmdts)])

    @property
    def dCdt(self):
        diffusion = self.diffusion_coefficients
        boundries = self.layer_radii[:-1]
        d_plus = np.array([(d1 + d0) / 2.0 for d0, d1 in zip(diffusion, diffusion[1:])])
        values = d_plus * self.concentration_gradients * boundries ** 2
        result = np.zeros(self.initial_layers)
        for i in range(self.outer_layer_index):
            result[i] += values[i]
            if i != self.outer_layer_index:
                result[i + 1] -= values[i]
            result *= 4.0 * np.pi
        return self.correct_derivative(result)

    @property
    def outer_layer_index(self):
        accumulator = 0
        for index, solvent_mass in enumerate(self.equilibrium_solvent_mass):
            accumulator += solvent_mass
            if self.mass_solvent <= accumulator + solvent_mass * 1e-5:
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
    def diffusion_coefficients(self)->np.array:
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
        return np.hstack((self.dmdt,self.dCdt, self.dTdt ,self.dvdt, self.drdt))

    def dmdt(self):
        Sh = self.sherwood_number

        D_function = self.solution.solvent.vapour_binary_diffusion_coefficient
        lam = D_function.lam

        T_inf = self.environment.temperature
        T = self.temperature
        D_inf = D_function(T_inf)

        # Apply temperature correction to diffusion coefficient appearing in mass flux.
        eps = 1e-8
        if np.abs(T_inf - T) < eps:
            C = 1  # ensure numerical stability as T -> T_inf
        else:
            C = (T_inf - T) / T_inf ** (lam - 1) * (2 - lam) / (T_inf ** (2 - lam) - T ** (2 - lam))
        D_eff = C * D_inf

        I = np.log((self.environment.pressure - self.vapour_pressure) /
                   (self.environment.pressure - self.environment.vapour_pressure))

        beta = self.fuchs_sutugin_correction

        return 4 * np.pi * self.radius * self.environment.density * (
                    self.solution.solvent.molar_mass / self.environment.molar_mass) * D_eff * Sh * beta * I

    @property
    def dTdt(self):
        """Time derivative of temperature from heat flux at the surface in K/s."""
        Nu = self.nusselt_number
        Gamma = 5.67e-8

        r = self.radius
        m = self.mass
        rho = self.density

        T = self.temperature
        T_inf = self.environment.temperature
        K = self.environment.thermal_conductivity

        L = self.solution.solvent.specific_latent_heat_vaporisation(T)
        c = self.solution.solvent.specific_heat_capacity(T)
        r = self.radius

        return 3 * K * (T_inf - T) * Nu / (c * rho * r ** 2) + L * self.dmdt() / (c * m) - 3 * Gamma * (
                    T ** 4 - T_inf ** 4) / (c * rho * r)

    def virtual_droplet(self, x):
        x = (x[0], x[1:1 + self.initial_layers], x[1 + self.initial_layers],
             x[2 + self.initial_layers:5 + self.initial_layers], x[5 + self.initial_layers:])
        return RadialDroplet(self.solution, self.environment, self.gravity, self.equilibrium_solvent_mass, *x)

    def state(self) -> np.array:
        return np.hstack((self.mass_solvent, self.log_mass_solute, self.temperature, self.velocity, self.position))