from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing_extensions import Self

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import numpy.typing as npt
from environment import Environment
from fit import kelvin_effect
from solution import Solution
from viscous_solution import ViscousSolution


@dataclass
class Droplet(ABC):
    """Abstract class completely describes the state of the droplet during its evolution.
    """

    solution: Solution | ViscousSolution
    environment: Environment
    gravity: np.array  # m/s^2
    temperature: float  # K
    velocity: np.array
    position: np.array

    @abstractmethod
    def state(self) -> npt.NDArray[np.float_]:
        """Returns the state of the droplet."""
        pass

    @abstractmethod
    def set_state(self,state:npt.NDArray[np.float_]):
        """Sets the state of the droplet."""
        pass

    @abstractmethod
    def dxdt(self)->npt.NDArray[np.float_]:
        """The time derivative of the state of the droplet."""
        pass

    def dmdt(self):
        """Time derivative of mass, i.e. the rate of evaporation in kg/s."""
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

    def dTdt(self):
        """Time derivative of temperature from heat flux at the surface in K/s."""
        Nu = self.nusselt_number
        Gamma = 5.67e-8

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

    @abstractmethod
    def mass_solute(self) -> float:
        """Returns the mass of solute in the droplet in kg."""
        pass

    @abstractmethod
    def mass_solvent(self)->float:
        """Returns the mass of solvent in the droplet in kg."""
        pass

    @property
    def refractive_index(self)->float:
        """Returns the refractive index of the droplet based on a mass fraction/density correction."""
        solutionD = self.density
        soluteD = self.solution.solid_density
        soluteRI = self.solution.solid_refractive_index
        solventD = self.solution.solvent.density(self.temperature)
        solventRI = self.solution.solvent.refractive_index
        mfs = self.mass_fraction_solute
        return np.sqrt((1 + 2 * solutionD * (((soluteRI ** 2 - 1) * mfs) / ((soluteRI ** 2 + 2) * soluteD) + (
                    (1 - mfs) * (solventRI ** 2 - 1)) / (solventD * (solventRI ** 2 + 2)))) / (1 - solutionD * (
                    ((soluteRI ** 2 - 1) * mfs) / ((soluteRI ** 2 + 2) * soluteD) + (
                        (1 - mfs) * (solventRI ** 2 - 1)) / (solventD * (solventRI ** 2 + 2)))))

    def copy(self):
        """Create an identical copy of this droplet."""
        return self.virtual_droplet(self.state().copy())

    @property
    def complete_state(self):
        """All droplet variables, including both independent and dependent variables that completely
        determine all droplet properties.

        This form is ready for a row within a table specifying e.g. a droplet's trajectory.
        """
        return dict(mass=self.mass,
                    mass_solute=self.mass_solute(),
                    mass_solvent=self.mass_solvent(),
                    mass_fraction_solute=self.mass_fraction_solute,
                    mass_fraction_solvent=self.mass_fraction_solvent,
                    mole_fraction_solute=self.mole_fraction_solute,
                    mole_fraction_solvent=self.mole_fraction_solvent,
                    density=self.density,
                    radius=self.radius,
                    refractive_index=self.refractive_index,
                    vapour_pressure=self.vapour_pressure,
                    temperature=self.temperature,
                    drag_coefficient=self.drag_coefficient,
                    reynolds_number=self.reynolds_number,
                    schmidt_number=self.schmidt_number,
                    prandtl_number=self.prandtl_number,
                    sherwood_number=self.sherwood_number,
                    nusselt_number=self.nusselt_number,
                    vx=self.velocity[0],
                    vy=self.velocity[1],
                    vz=self.velocity[2],
                    speed=self.speed,
                    x=self.position[0],
                    y=self.position[1],
                    z=self.position[2],
                    gx=self.gravity[0],
                    gy=self.gravity[1],
                    gz=self.gravity[2],
                    )

    @property
    def mass(self):
        """Total mass of droplet (both solvent and solute components) in kg."""
        return self.mass_solute() + self.mass_solvent()

    @property
    def mass_fraction_solute(self):
        """Mass fraction of solute (i.e. the non-volatile component).
        NB: Should be zero for pure solvent."""
        return self.mass_solute() / self.mass

    @property
    def mass_fraction_solvent(self):
        """Mass fraction of solvent."""
        return 1 - self.mass_fraction_solute

    @property
    def mole_fraction_solute(self):
        """Mole fraction of solute (i.e. the non-volatile component).
        NB: Should be zero for pure solvent."""
        return self.solution.mole_fraction_solute(self.mass_fraction_solute)

    @property
    def mole_fraction_solvent(self):
        """Mole fraction of solvent."""
        return 1 - self.mole_fraction_solute

    @property
    def density(self):
        """Droplet density in kg/m^3."""
        return self.solution.density(self.mass_fraction_solute)

    @abstractmethod
    def volume(self)->float:
        """Returns the volume of the droplet in m3."""
        pass

    @property
    def radius(self):
        """Droplet radius in metres."""
        return (self.volume() / (4 * np.pi / 3)) ** (1 / 3)

    @property
    def diameter(self):
        """Droplet diameter in metres."""
        return 2 * self.radius

    @property
    def vapour_pressure(self):
        """Vapour pressure at gas-liquid boundary in Pascals."""
        P = self.solution.solvent_activity_from_mass_fraction_solute(
            self.mass_fraction_solute) * self.solution.solvent.equilibrium_vapour_pressure(self.temperature)
        P *= kelvin_effect(self.solution.solvent.surface_tension(self.temperature),
                           self.solution.solvent.density(self.temperature),
                           self.solution.solvent.molar_mass,
                           self.temperature,
                           self.radius)
        return P

    @abstractmethod
    def surface_solvent_activity(self)->float:
        """Returns the solvent activity at the surface of the droplet"""
        pass

    @property
    def speed(self):
        """Magnitude of velocity vector in metres/second."""
        return np.linalg.norm(self.velocity)

    @property
    def relative_velocity(self):
        """Velocity relative to environment in metres/second."""
        return self.velocity - self.environment.velocity
        # return self.velocity - self.jet.velocity

    @property
    def relative_speed(self):
        """Magnitude of relative velocity vector in metres/second."""
        return np.linalg.norm(self.relative_velocity)

    @property
    def drag_coefficient(self):
        """Non-dimensional number describing strength of drag forces."""
        Re = self.reynolds_number
        if Re > 1000:
            return 0.424
        elif Re < 1e-12:
            return np.inf
        else:
            return (24 / Re) * (1 + Re ** (2 / 3) / 6)

    @property
    def reynolds_number(self):
        """Non-dimensional number describing the type of fluid flow."""
        return self.environment.density * self.diameter * self.speed / self.environment.dynamic_viscosity

    @property
    def schmidt_number(self):
        """Non-dimensional number describing the ratio of momentum diffusivity to mass diffusivity."""
        D_function = self.solution.solvent.vapour_binary_diffusion_coefficient
        T_inf = self.environment.temperature
        D_inf = D_function(T_inf)
        return self.environment.dynamic_viscosity / (self.environment.density * D_inf)

    @property
    def prandtl_number(self):
        """Non-dimensional number describing the ratio of momentum diffusivity to thermal diffusivity."""
        return self.environment.specific_heat_capacity * self.environment.dynamic_viscosity / self.environment.thermal_conductivity

    @property
    def sherwood_number(self):
        """Non-dimensional number describing mass transfer."""
        Re = self.reynolds_number
        Sc = self.schmidt_number
        return 1 + 0.3 * Re ** (1 / 2) * Sc ** (1 / 3)

    @property
    def nusselt_number(self):
        """Non-dimensional number describing conductive heat transfer."""
        Re = self.reynolds_number
        Pr = self.prandtl_number
        return 1 + 0.3 * Re ** (1 / 2) * Pr ** (1 / 3)

    @property
    def knudsen_number(self):
        """Non dimensional number describing size regime"""
        return self.environment.mean_free_path / self.radius

    @property
    def fuchs_sutugin_correction(self):
        Kn = self.knudsen_number
        alpha = 1  # parameter in Fuchs-Sutugin theory that we can take to be one (for now).
        return (1 + Kn) / (1 + (4 / 3 * (1 + Kn) / alpha + 0.377) * Kn)

    @property
    def cunningham_slip_correction(self):
        """Correction to Stokes' law for drag for small particles due to the onset
        of slip on the particle surface."""

        # Phenomenological parameters in the theory due to Davies (1945):
        A1 = 1.257
        A2 = 0.400
        A3 = 1.100

        Kn = self.knudsen_number
        return 1 + Kn * (A1 + A2 * np.exp(-A3 / Kn))

    def dvdt(self):
        """Time derivative of velocity, i.e. its acceleration from Newton's second law in m/s^2."""
        rho_p = self.density
        rho_g = self.environment.density
        g = self.gravity

        buoyancy = 1 - rho_g / rho_p
        acceleration = buoyancy * g

        C = self.drag_coefficient / self.cunningham_slip_correction
        if np.isfinite(C):
            acceleration -= 3 * C * rho_g * self.relative_speed * self.relative_velocity / (8 * rho_p * self.radius)

        return acceleration

    def drdt(self):
        """Time derivative of droplet position, i.e. its velocity in m/s."""
        return self.velocity

    @abstractmethod
    def virtual_droplet(self, x)->Self:
        """Returns a new droplet from the state x given"""
        pass

    @abstractmethod
    def solver(self, dxdt, time_range, first_step, rtol, events):
        pass

    def integrate(self, t, rtol=1e-8,
                  terminate_on_equilibration=False, equ_threshold=1e-4,
                  terminate_on_efflorescence=False, eff_threshold=0.5,
                  first_step=1e-12):
        """Integrate the droplet state forward in time.

        This solves an initial value problem with the current state as the initial conditions.
        The droplet state is updated to the final state after the integration.

        Args:
            t: total time to integrate over (s).
            rtol: relative tolerance used to set dynamic integration timestep between frames in
                trajectory (s). Smaller tolerance means a more accurate integration.
                NB: if numerical artifacts occur in the resulting trajectory, that suggests this
                parameter needs to be decreased.
            terminate_on_equilibration (default=False): if True, then the integration will stop if
                the evaporation rate falls below eps * the initial mass
            equ_threshold: threshold to use for the equilibration termination criterion.
            terminate_on_efflorescence (default=False): if True, then the integration will stop if
                the solvent activity falls below a threshold.
            eff_threshold: threshold to use for the efflorescence termination criterion.
            first_step: size of initial integration step. The subsequent timesteps are determined
                dynamically based on the rate of change and the error tolerance parameter rtol.
        Returns:
            Trajectory of historical droplets showing how it reaches the new state.
        """

        events = []
        if terminate_on_equilibration:
            m0 = self.mass
            equilibrated = lambda time, x: np.abs(self.virtual_droplet(x).dmdt()) - equ_threshold * m0
            equilibrated.terminal = True
            events += [equilibrated]

        if terminate_on_efflorescence:
            efflorescing = lambda time, x: self.virtual_droplet(x).surface_solvent_activity() - eff_threshold
            efflorescing.terminal = True
            events += [efflorescing]

        dxdt = lambda time, x: self.virtual_droplet(x).dxdt()

        trajectory = self.solver(dxdt,(0,t), first_step, rtol, events)

        self.set_state(trajectory.y[:, -1])
        return trajectory

    def complete_trajectory(self, trajectory):
        """Get the trajectory of all variables (including dependent ones) from a simulation (i.e.
        the output of UniformDroplet.integrate).

        Args:
            trajectory: the output of UniformDroplet.integrate, which gives the trajectory of independent
                        variables only.
        Returns:
            A pandas dataframe detailing the complete droplet history.
        """

        variables = self.complete_state
        for label in variables:
            variables[label] = np.empty(trajectory.t.size,dtype=object)

        for i, state in enumerate(trajectory.y.T):
            earlier_droplet = self.virtual_droplet(state)
            earlier_state = earlier_droplet.complete_state
            for label, value in earlier_state.items():
                variables[label][i] = value

        variables['time'] = trajectory.t

        return pd.DataFrame(variables)