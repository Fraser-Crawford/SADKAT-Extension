from dataclasses import dataclass
from collections.abc import Callable

from fit import VapourBinaryDiffusionCoefficient
from numba.experimental import jitclass
from numba import float64, boolean, int32, optional
import numba

spec = [
    ("molar_mass",float64),
    ("density",float64(float64).as_type()),
    ("specific_heat_capacity",float64(float64).as_type()),
    ("specific_latent_heat_vaporisation",float64(float64).as_type()),
    ("equilibrium_vapour_pressure",float64(float64).as_type()),
    ("surface_tension",float64(float64).as_type()),
    ("refractive_index",float64)
]

@jitclass(spec)
class Solvent:
    """Class to conveniently store all parameters needed to describe a solvent together."""
    molar_mass: float  # g/mol
    density: Callable[[float],float]  # kg/m^3
    specific_heat_capacity: Callable[[float],float] # J/kg/K
    specific_latent_heat_vaporisation: Callable[[float],float]  # J/kg
    equilibrium_vapour_pressure: Callable[[float],float]  # Pa
    vapour_binary_diffusion_coefficient: VapourBinaryDiffusionCoefficient # m^2/s
    surface_tension: Callable[[float],float]  # N/m
    refractive_index: float #real component
    def __init__(self, molar_mass, density, specific_heat_capacity,specific_latent_heat_vaporisation,equilibrium_vapour_pressure,vapour_binary_diffusion_coefficient,surface_tension,refractive_index):
        self.molar_mass = molar_mass
        self.density = density
        self.specific_heat_capacity = specific_heat_capacity
        self.specific_latent_heat_vaporisation = specific_latent_heat_vaporisation
        self.equilibrium_vapour_pressure = equilibrium_vapour_pressure
        self.vapour_binary_diffusion_coefficient = vapour_binary_diffusion_coefficient
        self.surface_tension = surface_tension
        self.refractive_index = refractive_index