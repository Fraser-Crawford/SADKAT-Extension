from dataclasses import dataclass
from collections.abc import Callable

from fit import VapourBinaryDiffusionCoefficient


@dataclass
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