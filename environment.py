from dataclasses import dataclass, field

import chemicals
import numpy as np
from fluids.constants import gas_constant

from solvent import Solvent
from water import water


@dataclass
class Environment:
    """Class to conveniently store all parameters needed to describe the surrounding gas together."""

    solvent: Solvent
    molar_mass: float               # g/mol
    pressure: float                 # Pa
    temperature: float              # K
    relative_humidity: float        # unitless, bounded in [0,1]
    specific_heat_capacity: float   # J/kg/K
    thermal_conductivity: float     # J/s/m/K
    dynamic_viscosity: float        # kg/m/s
    velocity: np.array=field(default_factory=lambda: np.zeros(3))  # m/s

    @property
    def density(self):
        """Density of gas assuming ideal gas law in kg/m^3."""
        return (1e-3*self.molar_mass) * self.pressure / (gas_constant * self.temperature) # kg/m^3

    @property
    def vapour_pressure(self):
        """Vapour pressure in Pascals."""
        return self.relative_humidity * self.solvent.equilibrium_vapour_pressure(self.temperature)

    @property
    def mean_free_path(self):
        """Calculate mean free path via hard sphere approximation."""
        return self.dynamic_viscosity / self.density * np.sqrt(np.pi * 1e-3*self.molar_mass / (2*gas_constant * self.temperature))

molar_mass_air = chemicals.air.lemmon2000_air_MW # g/mol
molar_density_air = lambda T: chemicals.air.lemmon2000_rho(T, 101325) # mol / m^3
density_air = lambda T: 1e-3*molar_mass_air * molar_density_air(T) # kg/m^3
thermal_conductivity_air = np.vectorize(lambda T: chemicals.thermal_conductivity.k_air_lemmon(T, molar_density_air(T))) #  J/s/m/K
molar_mass_dry_air = 28.9647
specific_heat_capacity_air = lambda T: 1.006e3 # J/kg/K
dynamic_viscosity_air = lambda T: chemicals.viscosity.mu_air_lemmon(T, molar_density_air(T)) # kg/m/s

def Atmosphere(temperature,
               relative_humidity=0,
               pressure=101325,
               velocity=np.zeros(3),
               solvent=water):
    """Set up conditions for Earth's atmosphere.

    Args:
        temperature: room temperature (K)
        relative_humidity: RH of water vapour in decimal (default=0 for dry air).
        pressure: room pressure (default=101325 Pa) (Pa)
        velocity: velocity in (m/s, dimensional vector)
        solvent: solvent used in the droplet solution
    Returns:
        Environment object describing room conditions.
    """
    vapour_pressure_water = relative_humidity * water.equilibrium_vapour_pressure(temperature)
    mole_fraction_water = vapour_pressure_water / pressure
    molar_mass = (1-mole_fraction_water) * molar_mass_dry_air + mole_fraction_water * water.molar_mass

    return Environment(solvent, molar_mass, pressure, temperature, relative_humidity,
                       specific_heat_capacity_air(temperature),
                       thermal_conductivity_air(temperature),
                       dynamic_viscosity_air(temperature),
                       velocity)