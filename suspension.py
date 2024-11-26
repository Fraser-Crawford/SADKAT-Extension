from dataclasses import dataclass
from typing import Callable

from solvent import Solvent
from water import water
import numpy as np
from scipy.constants import k

@dataclass
class Suspension:
    solvent: Solvent
    specific_heat_capacity: Callable[[float,float], float] #temp, mass fraction of particles -> heat cap.
    critical_volume_fraction: float
    viscosity: Callable[[float], float] #temp -> viscosity
    particle_radius: float
    particle_density: float

    def diffusion(self, temperature:float):
        return k*temperature/(6*np.pi*self.viscosity(temperature)*self.particle_radius)

    @property
    def particle_mass(self):
        return 4/3*self.particle_radius**3*np.pi*self.particle_density

def test_suspension(radius):
    heat_cap = lambda T, mfp: water.specific_heat_capacity(T)*(1-mfp)+mfp*4182
    viscosity = lambda T : -1.748e-5*(T-273)+1.336e-3
    return Suspension(water,heat_cap,np.pi/6,viscosity,radius,1000)

def silica(radius):
    heat_cap = lambda T, mfp: water.specific_heat_capacity(T)*(1-mfp)+mfp*703 #wikipedia
    viscosity = lambda T: -1.748e-5 * (T - 273) + 1.336e-3
    return Suspension(water, heat_cap, np.pi / 6, viscosity, radius, 2200) #From wicker chem source

def dummy_density(radius):
    heat_cap = lambda T, mfp: water.specific_heat_capacity(T)
    viscosity = lambda T: -1.748e-5 * (T - 273) + 1.336e-3
    return Suspension(water, heat_cap, np.pi / 6, viscosity, radius, 2000)