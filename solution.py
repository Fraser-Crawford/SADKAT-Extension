from dataclasses import dataclass
from collections.abc import Callable
import numpy as np
import numpy.typing as npt
from solvent import Solvent
from numba.experimental import jitclass
from numba import float64, boolean, int32, optional
import numba

spec = [
    ("molar_mass_solute",float64),
    ("num_ions",int32),
    ("solubility_limit",float64),
    ("solid_density",float64),
    ("density",float64[:](float64[:]).as_type()),
    ("activity",float64[:](float64[:]).as_type()),
    ("concentration_coefficients",float64[:]),
    ("density_derivative_coefficients",float64[:])
]

@jitclass(spec)
class Solution:
    """Class to conveniently store all parameters needed to describe a solute together."""

    solvent: Solvent
    molar_mass_solute: float  # g/mol
    num_ions: int  # unitless
    solubility_limit: float  # kg per kg solvent
    solid_density: float  # kg/m^3
    density: Callable[[float|npt.NDArray[np.float_]],float|npt.NDArray[np.float_]]  # kg/m^3
      # a fitting function for solvent activity (unitless) in
    # terms of the mass fraction of solute, or None (the
    # default) to assume Raoult's law for ideal mixtures
    activity: Callable[[npt.NDArray[np.float_]],npt.NDArray[np.float_]]
    solid_refractive_index: float
    def __init__(self,solvent,molar_mass_solute,num_ions,solubility_limit,solid_density,density,activity,solid_refractive_index):
        mfs = np.linspace(0, 1.0, 100)
        densities = self.density(mfs)
        concentrations = mfs * densities
        self.concentration_coefficients = np.polyfit(concentrations, mfs, 4)
        density_gradients = np.gradient(densities,mfs)
        self.density_derivative_coefficients = np.polyfit(mfs, density_gradients, 4)
        self.solvent = solvent
        self.molar_mass_solute = molar_mass_solute
        self.num_ions = num_ions
        self.solubility_limit = solubility_limit
        self.solid_density = solid_density
        self.density = density
        self.activity = activity
        self.solid_refractive_index = solid_refractive_index

    def mole_fraction_solute(self, mass_fraction_solute):
        """Mole fraction from mass fraction."""
        w_solute = mass_fraction_solute / self.molar_mass_solute
        w_solvent = (1 - mass_fraction_solute) / self.solvent.molar_mass
        return w_solute / (w_solute + w_solvent)

    def mole_fraction_solvent(self, mass_fraction_solute):
        """Mole fraction of *solvent* from mass fraction of *solute*."""
        return 1 - self.mole_fraction_solute(mass_fraction_solute)

    def concentration_to_solute_mass_fraction(self, concentration):
        return np.poly1d(self.concentration_coefficients)(concentration)

    def concentration(self, mfs):
        return mfs * self.density(mfs)

    def density_derivative(self, mfs):
        return np.poly1d(self.density_derivative_coefficients)(mfs)

    def mass_fraction_from_activity(self, activity):
        mfs = np.linspace(1.0, 0.0, 100)
        mfs_activity = self.activity(mfs)
        return np.interp(activity, mfs_activity, mfs)

    def refractive_index(self, mfs, temperature) -> float:
        """Returns the refractive index of the droplet based on a mass fraction/density correction."""
        solutionD = self.density(mfs)
        soluteD = self.solid_density
        soluteRI = self.solid_refractive_index
        solventD = self.solvent.density(temperature)
        solventRI = self.solvent.refractive_index
        return np.sqrt((1 + 2 * solutionD * (((soluteRI ** 2 - 1) * mfs) / ((soluteRI ** 2 + 2) * soluteD) + (
                (1 - mfs) * (solventRI ** 2 - 1)) / (solventD * (solventRI ** 2 + 2)))) / (1 - solutionD * (
                ((soluteRI ** 2 - 1) * mfs) / ((soluteRI ** 2 + 2) * soluteD) + (
                (1 - mfs) * (solventRI ** 2 - 1)) / (solventD * (solventRI ** 2 + 2)))))