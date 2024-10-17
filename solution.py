from dataclasses import dataclass
from collections.abc import Callable
import numpy as np
from solvent import Solvent


@dataclass
class Solution:
    """Class to conveniently store all parameters needed to describe a solute together."""

    solvent: Solvent
    molar_mass_solute: float  # g/mol
    num_ions: int  # unitless
    solubility_limit: float  # kg per kg solvent
    solid_density: float  # kg/m^3
    density: Callable[[float|np.array],float|np.array]  # kg/m^3
      # a fitting function for solvent activity (unitless) in
    # terms of the mass fraction of solute, or None (the
    # default) to assume Raoult's law for ideal mixtures
    solvent_activity_from_mass_fraction_solute: Callable[[float|np.array], float|np.array]
    solid_refractive_index: float

    def mole_fraction_solute(self, mass_fraction_solute):
        """Mole fraction from mass fraction."""
        w_solute = mass_fraction_solute / self.molar_mass_solute
        w_solvent = (1 - mass_fraction_solute) / self.solvent.molar_mass
        return w_solute / (w_solute + w_solvent)

    def mole_fraction_solvent(self, mass_fraction_solute):
        """Mole fraction of *solvent* from mass fraction of *solute*."""
        return 1 - self.mole_fraction_solute(mass_fraction_solute)

    def concentration_to_solute_mass_fraction(self, concentration):
        mfss = np.linspace(0, 1.0, 101)
        return np.interp(concentration, [self.density(mfs)*mfs for mfs in mfss], mfss)

    def concentration(self, mfs):
        return mfs * self.density(mfs)