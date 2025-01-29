from collections.abc import Callable
from dataclasses import dataclass
import numpy as np
from fluids.constants import gas_constant
import scipy
from numba.experimental import jitclass
from numba import float64, boolean, int32, optional, jit
import numba

spec = [
    ("D_ref",float64),
    ("T_ref",float64),
    ("lam",float64),
]


@jitclass(spec)
class VapourBinaryDiffusionCoefficient:
    """Standard fitting function to describe the temperature dependence
    of the binary diffusion coeffient for evaporated solvent in air.

    The functional form of this is taken from Xie et al., Indoor Air (2007).
    """
    D_ref: float  # m^2/s
    T_ref: float  # K
    lam: float  # unitless, scaling parameter
    def __init__(self, D_ref: float, T_ref: float, lam: float):
        self.D_ref = D_ref
        self.T_ref = T_ref
        self.lam = lam
    def get(self,T):
        """Fit itself.

                Args:
                    T: temperature in Kelvin.
                Returns:
                    The diffusion coefficient in m^2/s.
                """
        return self.D_ref * (T / self.T_ref) ** self.lam

@jit
def surface_tension(A, B, C, D, E, T_crit, T):
    """Surface tension fitting function (as function of temperature).

    Source:
    http://ddbonline.ddbst.de/DIPPR106SFTCalculation/DIPPR106SFTCalculationCGI.exe

    Args:
        A, B, C, D, E: parameters of fit.
        T_crit: temperature of critical point in Kelvin.
        T: temperature(s) to evaluate surface tension at (Kelvin).
    Returns:
        Surface tension in N/m.
    """

    T_r = T / T_crit
    power = B + (C * T_r) + (D * T_r ** 2) + (E * T_r ** 3)
    sigma = (A * (1 - T_r) ** power) / 1000  # convert from mN/m to N/m

    return sigma


def kelvin_effect(solvent_surface_tension, solvent_density, solvent_molar_mass, T, droplet_radius):
    """Drop in vapour pressure due to surface curvature.

    Read more: https://www.e-education.psu.edu/meteo300/node/676

    Args:
        solvent_surface_tension: surface tension of droplet/medium interface (N/m).
        solvent_density: density of droplet (kg/m^3).
        solvent_molar_mass: molar mass of droplet (g/mol).
        T: temperature of droplet (K)
        droplet_radius: radius of droplet (m).
    Returns:
        Multiplier to vapour pressure in the curved interface (unitless).
        Final vapour pressure is obtained by multiplying this by the flat result.
    """

    molarity = 1e3 * solvent_density / solvent_molar_mass  # mol / m^3
    return np.exp((2 * solvent_surface_tension) / (molarity * gas_constant * T * droplet_radius))

def rackett_equation(A, B, C, D, T):
    """Rackett equation used to parameterise density (vs temperature) for various solvents.

    Source:
    http://ddbonline.ddbst.de/DIPPR105DensityCalculation/DIPPR105CalculationCGI.exe

    Args:
        A, B, C, D: fit parameters.
        T: temperature in Kelvin.
    Returns:
        Density in kg/m^3
    """
    return A / (B ** (1 + ((1 - (T / C))) ** D))


def antoine_equation(T, A, B, C):
    """Semi-empirical fit function describing relation between vapour pressure and temperature.

    Args:
        T: temperature in Kelvin.
        A, B, C: fit parameters.
    Returns:
        Vapour pressure in Pa.
    """
    return 10 ** (A - (B / (T + C)))


def enthalpy_vapourisation(A, alpha, beta, T_c, T):
    """Fit function used by NIST for enthalpy of vaporisation H_vap (at saturation pressure).

    Fit function is that described in Majer and Svoboda (1985), referenced to by NIST here:
    https://webbook.nist.gov/cgi/cbook.cgi?ID=C64175&Mask=4

    Args:
        A, alpha, beta: fit parameters of function.
        T_c: critical temperature (Kelvin).
        T: temperature(s) to evaluate H_vap at (Kelvin).
    Returns:
        Enthalpy of vapourisation H_vap in J/mol.
    """
    T_r = T / T_c
    return 1000 * (A * np.exp(- alpha * T_r) * (1 - T_r) ** beta)

class IdealSolventActivity:
    """In ideal mixtures the solvent activity obeys Raoult's law."""

    def __init__(self, solution):
        """Construct the activity rule for this mixture.

        Args:
            solution: parameters describing the solution. We need this to calculate the mole fractions
            in the mixture to apply Raoult's law."""
        self.solution = solution

    def solvent_activity_from_mass_fraction_solute(self, mfs: float) -> float:
        """Solvent activity is simply the mole fraction of solvent under Raoult's law.

        Args:
            mfs: the mass fraction of the solute, a number bounded between 0 and 1
        """
        return self.solution.mole_fraction_solvent(mfs)

    def mass_fraction_solute_from_solvent_activity(self, aw:float ) -> float:
        """Invert Raoult's law to obtain the mass fraction at a particular solvent activity.

        This is useful for e.g. obtaining the equilibrium mass fraction when the water activity matches
        the ambient conditions.

        Args:
            aw: the solvent activity, a number bounded between 0 and 1.
        """
        x_solvent = aw
        x_solute = 1 - aw
        w_solvent = x_solvent * self.solution.solvent.molar_mass
        w_solute = x_solute * self.solution.molar_mass_solute
        return w_solute / (w_solute + w_solvent)

def invert_fit(y, yfit):
    """Invert a fit y(x) to yield x(y).

    Args:
        y: y value (a scalar).
        yfit: parameters of polynomial fit y(x).
    Returns:
        The value x(y).
    """
    x = (yfit - y).roots
    x = np.real(np.min(x[(np.isreal(x)) & (x >= 0)]))
    return x

def ActivityVsMfsParametrisation(coefficients):
    return lambda mfs: np.poly1d(np.flipud(coefficients))(mfs)

def DensityVsMassFractionFit(coefficients)->Callable[[float],float]:
    return lambda mfs: np.poly1d(np.flipud(coefficients))(np.sqrt(mfs))

def enrichment_factor(Pe):
    return lambda R: R**2*np.exp(Pe/2.0*R**2)

def beta(Pe):
    return scipy.integrate.quad(enrichment_factor(Pe), 0,1)[0]

median_observation_angle = np.radians(45)

def correction_factor(RI):
    return 1/(np.cos(median_observation_angle / 2.0) + RI * np.sin(median_observation_angle / 2.0) / (
        np.sqrt(1.0 + RI ** 2 - 2.0 * RI * np.cos(median_observation_angle / 2.0))))

def correct_radius(radius, previous_RI, new_RI):
    return radius / correction_factor(previous_RI) * correction_factor(new_RI)