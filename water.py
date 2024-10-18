from solvent import Solvent
import chemicals
import numpy as np
from fit import surface_tension,VapourBinaryDiffusionCoefficient

molar_mass_water = 2 * chemicals.periodic_table.H.MW + chemicals.periodic_table.O.MW  # g/mol
def density_water(temperature):
    """Fit for the density of pure water used in J Walker model.

    Originally from:

        Wagner and Pruss Addendum to J. Phys. Chem. Ref. Data 16, 893 (1987),
        J. Phys. Chem. Ref. Data, 1993, 22, 783–787

    Args:
        temperature: temperature in Kelvin.
    Returns:
        The water density in kg/m^3.
    """
    ref_T = 647.096  # K
    ref_density = 322  # kg/m^3
    b1, b2, b3, b4, b5, b6 = 1.99274064, 1.09965342, -0.510839303, -1.75493479, -45.5170352, -674694.45

    theta = temperature / ref_T
    tau = 1 - theta
    density = ref_density * (1 + b1 * pow(tau, 1 / 3) + b2 * pow(tau, 2 / 3) + b3 * pow(tau, 5 / 3) + b4 * pow(tau,
                                                                                                               16 / 3) + b5 * pow(
        tau, 45 / 3) + b6 * pow(tau, 110 / 3))

    return density


# IAPWS-95 https://chemicals.readthedocs.io/chemicals.iapws.html
specific_heat_capacity_water = np.vectorize(
    lambda T: chemicals.iapws.iapws95_properties(T, 101325)[5])  # J/kg/K

# Su, PCCP (2018)
specific_latent_heat_water = lambda T: 3.14566e6 - 2361.64 * T  # J/kg


def equilibrium_vapour_pressure_water(T):
    """Using the Buck equation (from integrating the Clausius-Clapeyron equation).

    Args:
        T: temperature in Kelvin.
    Returns:
        The vapour pressure in Pascals.
    """
    T_C = T - 273.15  # Celsius
    return 1e3 * 0.61161 * np.exp((18.678 - (T_C / 234.5)) * (T_C / (257.14 + T_C)))  # Pa


def surface_tension_water(T):
    """Surface tension of water as a function of temperature.

    Parameterisation of water surface tension from:
    http://ddbonline.ddbst.de/DIPPR106SFTCalculation/DIPPR106SFTCalculationCGI.exe

    Parameters of fit from source:
        Tc           Tmin    Tmax
        647.3        233     643
        A,B,C,D,E = 134.15, 1.6146, -2.035, 1.5598, 0

    Args:
        T: temperature in Kelvin.
    Returns:
        Surface temperature in N/m.
    """
    return surface_tension(134.15, 1.6146, -2.035, 1.5598, 0, 647.3, T)


water = Solvent(molar_mass_water,
                density_water,
                specific_heat_capacity_water,
                specific_latent_heat_water,
                equilibrium_vapour_pressure_water,
                VapourBinaryDiffusionCoefficient(0.2190e-4, 273.15, 1.81),
                surface_tension_water,
                1.335)