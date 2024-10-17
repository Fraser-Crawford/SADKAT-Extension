import chemicals
from fit import rackett_equation, enthalpy_vapourisation, antoine_equation, surface_tension, \
    VapourBinaryDiffusionCoefficient
from solvent import Solvent

def yaws_diffusion_polynomial(T, A, B, C):
    return 1e-4 * (A + B * T + C * T ** 2)

def vapour_binary_diffusion_coefficient_func(T, D_ref, T_ref, lam):
    return D_ref * (T / T_ref) ** lam

molar_mass_ethanol = 2 * chemicals.periodic_table.C.MW + 6 * chemicals.periodic_table.H.MW + chemicals.periodic_table.O.MW  # g/mol

def density_ethanol(T):
    """Density of ethanol as function of temperature.

    Functional form and fit parameters taken from (T_min, T_max = 191, 513):
    http://ddbonline.ddbst.de/DIPPR105DensityCalculation/DIPPR105CalculationCGI.exe?component=2-ethanol

    Args:
        T: temperature in Kelvin.
    Returns:
        The ethanol density in kg/m^3.
    """

    A, B, C, D = 99.3974, 0.310729, 513.18, 0.30514
    return rackett_equation(A, B, C, D, T)


def specific_heat_capacity_ethanol(T):
    """Specific heat capacity of ethanol vs temperature.

    Lacking better data, we assume this is constant wrt temperature taking
    numerical values from NIST:
    https://webbook.nist.gov/cgi/cbook.cgi?ID=C64175&Mask=2

    Args:
        T: temperature in Kelvin (though not currently used).
    Returns:
        Specific heat capacity in J/K/kg.
    """

    molar_heat_capacity_ethanol = 112.3  # J/K/mol
    return molar_heat_capacity_ethanol / molar_mass_ethanol * 1000  # J/K/kg


def specific_latent_heat_ethanol(T):
    """Enthalpy of vaporisation per unit mass.

    Functional form and parameters taken from Majer and Svoboda (1985), as referenced by NIST here:
    https://webbook.nist.gov/cgi/cbook.cgi?ID=C64175&Mask=4.

    Fit parameters for temperature range 298-469 K:
        A (kJ/mol)  50.43
        alpha       -0.4475
        beta         0.4989
        Tc (K)     513.9

    Returns:
        Specific latent heat in J/kg.
    """

    H_vap = enthalpy_vapourisation(50.43, -0.4475, 0.4989, 513.9, T)  # J / mol
    return 1e-3 * H_vap / molar_mass_ethanol  # J / kg


def equilibrium_vapour_pressure_ethanol(T):
    """Parameterisation of vapour pressure for ethanol vs temperature.

    Parameterisation taken from:
    https://webbook.nist.gov/cgi/inchi?ID=C71238&Mask=4&Type=ANTOINE&Plot=on

    This particular fit is probably a bit inaccurate below 20 deg c.

    Args:
       T: temperature in Kelvin.
    Returns:
       Vapour pressure in Pa.
    """

    P_bar = antoine_equation(T, 5.37229, 1670.409, -40.191)
    # Parameterisation above give pressure in bars, so we convert into Pa:
    return 1e5 * P_bar


def surface_tension_ethanol(T):
    """Surface tension of ethanol as a function of temperature.

    Functional form and parameters taken from:
    http://ddbonline.ddbst.de/DIPPR106SFTCalculation/DIPPR106SFTCalculationCGI.exe

    DIPPR106 parameters:
        A     131.38
        B       5.5437
        C      -8.4826
        D       4.3164
        E       0
        Tc    516.2
        Tmin  180
        Tmax  513

    Args:
        T: temperature in Kelvin.
    Returns:
        Surface tension in N/m.
    """

    return surface_tension(131.38, 5.5437, -8.4826, 4.3164, 0, 516.2, T)


Ethanol = Solvent(molar_mass_ethanol,
                  density_ethanol,
                  specific_heat_capacity_ethanol,
                  specific_latent_heat_ethanol,
                  equilibrium_vapour_pressure_ethanol,
                  VapourBinaryDiffusionCoefficient(0.2190e-4, 293.15, 1.81),
                  surface_tension_ethanol, 1.36)

molar_mass_propanol = 3 * chemicals.periodic_table.C.MW + 8 * chemicals.periodic_table.H.MW + chemicals.periodic_table.O.MW  # g/mol


def density_propanol(T):
    """Density of propanol as function of temperature.

    Functional form and fit parameters taken from (T_min, T_max = 186, 507):
    http://ddbonline.ddbst.de/DIPPR105DensityCalculation/DIPPR105CalculationCGI.exe?component=2-propanol

    Args:
        T: temperature in Kelvin.
    Returns:
        The propanol density in kg/m^3.
    """

    A, B, C, D = 74.5237, 0.27342, 508.3, 0.235299
    return rackett_equation(A, B, C, D, T)


def specific_heat_capacity_propanol(T):
    """Specific heat capacity of propanol vs temperature.

    Lacking better data, we assume this is constant wrt temperature taking
    numerical values from NIST:
    https://webbook.nist.gov/cgi/cbook.cgi?ID=C71238&Mask=2

    Args:
        T: temperature in Kelvin (though not currently used).
    Returns:
        Specific heat capacity in J/K/kg.
    """

    molar_heat_capacity_propanol = 145  # J/K/mol
    return molar_heat_capacity_propanol / molar_mass_propanol * 1000  # J/K/kg


def specific_latent_heat_propanol(T):
    """Enthalpy of vaporisation per unit mass.

    Functional form and parameters taken from Majer and Svoboda (1985), as referenced by NIST here:
    https://webbook.nist.gov/cgi/cbook.cgi?ID=C64175&Mask=4.

    Fit parameters for temperature range 298-390 K:
        A (kJ/mol)  52.06
        alpha       -0.8386
        beta         0.6888
        Tc (K)     536.7

    Returns:
        Specific latent heat in J/kg.
    """
    H_vap = enthalpy_vapourisation(50.43, -0.4475, 0.4989, 513.9, T)  # J / mol
    return 1e-3 * H_vap / molar_mass_propanol  # J / kg


def equilibrium_vapour_pressure_propanol(T):
    """Parameterisation of vapour pressure for propanol vs temperature.

    Parameterisation taken from:
    https://webbook.nist.gov/cgi/inchi?ID=C71238&Mask=4&Type=ANTOINE&Plot=on

    This particular fit is probably a bit inaccurate below 20 deg c.

    Args:
       T: temperature in Kelvin.
    Returns:
       Vapour pressure in Pa.
    """

    P_bar = antoine_equation(T, 5.31, 1690.86, -51.80)
    # Parameterisation above give pressure in bars, so we convert into Pa:
    return 1e5 * P_bar


def surface_tension_propanol(T):
    """Surface tension of propanol as a function of temperature.

    Functional form and parameters taken from:
    http://ddbonline.ddbst.de/DIPPR106SFTCalculation/DIPPR106SFTCalculationCGI.exe

    DIPPR106 parameters:
        A      46.507
        B       0.90053
        C       0
        D       0
        E       0
        Tc    508.3
        Tmin  287
        Tmax  353

    Args:
        T: temperature in Kelvin.
    Returns:
        Surface tension in N/m.
    """

    return surface_tension(46.507, 0.90053, 0, 0, 0, 508.3, T)


Propanol = Solvent(molar_mass_propanol,
                   density_propanol,
                   specific_heat_capacity_propanol,
                   specific_latent_heat_propanol,
                   equilibrium_vapour_pressure_propanol,
                   VapourBinaryDiffusionCoefficient(0.2190e-4, 293.15, 1.81),
                   surface_tension_propanol, 1.3829)

molar_mass_butanol = 4 * chemicals.periodic_table.C.MW + 10 * chemicals.periodic_table.H.MW + chemicals.periodic_table.O.MW  # g/mol


def density_butanol(T):
    """Density of butanol as function of temperature.

    Functional form and fit parameters taken from (T_min, T_max = 213, 558):
    http://ddbonline.ddbst.de/DIPPR105DensityCalculation/DIPPR105CalculationCGI.exe

    Args:
        T: temperature in Kelvin.
    Returns:
        The butanol density in kg/m^3.
    """

    A, B, C, D = 9.87035, 0.0998069, 568.017, 0.126276
    return rackett_equation(A, B, C, D, T)


def specific_heat_capacity_butanol(T):
    """Specific heat capacity of butanol vs temperature.

    Lacking better data, we assume this is constant wrt temperature taking
    numerical values from NIST:
    https://webbook.nist.gov/cgi/cbook.cgi?ID=C71238&Mask=2

    Args:
        T: temperature in Kelvin (though not currently used).
    Returns:
        Specific heat capacity in J/K/kg.
    """

    molar_heat_capacity_butanol = 177  # J/K/mol
    return molar_heat_capacity_butanol / molar_mass_butanol * 1000  # J/K/kg


def specific_latent_heat_butanol(T):
    """Enthalpy of vaporisation per unit mass.

    Functional form and parameters taken from Majer and Svoboda (1985), as referenced by NIST here:
    https://webbook.nist.gov/cgi/cbook.cgi?ID=C78922&Mask=4

    Fit parameters for temperature range 298 - 372 K:
        A (kJ/mol)  52.6
        alpha       -1.462
        beta         1.0701
        Tc (K)     536.

    Args:
        T: temperature in Kelvin.
    Returns:
        Specific latent heat in J/kg.
    """
    H_vap = enthalpy_vapourisation(50.43, -0.4475, 0.4989, 513.9, T)  # J / mol
    return 1e-3 * H_vap / molar_mass_butanol  # J / kg


def equilibrium_vapour_pressure_butanol(T):
    """Parameterisation of vapour pressure for butanol vs temperature.

    Parameterisation taken from:
    https://webbook.nist.gov/cgi/cbook.cgi?ID=C71363&Mask=4&Type=ANTOINE&Plot=on

    This particular fit is probabaly a bit inaccurate below 20 deg c.

    Args:
       T: temperature in Kelvin.
    Returns:
       Vapour pressure in Pa.
    """

    P_bar = antoine_equation(T, 4.55, 1351.56, -93.34)
    # Parameterisation above give pressure in bars, so we convert into Pa:
    return 1e5 * P_bar


def surface_tension_butanol(T):
    """Surface tension of ethanol as a function of temperature.

    Functional form and parameters taken from:
    http://ddbonline.ddbst.de/DIPPR106SFTCalculation/DIPPR106SFTCalculationCGI.exe

    DIPPR106 parameters:
        A     72.697
        B       3.0297
        C      -4.2681
        D       2.4776
        E       0
        Tc    562.9
        Tmin  238
        Tmax  543

    Args:
        T: temperature in Kelvin.
    Returns:
        Surface tension in N/m.
    """

    return surface_tension(72.697, 3.0297, -4.2681, 2.4776, 0, 562.9, T)


Butanol = Solvent(molar_mass_butanol,
                  density_butanol,
                  specific_heat_capacity_butanol,
                  specific_latent_heat_butanol,
                  equilibrium_vapour_pressure_butanol,
                  VapourBinaryDiffusionCoefficient(0.2190e-4, 293.15, 1.81),
                  surface_tension_butanol, 1.3993)