import numpy as np

from fit import DensityVsMassFractionFit, ActivityVsMfsParametrisation
from viscous_solution import ViscousSolution
from water import water
import numpy.typing as npt

def aqueous_NaCl_diffusion(mfs:float | npt.NDArray[np.float_], temperature):
    D_0=1e-9*(1.955 - 20.42*mfs + 141.7*mfs**2 - 539.8*mfs**3 + 995.6*mfs**4 - 698.7*mfs**5)
    water_viscosity = lambda T : -1.748e-5*(temperature-273)+1.336e-3
    return D_0*temperature/293*water_viscosity(293)/water_viscosity(temperature)

viscous_aqueous_NaCl = ViscousSolution(water,
                                       58.44,
                                       2,
                                       0.3,
                                       2170,
                                       DensityVsMassFractionFit(
                                           [998.2, -55.33776, 1326.69542, -2131.05669, 2895.88613, -940.62808]),
                                       ActivityVsMfsParametrisation(np.flipud(
                                           [48.5226539, -158.04388699, 186.59427048, -93.88696437, 19.28939256,
                                            -2.99894206, -0.47652352, 1.])),
                                       1.5442,
                                       aqueous_NaCl_diffusion)