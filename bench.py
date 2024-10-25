from scipy.stats import uniform

from environment import Atmosphere
from radial import RadialDroplet
from solution_definitions import aqueous_ammonium_sulfate, aqueous_NaCl
from uniform import UniformDroplet
from viscous_defintions import viscous_aqueous_NaCl
import matplotlib.pyplot as plt
import numpy as np
test_solution = viscous_aqueous_NaCl
mfs = 0.2
temperature = 313
environment = Atmosphere(temperature,0.0)
gravity = np.array([0,0,0])
radius = 50e-6
def benchmark_droplet(droplet, label, efflorescence_threshold = 0.45, efflorescence_termination = False):
    df = droplet.complete_trajectory(droplet.integrate(2,
                                                       eff_threshold=efflorescence_threshold,
                                                       terminate_on_efflorescence=efflorescence_termination))
    plt.plot(df.time,df.radius,label=f"{label}")
    plt.scatter(df.time.values[-1],df.radius.values[-1])

def plot_solution(solution):
    mfs = np.linspace(0.0,1.0,100)
    plt.plot(mfs,solution.density(mfs))
    plt.title("Density Fit")
    plt.xlabel("Mass fraction of solute")
    plt.ylabel("Density / kg/m3")
    plt.show()
    plt.plot(mfs, solution.activity(mfs))
    plt.xlabel("Mass fraction of solute")
    plt.ylabel("Solution activity")
    plt.show()

if __name__ == '__main__':
    layers = 10
    temp = 313
    print("Start")
    radial = RadialDroplet.from_mfs(viscous_aqueous_NaCl, Atmosphere(temp), np.array([0, 0, 0]), 50e-6, 0.1, temp, layers)
    trajectory = radial.integrate(2)
    print("integrated")
    df = radial.complete_trajectory(trajectory)
    plt.plot(df.time,df.radius)
    positions = []
    concentrations = []

    for i in range(layers-1):
        positions += [[df["layer_boundaries"].values[j][i] for j in range(len(df["layer_boundaries"].values))]]
        plt.plot(df.time,positions[-1])
    positions += [[df["radius"].values[i] for i in range(len(df["radius"].values))]]
    print("positions set")
    for i in range(layers):
        concentrations += [[df["layer_concentration"].values[j][i] for j in range(len(df["layer_concentration"].values))]]


    uniform = UniformDroplet.from_mfs(aqueous_NaCl, Atmosphere(temp), np.array([0, 0, 0]), 50e-6, 0.1, temp, )
    trajectory = uniform.integrate(2)
    df2 = uniform.complete_trajectory(trajectory)
    plt.scatter(df2.time, df2.radius)
    plt.show()

    for r,conc in zip(positions,concentrations):
        plt.plot(df.time,conc)
    plt.scatter(df2.time,df2.concentration)
    plt.show()

    plt.plot(df.time,df.surface_diffusion)
    plt.show()
    mfss = []
    for i in range(layers):
        mfss += [[df["layer_mass_fraction_solute"].values[j][i] for j in
                  range(len(df["layer_mass_fraction_solute"].values))]]
        plt.plot(df.time,mfss[i])
    plt.show()
