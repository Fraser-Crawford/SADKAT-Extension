from environment import Atmosphere
from radial import RadialDroplet
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

if __name__ == '__main__':
    layers = [1,2,3,4,5,10]
    benchmark_droplet(UniformDroplet.from_mfs(test_solution, environment, gravity, radius, mfs, temperature), "Uniform",
                      efflorescence_termination=True)
    droplets = []
    plt.title("Radial Droplet: number of layers comparison")
    for layer in layers:
        droplets.append(RadialDroplet.from_mfs(test_solution,environment,gravity,radius,mfs,temperature,layer))
    for droplet in droplets:
        print(droplet.initial_layers)
        benchmark_droplet(droplet,droplet.initial_layers,efflorescence_termination=True)

    plt.legend()
    plt.show()