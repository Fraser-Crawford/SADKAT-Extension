from environment import Atmosphere
from radial import RadialDroplet
from solution_definitions import aqueous_NaCl
from suspension import test_suspension, silica
from suspension_droplet import SuspensionDroplet
from uniform import UniformDroplet
from viscous_defintions import viscous_aqueous_NaCl
import matplotlib.pyplot as plt
import numpy as np

gravity = np.array([0,0,0])

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

def peclet_bench():
    print("Starting 20 layer")
    radial = RadialDroplet.from_mfs(viscous_aqueous_NaCl, Atmosphere(313), gravity, 50e-6, 0.1, 313, 20)
    df = radial.complete_trajectory(radial.integrate(1))
    print("Integrated")
    plt.plot(df.time,df.real_enrichment,label="20 layer Surface Enrichment")
    plt.plot(df.time,df.predicted_enrichment,"--",label="Predicted Surface Enrichment")
    plt.plot(df.time, df.max_enrichment, "--", label="Max Surface Enrichment")
    print("Starting 100 layer")
    radial = RadialDroplet.from_mfs(viscous_aqueous_NaCl, Atmosphere(313), gravity, 50e-6, 0.1, 313, 100)
    df = radial.complete_trajectory(radial.integrate(1))
    print("Integrated")
    plt.plot(df.time, df.real_enrichment, label="100 layer Surface Enrichment")
    plt.legend()
    plt.show()

def radial_bench(layers):
    print("Starting radial benchmark")
    radial = RadialDroplet.from_mfs(viscous_aqueous_NaCl, Atmosphere(313), gravity, 50e-6, 0.1, 313, layers)
    df = radial.complete_trajectory(radial.integrate(1))
    print("Integrated")
    positions = []
    concentrations = []
    max_time = np.max(df.time)
    step = max_time / 20
    goal = 0
    for i in range(len(df.time.values)):
        concentrations.append(df.layer_concentrations[i])
        positions.append(df.layer_positions[i])
        time = df.time.values[i]
        if time >= goal:
            plt.plot(df.layer_positions[i], df.layer_concentrations[i])
            goal += step
    plt.plot(positions, concentrations, "--")
    plt.xlim(left=0)
    plt.xlabel("Distance from center of droplet / m")
    plt.ylabel("Concentration / g/L")
    plt.show()

def dummy_suspension_bench():
    suspension = SuspensionDroplet.from_mfp(test_suspension(10e-9),Atmosphere(313), gravity, 50e-6, 0.3, 313, layers=20)
    df = suspension.complete_trajectory(suspension.integrate(2))
    uniform = UniformDroplet.from_mfs(aqueous_NaCl, Atmosphere(313), gravity, 50e-6, 0.0, 313)
    df2 = uniform.complete_trajectory(uniform.integrate(2))
    plt.plot(df2.time, df2.radius, "--")
    plt.plot(df.time,df.radius)
    plt.xlabel("Time / s")
    plt.ylabel("Droplet Radius / m")
    plt.show()

    plt.plot(df.time,df.surface_particle_concentration,label="Surface Particle Concentration")
    plt.plot(df.time,df.average_particle_concentration,label="Average Particle Concentration")
    plt.xlabel("Time / s")
    plt.ylabel("Number Concentration / m-3")
    plt.legend()
    plt.show()

def silica_bench(droplet_radius,silica_volume_fraction):
    silica_suspension = silica(12e-9/2)
    mass_fraction = silica_volume_fraction*2200/((1-silica_volume_fraction)*1000+silica_volume_fraction*2200)
    time_result = []
    layers = [2,3,4,5,10,20,30,40,50,60,70,80,90,100]
    for layer in layers:
        print(layer)
        suspension = SuspensionDroplet.from_mfp(silica_suspension,Atmosphere(303),gravity,droplet_radius,mass_fraction,303,layer)
        df = suspension.complete_trajectory(suspension.integrate(2))
        print(np.max(df.time),np.min(df.radius))
        print()
        positions = []
        concentrations = []
        max_time = np.max(df.time)
        step = max_time / 20
        goal = step
        for i in range(len(df.time.values)):
            concentrations.append(df.layer_concentrations[i])
            positions.append(df.all_positions[i])
            time = df.time.values[i]
            if time >= goal:
                plt.plot(df.all_positions[i], df.layer_concentrations[i])
                goal += step
        plt.title(f"{layer} layers, locking point time of {np.max(df.time):.2f} s, locking point radius of {np.min(df.radius)*1e6:.2f} um")
        plt.plot(positions[-1], concentrations[-1])
        plt.plot(positions, concentrations, "--")
        plt.xlabel("Distance from center of droplet / m")
        plt.ylabel("Concentration / g/L")
        plt.show()
        time_result.append(np.max(df.time))
    plt.plot(layers, time_result)
    plt.xlabel("Layers")
    plt.ylabel("Locking Point Time / s")
    plt.show()

def pure_bench(radius,temperature,rh):
    environment = Atmosphere(temperature,rh)
    uniform = UniformDroplet.from_mfs(aqueous_NaCl, environment, gravity, radius,0.0,temperature)
    df = uniform.complete_trajectory(uniform.integrate(1.6))
    print(np.interp(11.3e-6,df.radius[::-1],df.time[::-1]))
    plt.plot(df.time,df.radius)
    plt.show()

def linear_layer_concentrations():
    radial = RadialDroplet.from_mfs(viscous_aqueous_NaCl, Atmosphere(313), gravity, 50e-6, 0.1, 313, 20)
    plt.plot(radial.all_positions(),radial.linear_layer_concentrations())
    plt.scatter(radial.all_positions(),radial.linear_layer_concentrations())
    plt.show()

def mass_conservation():
    silica_suspension = silica(90e-9)
    volume = 1
    particle_volume = volume * 0.06e-2
    solvent_volume = volume - particle_volume
    mass_fraction = particle_volume * 2200 / (solvent_volume * 1000 + particle_volume * 2200)
    layers = [2,3,4,5,6,7,8,9,10]
    for layer in layers:
        print(layer)
        suspension = SuspensionDroplet.from_mfp(silica_suspension, Atmosphere(303), gravity, 25.6e-6,
                                                mass_fraction, 303, layer)
        df = suspension.complete_trajectory(suspension.integrate(2))
        plt.plot(df.time,df.mass_particles)
        print(np.mean(df.mass_particles))
    plt.show()

if __name__ == '__main__':
    #pure_bench(26.5e-6,303,0.1)
    silica_bench(26.5e-6,0.6/100)
    #radial_bench(100)
    #peclet_bench()
    #dummy_suspension_bench()
    #linear_layer_concentrations()
    #mass_conservation()
