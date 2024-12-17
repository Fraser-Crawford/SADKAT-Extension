from environment import Atmosphere
from radial import RadialDroplet
from solution_definitions import aqueous_NaCl
from suspension import test_suspension, silica, dummy_density
from suspension_droplet import SuspensionDroplet, crossing_rate
from uniform import UniformDroplet
from viscous_defintions import viscous_aqueous_NaCl
import matplotlib.pyplot as plt
import numpy as np

from water import water

gravity = np.array([0, 0, 0])


def benchmark_droplet(droplet, label, efflorescence_threshold=0.45, efflorescence_termination=False):
    df = droplet.complete_trajectory(droplet.integrate(2,
                                                       eff_threshold=efflorescence_threshold,
                                                       terminate_on_efflorescence=efflorescence_termination))
    plt.plot(df.time, df.radius, label=f"{label}")
    plt.scatter(df.time.values[-1], df.radius.values[-1])


def plot_solution(solution):
    mfs = np.linspace(0.0, 1.0, 100)
    plt.plot(mfs, solution.density(mfs))
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
    plt.plot(df.time, df.real_enrichment, label="20 layer Surface Enrichment")
    plt.plot(df.time, df.predicted_enrichment, "--", label="Predicted Surface Enrichment")
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
    suspension = SuspensionDroplet.from_mfp(test_suspension(10e-9), Atmosphere(313), gravity, 50e-6, 0.3, 313,
                                            layers=20)
    df = suspension.complete_trajectory(suspension.integrate(2))
    uniform = UniformDroplet.from_mfs(aqueous_NaCl, Atmosphere(313), gravity, 50e-6, 0.0, 313)
    df2 = uniform.complete_trajectory(uniform.integrate(2))
    plt.plot(df2.time, df2.radius, "--")
    plt.plot(df.time, df.radius)
    plt.xlabel("Time / s")
    plt.ylabel("Droplet Radius / m")
    plt.show()

    plt.plot(df.time, df.surface_particle_concentration, label="Surface Particle Concentration")
    plt.plot(df.time, df.average_particle_concentration, label="Average Particle Concentration")
    plt.xlabel("Time / s")
    plt.ylabel("Number Concentration / m-3")
    plt.legend()
    plt.show()


def silica_bench(droplet_radius, silica_volume_fraction,temp, rh):
    silica_suspension = silica(180e-9 / 2)
    mass_fraction = silica_volume_fraction * 2200 / (
                (1 - silica_volume_fraction) * 1000 + silica_volume_fraction * 2200)
    layers = [100]
    for layer in layers:
        print(layer)
        suspension = SuspensionDroplet.from_mfp(silica_suspension,
                                                Atmosphere(temp, velocity=np.array([0.02, 0, 0]), relative_humidity=rh),
                                                gravity, droplet_radius, mass_fraction, temp, layer)
        df = suspension.complete_trajectory(suspension.integrate(20))
        print(np.max(df.time), np.min(df.radius))
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
        plt.title(
            f"Initial radius {droplet_radius*1e6:.1f} um, T = {temp:.0f} K, RH = {rh*100:.0f}%, {silica_volume_fraction*100:.1f}% v/v \n locking point time of {np.max(df.time):.2f} s, locking point radius of {np.min(df.radius) * 1e6:.2f} um")
        plt.plot(positions[-1], concentrations[-1])
        plt.plot(positions, concentrations, "--")
        plt.xlabel("Distance from center of droplet / m")
        plt.ylabel("Concentration / g/L")
        plt.show()


def pure_bench(radius, temperature, rh):
    environment = Atmosphere(temperature, rh)
    uniform = UniformDroplet.from_mfs(aqueous_NaCl, environment, gravity, radius, 0.0, temperature)
    df = uniform.complete_trajectory(uniform.integrate(1.6))
    print(np.interp(11.3e-6, df.radius[::-1], df.time[::-1]))
    plt.plot(df.time, df.radius)
    plt.show()


def linear_layer_concentrations():
    radial = RadialDroplet.from_mfs(viscous_aqueous_NaCl, Atmosphere(313), gravity, 50e-6, 0.1, 313, 20)
    plt.plot(radial.all_positions(), radial.linear_layer_concentrations())
    plt.scatter(radial.all_positions(), radial.linear_layer_concentrations())
    plt.show()


def mass_conservation():
    silica_suspension = silica(90e-9)
    volume = 1
    particle_volume = volume * 0.06e-2
    solvent_volume = volume - particle_volume
    mass_fraction = particle_volume * 2200 / (solvent_volume * 1000 + particle_volume * 2200)
    layers = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    for layer in layers:
        print(layer)
        suspension = SuspensionDroplet.from_mfp(silica_suspension, Atmosphere(303), gravity, 25.6e-6,
                                                mass_fraction, 303, layer)
        df = suspension.complete_trajectory(suspension.integrate(2))
        plt.plot(df.time, df.mass_particles)
        print(np.mean(df.mass_particles))
    plt.show()


# Given a RH and air speed value, calculate the kappa value of the linear r2 region.
def pure_water_probe(rh, air_speed, temperature):
    print(rh, air_speed, temperature)
    droplet = UniformDroplet.from_mfs(aqueous_NaCl, Atmosphere(temperature, rh, velocity=np.array([air_speed, 0, 0])),
                                      gravity, 25e-6, 0.0, temperature)
    print(droplet.wet_bulb_temperature)
    print(droplet.kappa)
    df = droplet.complete_trajectory(droplet.integrate(1.0))
    plt.plot(df.temperature)
    plt.plot(df.temperature)
    plt.show()
    temp_derivative = np.gradient(df.temperature, df.time)
    mask = np.abs(temp_derivative) < 0.01
    x = df.time[mask]
    y = np.array(df.radius[mask], dtype="float64") ** 2
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m

def wet_bulb_bench():
    temperature = 273.15+17.9
    rhs = np.linspace(0.05,0.4,8)
    data = 1e-12*np.array([-384.3789297883893, -357.5978481521835, -332.3947123690286, -308.0702741011096, -286.36919829362466, -264.6921321164674, -243.04471193334453, -221.7899312880983])
    sizes = [20e-6]
    gradients = []
    for size in sizes:
        droplets = [UniformDroplet.from_mfs(aqueous_NaCl, Atmosphere(temperature,relative_humidity=rh), gravity, size, 0.0, temperature, 0) for rh in rhs]
        for droplet in droplets:
            gradients += [-droplet.kappa/4]
        plt.plot(rhs,gradients,label=f"Still model")
    plt.plot(rhs, data,label=f"Experiment data")
    plt.xlabel("Relative Humidity")
    plt.ylabel("d(r^2)/dt / m2 s-1")
    plt.legend()
    plt.show()
    droplet = UniformDroplet.from_mfs(aqueous_NaCl, Atmosphere(temperature), gravity, 20e-6, 0.0, temperature, 0)
    droplet.velocity = np.array([1.0,0.0,0.0])
    print(droplet.reynolds_number)
    reynolds = ((data/np.array(gradients)-1)/(0.3*np.cbrt(droplet.schmidt_number)))**2
    plt.plot(rhs,reynolds)
    plt.xlabel("Relative Humidity")
    plt.ylabel("Reynolds Number")
    plt.show()
    df = droplet.complete_trajectory(droplet.integrate(2.0))
    plt.plot(df.time,df.sherwood_number)
    plt.show()

def poly2d(rh, v, c):
    return c[0] * rh ** 2 * v ** 2 + c[1] * rh * v ** 2 + c[2] * v ** 2 + c[3] * rh ** 2 * v + c[4] * rh * v + c[5] * v + c[6] * rh ** 2 + c[7] * rh + (rh * 0 + c[6])

def polymer_samples(particle_diameter,droplet_radius,rh,temperature,volume_fraction):
    suspension = dummy_density(particle_diameter/2.0)
    mass_fraction = volume_fraction * suspension.particle_density / (
            (1 - volume_fraction) * suspension.solvent.density(temperature) + volume_fraction * suspension.particle_density)
    droplet = SuspensionDroplet.from_mfp(suspension, Atmosphere(temperature, rh, velocity=np.array([0.02, 0, 0])),np.array([0,0,0]),droplet_radius,mass_fraction,temperature,layers=100)
    df = droplet.complete_trajectory(droplet.integrate(20))
    print(np.max(df.time), np.min(df.radius))
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
    plt.title(f"Particle radius of {particle_diameter*1e9/2.0:.1f} nm, locking point time of {np.max(df.time):.2f} s,\n from initial droplet radius {droplet_radius * 1e6:.2f} um to locking point radius of {np.min(df.radius) * 1e6:.2f} um")
    plt.plot(positions[-1], concentrations[-1])
    plt.plot(positions, concentrations, "--")
    plt.xlabel("Distance from center of droplet / m")
    plt.ylabel("Concentration / g/L")
    plt.show()

def wet_bulb_salts(rh,temperature):
    environment = Atmosphere(temperature,relative_humidity=rh,velocity=np.array([0.2,0.0,0.0]))
    droplet = UniformDroplet.from_mfs(aqueous_NaCl, environment, gravity, 20e-6, 0.0, temperature)
    df = droplet.complete_trajectory(droplet.integrate(2.0))
    fig, ax1 = plt.subplots()
    ax1.plot(df.time,df.temperature)
    ax1.hlines([temperature,environment.wet_bulb_temperature],df.time.min(),df.time.max(),linestyle="--")
    ax2 = ax1.twinx()
    ax2.plot(df.time,df.radius,color="black")
    plt.show()
    plt.plot(df.time,df.vapour_pressure)
    plt.plot(df.time,[water.equilibrium_vapour_pressure(t) for t in df.temperature])
    plt.show()

if __name__ == '__main__':
    wet_bulb_bench()
    #print(pure_water_probe(0.1,0.0,273.15+17.9))
    #polymer_samples(73.7e-9,23.8e-6,0.489,273.15+18,1.0e-2)
    # pure_bench(26.5e-6,303,0.1)
    #silica_bench(28.5e-6, 0.5 / 100, 294, 0.8)
    # radial_bench(100)
    # peclet_bench()
    # dummy_suspension_bench()
    # linear_layer_concentrations()
    # mass_conservation()
    # mfs = aqueous_NaCl.mass_fraction_from_activity(0.47)
    # concentration = aqueous_NaCl.concentration(mfs)
    # print(concentration)
    #wet_bulb_salts(0.05,293)

