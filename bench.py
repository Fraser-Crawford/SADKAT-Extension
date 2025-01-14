from environment import Atmosphere
from radial import RadialDroplet
from solution_definitions import aqueous_NaCl
from suspension import test_suspension, silica, dummy_density
from suspension_droplet import SuspensionDroplet, crossing_rate
from uniform import UniformDroplet
from viscous_defintions import viscous_aqueous_NaCl
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from viscous_solution import ViscousSolution
from water import water
import pandas as pd

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


def radial_bench(solution: ViscousSolution, environments, radii, mfss, temperatures, data=None):
    print("Starting radial benchmark")
    print(len(environments), len(radii), len(mfss), len(temperatures))
    if data is None:
        data = [None] * len(environments)
    for environment, radius, mfs, temperature, datum in zip(environments, radii, mfss, temperatures, data):
        radial = RadialDroplet.from_mfs(solution, environment, gravity, radius, mfs, temperature, 50)
        df = radial.complete_trajectory(radial.integrate(5, terminate_on_efflorescence=True, eff_threshold=0.45))
        print("Integrated")
        max_time = np.max(df.time)
        step = max_time / 20
        goal = 0
        for i in range(len(df.time.values)):
            time = df.time.values[i]
            if time >= goal:
                plt.plot(df.layer_positions[i], df.layer_concentrations[i])
                goal += step
        plt.plot(df.layer_positions[i], df.layer_concentrations[i])
        plt.xlim(left=0)
        plt.xlabel("Distance from center of droplet / m")
        plt.ylabel("Concentration / g/L")
        plt.hlines(viscous_aqueous_NaCl.concentration(viscous_aqueous_NaCl.mass_fraction_from_activity(0.45)), 0,
                   radius, "black", "--")
        plt.title(
            f"{radius * 1e6:.1f} um radius droplet of {mfs:.3f} mass fraction of solute \n {environment.relative_humidity * 100:.1f}% RH, {environment.temperature:.1f} K and {np.linalg.norm(environment.velocity):.2f} m/s air flow speed. EFF. = {max_time:.2f} s")
        plt.show()
        if datum is not None:
            for droplet in datum:
                plt.scatter(droplet.relative_time, droplet.radius * 1e-6, s=4)
        plt.plot(df.time, df.measured_radius, color="black")
        plt.xlabel("Time / s")
        plt.ylabel("Droplet radius / m")
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


def silica_bench(droplet_radius, silica_volume_fraction, temp, rh):
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
            f"Initial radius {droplet_radius * 1e6:.1f} um, T = {temp:.0f} K, RH = {rh * 100:.0f}%, {silica_volume_fraction * 100:.1f}% v/v \n locking point time of {np.max(df.time):.2f} s, locking point radius of {np.min(df.radius) * 1e6:.2f} um")
        plt.plot(positions[-1], concentrations[-1])
        plt.plot(positions, concentrations, "--")
        plt.xlabel("Distance from center of droplet / m")
        plt.ylabel("Concentration / g/L")
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
    temperature = 273.15 + 17.9
    rhs = np.linspace(0.05, 0.4, 8)
    data = 1e-12 * np.array(
        [-384.3789297883893, -357.5978481521835, -332.3947123690286, -308.0702741011096, -286.36919829362466,
         -264.6921321164674, -243.04471193334453, -221.7899312880983])
    sizes = [20e-6]
    gradients = []
    for size in sizes:
        droplets = [
            UniformDroplet.from_mfs(aqueous_NaCl, Atmosphere(temperature, relative_humidity=rh), gravity, size, 0.0,
                                    temperature, 0) for rh in rhs]
        for droplet in droplets:
            gradients += [-droplet.kappa / 4]
        plt.plot(rhs, gradients, label=f"Still model")
    plt.plot(rhs, data, label=f"Experiment data")
    plt.xlabel("Relative Humidity")
    plt.ylabel("d(r^2)/dt / m2 s-1")
    plt.legend()
    plt.show()
    droplet = UniformDroplet.from_mfs(aqueous_NaCl, Atmosphere(temperature), gravity, 20e-6, 0.0, temperature, 0)
    droplet.velocity = np.array([1.0, 0.0, 0.0])
    print(droplet.reynolds_number)
    reynolds = ((data / np.array(gradients) - 1) / (0.3 * np.cbrt(droplet.schmidt_number))) ** 2
    plt.plot(rhs, reynolds)
    plt.xlabel("Relative Humidity")
    plt.ylabel("Reynolds Number")
    plt.show()
    df = droplet.complete_trajectory(droplet.integrate(2.0))
    plt.plot(df.time, df.sherwood_number)
    plt.show()


def poly2d(rh, v, c):
    return c[0] * rh ** 2 * v ** 2 + c[1] * rh * v ** 2 + c[2] * v ** 2 + c[3] * rh ** 2 * v + c[4] * rh * v + c[
        5] * v + c[6] * rh ** 2 + c[7] * rh + (rh * 0 + c[6])


def polymer_samples(particle_diameter, droplet_radius, rh, temperature, volume_fraction):
    suspension = dummy_density(particle_diameter / 2.0)
    mass_fraction = volume_fraction * suspension.particle_density / (
            (1 - volume_fraction) * suspension.solvent.density(
        temperature) + volume_fraction * suspension.particle_density)
    droplet = SuspensionDroplet.from_mfp(suspension, Atmosphere(temperature, rh, velocity=np.array([0.02, 0, 0])),
                                         np.array([0, 0, 0]), droplet_radius, mass_fraction, temperature, layers=100)
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
    plt.title(
        f"Particle radius of {particle_diameter * 1e9 / 2.0:.1f} nm, locking point time of {np.max(df.time):.2f} s,\n from initial droplet radius {droplet_radius * 1e6:.2f} um to locking point radius of {np.min(df.radius) * 1e6:.2f} um")
    plt.plot(positions[-1], concentrations[-1])
    plt.plot(positions, concentrations, "--")
    plt.xlabel("Distance from center of droplet / m")
    plt.ylabel("Concentration / g/L")
    plt.show()


def wet_bulb_salts(rh, temperature):
    environment = Atmosphere(temperature, relative_humidity=rh, velocity=np.array([0.2, 0.0, 0.0]))
    droplet = UniformDroplet.from_mfs(aqueous_NaCl, environment, gravity, 20e-6, 0.0, temperature)
    df = droplet.complete_trajectory(droplet.integrate(2.0))
    fig, ax1 = plt.subplots()
    ax1.plot(df.time, df.temperature)
    ax1.hlines([temperature, environment.wet_bulb_temperature], df.time.min(), df.time.max(), linestyle="--")
    ax2 = ax1.twinx()
    ax2.plot(df.time, df.radius, color="black")
    plt.show()
    plt.plot(df.time, df.vapour_pressure)
    plt.plot(df.time, [water.equilibrium_vapour_pressure(t) for t in df.temperature])
    plt.show()


def salt_data():
    directories = [
        fr"C:\Users\lh19417\OneDrive - University of Bristol\PHD\Documents\Data\24-11-21\24-11-21_1228s 40% salt",
        fr"C:\Users\lh19417\OneDrive - University of Bristol\PHD\Documents\Data\24-11-21\24-11-21_1240s 35% salt",
        fr"C:\Users\lh19417\OneDrive - University of Bristol\PHD\Documents\Data\24-11-21\24-11-21_1255s 30% salt",
        fr"C:\Users\lh19417\OneDrive - University of Bristol\PHD\Documents\Data\24-11-21\24-11-21_1310s 25% salt",
        fr"C:\Users\lh19417\OneDrive - University of Bristol\PHD\Documents\Data\24-11-21\24-11-21_1326s 20% salt",
        fr"C:\Users\lh19417\OneDrive - University of Bristol\PHD\Documents\Data\24-11-21\24-11-21_1340s 15% salt",
        fr"C:\Users\lh19417\OneDrive - University of Bristol\PHD\Documents\Data\24-11-21\24-11-21_1354s 10% salt",
        fr"C:\Users\lh19417\OneDrive - University of Bristol\PHD\Documents\Data\24-11-21\24-11-21_1407s 5% salt",
        fr"C:\Users\lh19417\OneDrive - University of Bristol\PHD\Documents\Data\24-11-21\24-11-21_1418s 0% salt"]
    data = []
    for folder in directories:
        files = glob(fr"{folder}\TrimmedDroplets\*.csv")
        results = [pd.read_csv(f"{file}") for file in files]
        data.append(results)
    temperatures = [273.15 + 17.7, 273.15 + 17.7, 273.15 + 17.7, 273.15 + 17.7, 273.15 + 17.41, 273.15 + 17.41,
                    273.15 + 17.41, 273.15 + 17.41, 273.15 + 17.53]
    RHs = [0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.0]
    air_flows = [np.array([0.6, 0, 0]), np.array([0.6, 0, 0]), np.array([0.6, 0, 0]), np.array([0.6, 0, 0]),
                 np.array([0.65, 0, 0]), np.array([0.75, 0, 0]), np.array([0.7, 0, 0]), np.array([0.7, 0, 0]),
                 np.array([0.6, 0, 0])]
    radii = [27.5e-6, 27.5e-6, 27.5e-6, 27.5e-6, 27.5e-6, 27.5e-6, 27.5e-6, 27.5e-6, 27.5e-6]
    solution = viscous_aqueous_NaCl
    mfss = [solution.concentration_to_solute_mass_fraction(22)] * len(temperatures)
    environments = [Atmosphere(t, rh, velocity=air_flow) for t, rh, air_flow in zip(temperatures, RHs, air_flows)]
    radial_bench(solution, environments, radii, mfss, temperatures, data)


def suspension_paper(Ts, RHs, R0s, silica_volume_fraction):
    silica_suspension = silica(180e-9 / 2)
    mass_fraction = silica_volume_fraction * 2200 / (
            (1 - silica_volume_fraction) * 1000 + silica_volume_fraction * 2200)
    for T, RH, R0 in zip(Ts, RHs, R0s):
        print(T, RH, R0)
        suspension = SuspensionDroplet.from_mfp(silica_suspension,
                                                Atmosphere(T, velocity=np.array([0.02, 0, 0]), relative_humidity=RH),
                                                gravity, R0, mass_fraction, T, 100)
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
            f"Initial radius {R0 * 1e6:.1f} um, T = {T:.0f} K, RH = {RH * 100:.0f}%, {silica_volume_fraction * 100:.1f}% v/v \n locking point time of {np.max(df.time):.2f} s, locking point radius of {np.min(df.radius) * 1e6:.2f} um")
        plt.plot(positions[-1], concentrations[-1])
        plt.plot(positions, concentrations, "--")
        plt.xlabel("Distance from center of droplet / m")
        plt.ylabel("Concentration / g/L")
        plt.show()


if __name__ == '__main__':
    Ts = [263, 273, 282, 289, 294, 303, 311, 318, 326]
    RHs = np.array([1.9, 5.0, 6.5, 3.2, 4.8, 13.6, 3.8, 3.0, 7.3])
    RHs /= 100
    R0s = np.array([30.93654,
                    28.4552,
                    30.75283,
                    30.74567,
                    28.75962,
                    26.53845,
                    29.03372,
                    30.01602,
                    31.56409,])
    R0s *= 1e-6
    suspension_paper(Ts,RHs,R0s,0.6/100)
