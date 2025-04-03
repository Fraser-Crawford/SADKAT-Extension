import time
from cProfile import label
from dataclasses import dataclass

import scipy

import suspension
from environment import Atmosphere, thermal_conductivity_air
from fit import lorentz_lorenz, correct_radius
from radial import RadialDroplet
from rust import RustDroplet, DataDroplet
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
from scipy.optimize import curve_fit
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
    silica_suspension = silica(180e-9 / 2,
                               critical_shell_thickness=6,
                               critical_volume_fraction=np.pi/6,
                               max_volume_fraction=1,
                               rigidity=1)
    mass_fraction = silica_volume_fraction * 2200 / (
            (1 - silica_volume_fraction) * 1000 + silica_volume_fraction * 2200)
    for T, RH, R0 in zip(Ts, RHs, R0s):
        print(T, RH, R0)
        suspension = SuspensionDroplet.from_mfp(silica_suspension,
                                                Atmosphere(T, velocity=np.array([0.02, 0, 0]), relative_humidity=RH),
                                                gravity, R0, mass_fraction, T, 100)
        df = suspension.complete_trajectory(suspension.integrate(40))
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

def run(inputs,critical_shell_thickness):
    print(critical_shell_thickness)
    silica_suspension = silica(180e-9 / 2,
                               critical_shell_thickness=critical_shell_thickness,
                               max_volume_fraction=1,
                               rigidity=1)
    Ts,RHs,R0s,factors = inputs
    print(len(Ts),len(RHs),len(R0s),len(factors))
    result = []
    silica_volume_fraction = 0.6/100
    mass_fraction = silica_volume_fraction * 2200 / (
            (1 - silica_volume_fraction) * 1000 + silica_volume_fraction * 2200)
    for T,RH,R0 in zip(Ts,RHs,R0s):
        print(T)
        suspension = SuspensionDroplet.from_mfp(silica_suspension,
                                                Atmosphere(T, velocity=np.array([0.02, 0, 0]), relative_humidity=RH),
                                                gravity, R0, mass_fraction, T, 100)
        df = suspension.complete_trajectory(suspension.integrate(40))
        result.append(np.max(df.time))
    print(result)
    result = np.array(result)/factors
    print(result)
    return result

def find_parameters():
    Ts = np.array([263, 273, 282, 289, 294, 303, 311, 318, 326])
    #Found RHs = np.array([1.9, 5.0, 6.5, 3.2, 4.8, 13.6, 3.8, 3.0, 7.3])
    RHs = np.array([1.9, 5.0, 6.5, 3.2, 4.8, 5.0, 3.8, 3.0, 5.0])
    RHs /= 100
    R0s = np.array([30.93654, 28.4552, 30.75283, 30.74567, 28.75962, 26.53845, 29.03372, 30.01602, 31.56409])
    R0s *= 1e-6
    exp_times = np.array([11.55,5.58,4.00,2.89,2.26,1.31,1.24,1.04,0.87])
    popt,pcov = curve_fit(run,(Ts,RHs,R0s,exp_times),exp_times/exp_times,6,bounds=(0,np.inf))

def matrix():
    radii = np.array([10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0,100.0,110.0,120.0,130.0,140.0,150.0,160.0,170.0,180.0,190.0,200.0]*20)*1e-9
    volume_fraction = np.array([0.1]*20+[0.2]*20+[0.3]*20+[0.4]*20+[0.5]*20+[0.6]*20+[0.7]*20+[0.8]*20+[0.9]*20+[1.0]*20+
    [1.1]*20+[1.2]*20+[1.3]*20+[1.4]*20+[1.5]*20+[1.6]*20+[1.7]*20+[1.8]*20+[1.9]*20+[2.0]*20)/100
    get_locking((radii,volume_fraction),293,0.4,30e-6)

def water_test():
    mfs = 0.03
    sizes = np.logspace(-6,-5,10)
    for size in sizes:
        water_drop = UniformDroplet.from_mfs(aqueous_NaCl, Atmosphere(293, 0.45),gravity,size,mfs,293)
        trajectory = water_drop.complete_trajectory(water_drop.integrate(40,terminate_on_equilibration=True))
        plt.plot(trajectory.time,trajectory.radius)
    plt.show()

def rust_test():
    concentration = 200
    mfs = aqueous_NaCl.concentration_to_solute_mass_fraction(concentration)
    for i in range(1,101,10):
        droplet = RustDroplet("aqueous_NaCl","silica",i,293,0.45,0.0,90e-9)
        trajectory = droplet.integrate(40,30e-6,concentration,0.0,terminate_on_equilibration=True)
        df = droplet.complete_trajectory(trajectory)
        plt.plot(df.time,df.radius,label=f"Rust {i} layers")
        print()
    for i in range(1):
        droplet = RadialDroplet.from_mfs(viscous_aqueous_NaCl, Atmosphere(293, 0.45),gravity,30e-6,mfs,293)
        trajectory = droplet.integrate(40,terminate_on_equilibration=True)
    df = droplet.complete_trajectory(trajectory)
    plt.plot(df.time, df.radius,label="Python")
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Radius (m)")
    plt.show()

def rust_test2():
    mfs = aqueous_NaCl.concentration_to_solute_mass_fraction(100)
    start = time.time()
    for i in range(1):
        droplet = RustDroplet("aqueous_NaCl","silica",1,293,0.45,0.0,90e-9)
        trajectory = droplet.integrate(5,30e-6,100,0.0,terminate_on_equilibration=True)
    plt.plot(trajectory.t,trajectory.y[1],label="Rust")
    print(time.time() - start)
    start = time.time()
    for i in range(1):
        droplet = UniformDroplet.from_mfs(aqueous_NaCl, Atmosphere(293, 0.45),gravity,30e-6,mfs,293)
        trajectory = droplet.integrate(5,terminate_on_equilibration=True)
    print(time.time() - start)
    df = droplet.complete_trajectory(trajectory)
    plt.plot(df.time, df.temperature,label="Python")
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (K)")
    plt.show()

def rust_test3():
    print()
    mfs = aqueous_NaCl.concentration_to_solute_mass_fraction(100)
    for i in range(1):
        droplet = RustDroplet("aqueous_NaCl","silica",1,293,0.45,0.0,90e-9)
        trajectory = droplet.integrate(5,30e-6,100,0.0,terminate_on_equilibration=True)
    plt.plot(np.exp(trajectory.y[0]),trajectory.y[-2],label="Rust")
    plt.show()
    for i in range(1):
        droplet = UniformDroplet.from_mfs(aqueous_NaCl, Atmosphere(293, 0.45),gravity,30e-6,mfs,293)
        trajectory = droplet.integrate(5,terminate_on_equilibration=True)
        print(droplet.mass_solute())
    df = droplet.complete_trajectory(trajectory)
    plt.plot(df.mass_solvent, df.radius*1e6,label="Python")
    plt.legend()
    plt.xlabel("Mass of solvent (kg)")
    plt.ylabel("Radius (um)")
    plt.show()

def diffusion_test():
    mfs = viscous_aqueous_NaCl.concentration_to_solute_mass_fraction(10)
    droplet = RadialDroplet.from_mfs(viscous_aqueous_NaCl, Atmosphere(293, 0.45),gravity,30e-6,mfs,293,10)
    print(droplet.layer_mass_solute)
    print(droplet.change_in_solute_mass())

def vapour_comparison():
    T = np.linspace(260,300,100)
    plt.plot(T,water.equilibrium_vapour_pressure(T))
    def equilibrium_vapour_pressure_water(temperature):
        T_C = temperature - 273.15
        return 1e3 * 0.61161 * np.exp((18.678 - (T_C / 234.5)) * (T_C / (257.14 + T_C)))
    plt.plot(T,equilibrium_vapour_pressure_water(T))
    plt.show()

def heat_cap_difference():
    T = np.linspace(274, 360, 100)
    y = water.specific_heat_capacity(T)
    c = np.polyfit(T,y,6)
    print(c)
    y_prime = np.poly1d(c)(T)
    plt.plot(T,y)
    plt.plot(T,y_prime)
    plt.show()

def thermal_conductivity():
    T = np.linspace(274, 360, 100)
    y = thermal_conductivity_air(T)
    c = np.polyfit(T, y, 1)
    print(c)
    y_prime = np.poly1d(c)(T)
    plt.plot(T, y)
    plt.plot(T, y_prime)
    plt.show()

def density_diff():
    mfs = np.linspace(0,100,101)
    density = aqueous_NaCl.density(mfs)
    def density_rust(mfs):
        return np.poly1d([-940.62808, 2895.88613, -2131.05669, 1326.69542, -55.33776, 998.2])(np.sqrt(mfs))
    density_prime = density_rust(mfs)
    plt.plot(mfs, density)
    plt.plot(mfs, density_prime)
    plt.show()

def GregsonTest():
    Ts = [293,308,318]
    mfs = 0.01
    concentration = aqueous_NaCl.concentration(mfs)
    for T in Ts:
        print(T)
        droplet = RustDroplet("aqueous_NaCl","silica",10,T,0.0,0.0,90e-9)
        trajectory = droplet.integrate(40.0,24e-6,concentration,0.0,terminate_on_efflorescence=True,eff_threshold=0.45)
        df = droplet.complete_trajectory(trajectory)
        plt.plot(df.time, df.radius, label=T)
        droplet = UniformDroplet.from_mfs(aqueous_NaCl,Atmosphere(T),gravity,24e-6,mfs,T)
        trajectory = droplet.integrate(40.0, terminate_on_efflorescence=True,eff_threshold=0.45)
        df = droplet.complete_trajectory(trajectory)
        plt.plot(df.time, df.radius, label=f"{T} Uniform",linestyle="--")
    plt.legend()
    plt.show()

def layer_test():
    concentration = 25
    mfs = aqueous_NaCl.concentration_to_solute_mass_fraction(concentration)
    T = 313
    rh = 0.1

    layers = [10,20,40,60,80,100]

    for layer in layers:
        start = time.time()
        droplet = RustDroplet("aqueous_NaCl", "silica", layer, T, rh, 0.0, 90e-9)
        trajectory = droplet.integrate(40.0, 24e-6, concentration, 0.0,terminate_on_efflorescence=True,eff_threshold=0.45)
        print(trajectory.message)
        df = droplet.complete_trajectory(trajectory)
        timer = time.time() - start
        plt.plot(df.time,df.radius,label=f"Rust {layer} layers, {timer:.2f} s")

    start = time.time()
    droplet = RadialDroplet.from_mfs(viscous_aqueous_NaCl,Atmosphere(T,rh),gravity,24e-6,mfs,T,10)
    trajectory = droplet.integrate(40.0,terminate_on_efflorescence=True, eff_threshold=0.45)
    print(trajectory.message)
    df = droplet.complete_trajectory(trajectory)
    timer = time.time() - start
    plt.plot(df.time, df.radius, linestyle="--",label=f"Radial 10 layers, {timer:.2f} s")

    start = time.time()
    droplet = UniformDroplet.from_mfs(viscous_aqueous_NaCl, Atmosphere(T, rh), gravity, 24e-6, mfs, T)
    trajectory = droplet.integrate(40.0, terminate_on_efflorescence=True, eff_threshold=0.45)
    print(trajectory.message)
    df = droplet.complete_trajectory(trajectory)
    timer = time.time() - start
    plt.plot(df.time, df.radius, linestyle="-.",label=f"Uniform, {timer:.2f} s")
    plt.legend()
    plt.ylabel("Droplet Radius / m")
    plt.xlabel("Time / s")
    plt.title(f"Parity Results from Rust Vs. Old methods\nT = {T}, relative humidity = {rh*100}%, C = {concentration} g/L")
    plt.show()

def get_locking(inputs,T,RH,R0):
    radii,volume_fraction = inputs
    mass_fraction = volume_fraction * 2200 / ((1 - volume_fraction) * 1000 + volume_fraction * 2200)
    result = []
    for radius, mfp,vfp in zip(radii, mass_fraction,volume_fraction):
        print(radius,vfp*100)
        silica_suspension = silica(radius,critical_shell_thickness=500e-9/radius)
        suspension = SuspensionDroplet.from_mfp(silica_suspension,
                                                Atmosphere(T, velocity=np.array([0.02, 0, 0]), relative_humidity=RH),
                                                gravity, R0, mfp, T, 100)
        df = suspension.complete_trajectory(suspension.integrate(40))
        result.append(np.max(df.time))
        print(result[-1])
        print()
    print(result)

def locking_rust_temperature():
    Ts = np.array([263, 273, 282, 289, 294, 303, 311, 318, 326])
    N = len(Ts)
    R0s = np.array([30.93654, 28.4552, 30.75283, 30.74567, 28.75962, 26.53845, 29.03372, 30.01602, 31.56409])
    R0s *= 1e-6
    RHs = [0,0.1]
    exp_times = np.array([11.55, 5.58, 4.00, 2.89, 2.26, 1.31, 1.24, 1.04, 0.87])
    concentration = 0.6/100*2200
    print(concentration)
    plt.scatter(Ts,exp_times, label="Experimental",marker="D",zorder=10)
    results = []
    for RH in RHs:
        y = []
        input_rh = [RH]*N
        for T, rh, R0, exp_time in zip(Ts, input_rh, R0s,exp_times):
            print(T,rh,R0)
            droplet = RustDroplet("aqueous_NaCl", "silica",100, T, rh, 0.03, 90e-9)
            trajectory = droplet.integrate(40.0, R0, 0.0, concentration,
                                           terminate_on_locking=True,locking_threshold=300.0e-9)
            print(trajectory.message)
            y += [trajectory.t_events[0][0]]
            print(f"{trajectory.t_events[0][0]:.2f} s modelled, {exp_time:.2f} s measured")
            print()
        results.append(y)
    plt.fill_between(Ts,results[0],results[1],alpha=0.3,label="RH Uncertainty\nInterval")
    plt.legend()
    plt.xlabel("Temperature / K")
    plt.ylabel("Locking Point Time / s")
    plt.yscale("log")
    plt.show()

def locking_rust_rh():
    RHs = np.array([0,10,20,30,40,50,60,70,80,90])
    N = len(RHs)
    R0s = np.array([29.12481,
28.84665,
28.8417,
28.84905,
28.8639,
28.49274,
28.85276,
28.73958,
28.79753,
29.09102
])
    R0s *= 1e-6
    T = 294
    exp_times = np.array([1.91984,
2.13332,
2.49858,
2.96926,
3.6114,
4.68208,
6.02246,
9.05544,
14.52697,
33.87833
])
    concentration = 0.5 / 100 * 2200
    print(concentration)
    plt.scatter(RHs, exp_times, label="Experimental",marker="D",zorder=10)

    RH_minus = RHs - 5
    results = []
    y1 = []
    for rh, R0, exp_time in zip(RH_minus, R0s, exp_times):
        print(T, rh, R0)
        droplet = RustDroplet("aqueous_NaCl", "silica", 100, T, rh/100, 0.03, 90e-9)
        trajectory = droplet.integrate(100.0, R0, 0.0, concentration,
                                       terminate_on_locking=True, locking_threshold=300.0e-9)
        print(trajectory.message)
        y1 += [trajectory.t_events[0][0]]
        print(f"{trajectory.t_events[0][0]:.2f} s modelled, {exp_time:.2f} s measured")
        print()
    results.append(y1)
    RH_plus = RHs + 5
    y2 = []
    for rh, R0, exp_time in zip(RH_plus, R0s, exp_times):
        print(T, rh, R0)
        droplet = RustDroplet("aqueous_NaCl", "silica", 100, T, rh/100, 0.03, 90e-9)
        trajectory = droplet.integrate(100.0, R0, 0.0, concentration,
                                       terminate_on_locking=True, locking_threshold=300.0e-9)
        print(trajectory.message)
        y2 += [trajectory.t_events[0][0]]
        print(f"{trajectory.t_events[0][0]:.2f} s modelled, {exp_time:.2f} s measured")
        print()
    results.append(y2)
    plt.fill_between(RHs,results[0],results[1],alpha=0.3,label="RH Uncertainty\nInterval")
    plt.legend()
    plt.xlabel("Relative Humidity / %")
    plt.ylabel("Locking Point Time / s")
    plt.yscale("log")
    plt.show()

def test_particle_concentrations():
    concentration = 2.0 / 100 * 2200
    layers = 100
    for U in [0.0,0.05,0.10,0.15,0.20,0.25]:
        droplet = RustDroplet("aqueous_NaCl", "silica", layers, 293, 0.0, U, 90e-9)
        trajectory = droplet.integrate(40.0, 30e-6, 0.0, concentration,
                                       terminate_on_locking=True, locking_threshold=300.0e-9)
        df = droplet.complete_trajectory(trajectory)
        print(trajectory.message)
        print(np.max(df.time))
        for boundaries,concentrations in zip(df.true_boundaries,df.layer_particle_concentrations):
            plt.step(boundaries[:-1],concentrations)
        plt.show()

        for boundaries,concentrations in zip(df.true_boundaries,df.layer_concentrations):
            plt.step(boundaries[:-1],concentrations)
        plt.show()

def test_solute_concentrations():
    layers=100
    droplet = RustDroplet("aqueous_NaCl", "silica", layers, 293, 0.0, 0.0, 90e-9)
    trajectory = droplet.integrate(40.0, 30e-6, 25, 0.0
                                   ,terminate_on_efflorescence=True,eff_threshold=0.45)
    print(trajectory.message)
    df = droplet.complete_trajectory(trajectory)
    print(np.max(df.time))
    for boundaries, concentrations in zip(df.true_boundaries, df.layer_concentrations):
        plt.step(boundaries,np.concatenate(([concentrations[0]],concentrations)))
    plt.show()

    for i in range(layers-1):
        plt.plot(df.time,[position[i] for position in df.layer_positions])
    plt.plot(df.time,df.radius)
    plt.show()

    for i in range(0,10):
        plt.plot(df.time,[concentrations[i] for concentrations in df.layer_concentrations])
    plt.show()

    for i in range(0,layers-1):
        plt.plot(df.time,[concentrations[i] for concentrations in df.layer_particle_concentrations],label=i)
    plt.legend()
    plt.show()

def test_air_flow_time():
    Us = np.linspace(0.0,30.0,31)
    concentration = 0.6 / 100 * 2200
    layers = 100
    times = []
    for U in Us:
        print(U)
        droplet = RustDroplet("aqueous_NaCl", "silica", layers, 293, 1.0, U, 90e-9)
        trajectory = droplet.integrate(40.0, 30e-6, 0.0, concentration,
                                       terminate_on_locking=True, locking_threshold=300.0e-9)
        df = droplet.complete_trajectory(trajectory)
        times.append(np.max(df.time))
    plt.plot(Us,times)
    plt.xlabel("Air flow speed / ms-1")
    plt.ylabel("Locking point time / s")
    plt.title("Air flow speed vs. locking point time\n 293 K, 0% RH, silica NP")
    plt.show()

def test_air_flow_size():
    Us = np.linspace(0.0,30.0,31)
    concentration = 0.6 / 100 * 2200
    layers = 50
    radii = []
    for U in Us:
        print(U)
        droplet = RustDroplet("aqueous_NaCl", "silica", layers, 293, 0.5, U, 65e-9)
        trajectory = droplet.integrate(40.0, 30e-6, 0.0, concentration,
                                       terminate_on_locking=True, locking_threshold=100.0e-9)
        df = droplet.complete_trajectory(trajectory)
        radii.append(np.min(df.radius))
    plt.plot(Us,radii)
    plt.xlabel("Air flow speed / ms-1")
    plt.ylabel("Locking point radius / m")
    plt.title("Air flow speed vs. locking point radius\n 293 K, 0% RH, silica NP")
    plt.show()

def convection_fitting():
    from glob import glob
    bins = 30
    threshold = 10000
    shell_thickness = 500e-9
    directory = fr"C:\Users\lh19417\OneDrive - University of Bristol\PHD\Documents\Papers\Solid particle suspensions\CONVECTION DATA\200 sccm\Droplets"
    files = glob(fr"{directory}\*.csv")
    results = [pd.read_csv(f"{file}") for file in files]
    x = np.array([])
    y = np.array([])
    for droplet in results:
        x = np.append(x,droplet.relative_time)
        y = np.append(y,droplet.radius*1e-6)
    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]
    data, x_e, y_e = np.histogram2d(x,y,bins=bins,density=True,range=np.array([[np.min(x),np.max(x)],[np.min(y),np.max(y)]]))
    z = scipy.interpolate.interpn((0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])), data, np.vstack([x, y]).T, method="splinef2d",bounds_error=False)
    z[np.where(np.isnan(z))] = 0.0
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    min = np.min(z[z > threshold])
    max = np.max(z[z > threshold])*2
    plt.scatter(x[z > threshold], y[z > threshold], s=25, c="blue", alpha=(z[z > threshold] - min) / max,edgecolors="none")

    directory = fr"C:\Users\lh19417\OneDrive - University of Bristol\PHD\Documents\Papers\Solid particle suspensions\CONVECTION DATA\100 sccm\Droplets"
    files = glob(fr"{directory}\*.csv")
    results = [pd.read_csv(f"{file}") for file in files]
    x = np.array([])
    y = np.array([])
    for droplet in results:
        x = np.append(x, droplet.relative_time)
        y = np.append(y, droplet.radius * 1e-6)
    data, x_e, y_e = np.histogram2d(x, y, bins=bins, density=True, range=np.array([[np.min(x), np.max(x)], [np.min(y), np.max(y)]]))
    z = scipy.interpolate.interpn((0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])), data, np.vstack([x, y]).T,
                                  method="splinef2d", bounds_error=False)
    z[np.where(np.isnan(z))] = 0.0
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    min = np.min(z[z>threshold])
    max = np.max(z[z>threshold])*2
    plt.scatter(x[z>threshold], y[z>threshold], s=25, c="orange", alpha=(z[z>threshold]-min)/max,edgecolors="none")
    Us = [0.0,0.5]
    cs = ["red","black"]
    R0s = [24e-6,24e-6]
    concentration = 2.0 / 100 * 2200
    max_times = []
    min_radii = []
    for U,c,R0 in zip(Us,cs,R0s):
        print(U)

        droplet = RustDroplet("aqueous_NaCl", "silica", 50, 273.15+17.66, 0.46, U, 65e-9)
        trajectory = droplet.integrate(40.0, R0, 0.0, concentration,
                                       terminate_on_locking=True, locking_threshold=shell_thickness)

        print(trajectory.message)
        df = droplet.complete_trajectory(trajectory)
        n = np.array([lorentz_lorenz(1-phi,phi,1.335,1.473) for phi in df.particle_volume_fraction])
        new_radii = correct_radius(df.radius,n,1.335)
        plt.plot(df.time,new_radii,label=f"{U:.1f} m/s",c=c,linewidth=3,alpha=0.5)
        max_times.append(np.max(df.time))
        min_radii.append(np.min(new_radii))
    plt.scatter(max_times, min_radii, marker="o", s=200, c="black", zorder=9)
    plt.scatter(max_times,min_radii,marker="o",s=100,c="orange",zorder=10)
    plt.xlabel("Time / s")
    plt.ylabel("Droplet Radius / m")
    plt.title(f"Convectional Mixing at 45% RH, 17.7 C \n Shell thickness = {shell_thickness*1e9:.0f} nm")
    plt.xlim(0,4.0)
    plt.ylim(5e-6,25e-6)
    plt.legend()
    plt.show()

def interval_convection_fitting():
    from glob import glob
    shell_thickness = 500e-9
    directory = fr"C:\Users\lh19417\OneDrive - University of Bristol\PHD\Documents\Papers\Solid particle suspensions\CONVECTION DATA\200 sccm\Droplets"
    files = glob(fr"{directory}\*.csv")
    results = [pd.read_csv(f"{file}") for file in files]
    times_array = [droplet.relative_time for droplet in results]
    length_array = np.array([len(times) for times in times_array])
    longest_times = max(times_array, key=len)
    longest = np.max(length_array)
    radii = np.zeros(longest)
    normalisation_factors = np.array([np.count_nonzero(length_array >= i) for i in range(1, longest + 1)])
    for droplet in results:
        radii[:len(droplet.relative_time)] += droplet.radius*1e-6
    mean_radii = radii / normalisation_factors
    variances = np.zeros(len(mean_radii))
    for droplet in results:
        variances[:len(droplet.radius)] += (droplet.radius*1e-6 - mean_radii[:len(droplet.radius)])**2
    variances /= normalisation_factors
    standard_deviations = np.sqrt(variances)
    x = longest_times
    plt.fill_between(x, mean_radii - standard_deviations, mean_radii + standard_deviations,
                     alpha=0.3)
    plt.plot(x, mean_radii, linewidth=4)

    directory = fr"C:\Users\lh19417\OneDrive - University of Bristol\PHD\Documents\Papers\Solid particle suspensions\CONVECTION DATA\100 sccm\Droplets"
    files = glob(fr"{directory}\*.csv")
    results = [pd.read_csv(f"{file}") for file in files]
    times_array = [droplet.relative_time for droplet in results]
    length_array = np.array([len(times) for times in times_array])
    longest_times = max(times_array, key=len)
    longest = np.max(length_array)
    radii = np.zeros(longest)
    normalisation_factors = np.array([np.count_nonzero(length_array >= i) for i in range(1, longest + 1)])
    for droplet in results:
        radii[:len(droplet.relative_time)] += droplet.radius[~np.isnan(droplet.radius)]*1e-6
    mean_radii = radii / normalisation_factors
    variances = np.zeros(len(mean_radii))
    for droplet in results:
        variances[:len(droplet.radius)] += (droplet.radius*1e-6 - mean_radii[:len(droplet.radius)])**2
    variances /= normalisation_factors
    standard_deviations = np.sqrt(variances)
    x = longest_times
    plt.fill_between(x, mean_radii - standard_deviations, mean_radii + standard_deviations,
                     alpha=0.3)
    plt.plot(x, mean_radii, linewidth=4)

    Us = [0.0,0.5]
    cs = ["red","black"]
    R0s = [24e-6,24e-6]
    concentration = 2.0 / 100 * 2200
    max_times = []
    min_radii = []
    for U,c,R0 in zip(Us,cs,R0s):
        print(U)

        droplet = RustDroplet("aqueous_NaCl", "silica", 50, 273.15+17.66, 0.46, U, 65e-9)
        trajectory = droplet.integrate(40.0, R0, 0.0, concentration,
                                       terminate_on_locking=True, locking_threshold=shell_thickness)

        print(trajectory.message)
        df = droplet.complete_trajectory(trajectory)
        n = np.array([lorentz_lorenz(1-phi,phi,1.335,1.473) for phi in df.particle_volume_fraction])
        new_radii = correct_radius(df.radius,n,1.335)
        plt.plot(df.time,new_radii,label=f"{U:.1f} m/s",c=c,linewidth=3,alpha=1.0,linestyle=":")
        max_times.append(np.max(df.time))
        min_radii.append(np.min(new_radii))
    plt.scatter(max_times, min_radii, marker="D", s=200, c="black", zorder=9)
    plt.scatter(max_times,min_radii,marker="D",s=100,c="orange",zorder=10)
    plt.xlabel("Time / s")
    plt.ylabel("Droplet Radius / m")
    #plt.title(f"Convectional Mixing at 45% RH, 17.7 C \n Shell thickness = {shell_thickness*1e9:.0f} nm")
    plt.xlim(0,4.0)
    plt.ylim(5e-6,25e-6)
    plt.legend()
    plt.show()

def diffusion_fitting_air_flow():
    layers = 50
    r0 = 30e-6
    Us = np.linspace(0.05,2,40)
    Ds = []
    for U in Us[1:]:
        print(U)
        droplet = RustDroplet("aqueous_NaCl", "silica", layers, 273.15 + 20, 0.5, U, 500e-9)
        trajectory = droplet.integrate(40.0, r0, 0.0, 2.0 / 100 * 2200,
                                       terminate_on_locking=True, locking_threshold=300e-9)
        print(trajectory.message)
        df = droplet.complete_trajectory(trajectory)
        mask = df.time > 0.5
        coeff = np.polyfit(np.array(df.time[mask],dtype=float),np.array((df.radius[mask]*2)**2,dtype=float),1)
        kappa = -coeff[0]
        for index,time in enumerate(df.time):
            if time > 2.0:
                concentration_profile = df.layer_particle_concentrations[index]
                break
        Pe = np.log(concentration_profile[:int(0.8*layers)][-1]/concentration_profile[0])*2/0.8**2
        print(Pe)
        D = kappa/(8*Pe) - 1.38e-23*293.15/(6*np.pi*1e-3*500e-9)
        Ds.append(D)
    plt.plot(Us[1:],2/9*r0*Us[1:]/(2*np.pi**2*np.log(2)*(2+np.pi)*(1+1000/18.13)),linestyle="--",label="A=2/9",linewidth=3)
    plt.plot(Us[1:], 1 / 9 * r0 * Us[1:] / (2 * np.pi ** 2 * np.log(2) * (2 + np.pi) * (1 + 1000 / 18.13)), linestyle="--",linewidth=3,
             label="A=1/9")
    plt.plot(Us[1:], 2/27 * r0 * Us[1:] / (2 * np.pi ** 2 * np.log(2) * (2 + np.pi) * (1 + 1000 / 18.13)), linestyle="--",linewidth=3,
             label="A=2/27")
    plt.plot(Us[1:], 1 / 27 * r0 * Us[1:] / (2 * np.pi ** 2 * np.log(2) * (2 + np.pi) * (1 + 1000 / 18.13)), linestyle="--",
             label="A=1/27",linewidth=3)
    plt.scatter(Us[1:], Ds, marker="D", label="Extracted",zorder=10)
    plt.legend()
    plt.xlabel("Air Flow Speed / m s-1")
    plt.ylabel("Diffusion Coefficient / m2 s-1")
    plt.show()

def diffusion_fitting_size():
    layers = 50
    r0s = np.linspace(10e-6,30e-6,20)
    U = 0.5
    Ds = []
    for r0 in r0s:
        print(r0)
        droplet = RustDroplet("aqueous_NaCl", "silica", layers, 273.15 + 20, 0.5, U, 500e-9)
        trajectory = droplet.integrate(40.0, r0, 0.0, 2.0 / 100 * 2200,
                                       terminate_on_locking=True, locking_threshold=300e-9)
        print(trajectory.message)
        df = droplet.complete_trajectory(trajectory)
        mask = df.time > 0.5
        coeff = np.polyfit(np.array(df.time[mask], dtype=float), np.array((df.radius[mask] * 2) ** 2, dtype=float), 1)
        kappa = -coeff[0]
        threshold = 0.5*np.max(df.time)
        for index, time in enumerate(df.time):
            if time > threshold:
                concentration_profile = df.layer_particle_concentrations[index]
                break
        Pe = np.log(concentration_profile[:int(0.8 * layers)][-1] / concentration_profile[0]) * 2 / 0.8 ** 2
        print(Pe)
        D = kappa / (8 * Pe) - 1.38e-23 * 293.15 / (6 * np.pi * 1e-3 * 500e-9)
        Ds.append(D)
    plt.plot(r0s, 2 / 9 * r0s * U / (2 * np.pi ** 2 * np.log(2) * (2 + np.pi) * (1 + 1000 / 18.13)),
             linestyle="--", label="A=2/9", linewidth=3)
    plt.plot(r0s, 1 / 9 * r0s * U / (2 * np.pi ** 2 * np.log(2) * (2 + np.pi) * (1 + 1000 / 18.13)),
             linestyle="--", linewidth=3,
             label="A=1/9")
    plt.plot(r0s, 2 / 27 * r0s * U / (2 * np.pi ** 2 * np.log(2) * (2 + np.pi) * (1 + 1000 / 18.13)),
             linestyle="--", linewidth=3,
             label="A=2/27")
    plt.plot(r0s, 1 / 27 * r0s * U / (2 * np.pi ** 2 * np.log(2) * (2 + np.pi) * (1 + 1000 / 18.13)),
             linestyle="--",
             label="A=1/27", linewidth=3)
    plt.scatter(r0s, Ds, marker="D", label="Extracted", zorder=10)
    plt.legend()
    plt.xlabel("Initial Droplet Radius / m")
    plt.ylabel("Diffusion Coefficient / m2 s-1")
    plt.show()

def different_sizes_convection():
    rps = np.logspace(-10,-6,40)
    particle_sizes = [30.5e-9/2,86.1e-9/2,121.5e-9/2]
    volumes = []
    for rp in rps:
        print(rp)
        droplet = RustDroplet("aqueous_NaCl", "silica", 50, 273.15 + 20, 0.46, 0.0, rp)
        trajectory = droplet.integrate(40.0, 23e-6, 0.0, 1.0 / 100 * 2200,
                                       terminate_on_locking=True, locking_threshold=400e-9)
        print(trajectory.message)
        df = droplet.complete_trajectory(trajectory)
        volumes.append(np.min(df.radius/23e-6)**3)
    plt.plot(rps,np.array(volumes))
    plt.scatter(particle_sizes,np.interp(particle_sizes,rps,volumes))
    plt.xlabel("Nanoparticle Radius / m")
    plt.ylabel("Fraction of Volume Remaining")
    plt.xscale("log")
    plt.title("How Nanoparticle Radius Affects Locking Point at 0.0 m/s Airflow")
    plt.show()

def different_size_convection_experiment():
    colours = ["blue", "orange", "green"]
    fast_directories = [
        fr"C:\Users\lh19417\OneDrive - University of Bristol\PHD\Documents\Papers\Solid particle suspensions\CONVECTION DATA\Sample A 200 sccm",
        fr"C:\Users\lh19417\OneDrive - University of Bristol\PHD\Documents\Papers\Solid particle suspensions\CONVECTION DATA\Sample C 200 sccm",
        fr"C:\Users\lh19417\OneDrive - University of Bristol\PHD\Documents\Papers\Solid particle suspensions\CONVECTION DATA\Sample D 200 sccm"]
    slow_directories = [
        fr"C:\Users\lh19417\OneDrive - University of Bristol\PHD\Documents\Papers\Solid particle suspensions\CONVECTION DATA\Sample A 100 sccm",
        fr"C:\Users\lh19417\OneDrive - University of Bristol\PHD\Documents\Papers\Solid particle suspensions\CONVECTION DATA\Sample C 100 sccm",
        fr"C:\Users\lh19417\OneDrive - University of Bristol\PHD\Documents\Papers\Solid particle suspensions\CONVECTION DATA\Sample D 100 sccm"]
    handle = []
    for directory, colour in zip(slow_directories, colours):
        files = glob(fr"{directory}\TrimmedDroplets\*.csv")
        results = [pd.read_csv(f"{file}") for file in files]
        starting_radii = np.array([droplet.radius[0] for droplet in results])
        times_array = [droplet.relative_time for droplet in results]
        length_array = np.array([len(times) for times in times_array])
        longest_times = max(times_array, key=len)
        final_time = np.median([times.values[-1] for times in times_array])
        longest = np.max(length_array)
        volume_fractions = np.zeros(longest)
        normalisation_factors = np.array([np.count_nonzero(length_array >= i) for i in range(1, longest + 1)])
        for droplet, starting_radius in zip(results, starting_radii):
            volume_fractions[:len(droplet.relative_time)] += droplet.radius ** 3 / starting_radius ** 3
        mean_volume_fractions = volume_fractions / normalisation_factors
        variances = np.zeros(len(mean_volume_fractions))
        for droplet, starting_radius in zip(results, starting_radii):
            variances[:len(droplet.radius)] += (droplet.radius ** 3 / starting_radius ** 3 - mean_volume_fractions[
                                                                                             :len(droplet.radius)]) ** 2
        variances /= normalisation_factors
        standard_deviations = np.sqrt(variances)
        x = longest_times / final_time
        plt.fill_between(x, mean_volume_fractions - standard_deviations, mean_volume_fractions + standard_deviations,
                         alpha=0.3)
        plt.plot(x, mean_volume_fractions, linewidth=4)
        handle.append(plt.scatter(0, 0, s=4, c=colour))

    plt.ylim(0, 0.2)
    plt.xlim(0.8, 1)
    plt.hlines(0.02, 0.8, 1.05, linestyles="--")
    plt.xlabel(fr"Time since Release / Median Lifetime")
    plt.ylabel("Fraction of Volume Remaining")
    plt.title("100 sccm Airflow")
    plt.legend(handle, ["D = 30.5 nm", "D = 86.1 nm", "D = 121.5 nm"], loc="upper right")
    plt.show()
    handle = []
    for directory, colour in zip(fast_directories, colours):
        files = glob(fr"{directory}\TrimmedDroplets\*.csv")
        results = [pd.read_csv(f"{file}") for file in files]
        starting_radii = np.array([droplet.radius[0] for droplet in results])
        times_array = [droplet.relative_time for droplet in results]
        length_array = np.array([len(times) for times in times_array])
        longest_times = max(times_array, key=len)
        final_time = np.median([times.values[-1] for times in times_array])
        longest = np.max(length_array)
        volume_fractions = np.zeros(longest)
        normalisation_factors = np.array([np.count_nonzero(length_array >= i) for i in range(1, longest + 1)])
        for droplet, starting_radius in zip(results, starting_radii):
            volume_fractions[:len(droplet.relative_time)] += droplet.radius ** 3 / starting_radius ** 3
        mean_volume_fractions = volume_fractions / normalisation_factors
        variances = np.zeros(len(mean_volume_fractions))
        for droplet, starting_radius in zip(results, starting_radii):
            variances[:len(droplet.radius)] += (droplet.radius ** 3 / starting_radius ** 3 - mean_volume_fractions[
                                                                                             :len(droplet.radius)]) ** 2
        variances /= normalisation_factors
        standard_deviations = np.sqrt(variances)
        x = longest_times / final_time
        plt.fill_between(x, mean_volume_fractions - standard_deviations, mean_volume_fractions + standard_deviations,
                         alpha=0.3)
        plt.plot(x, mean_volume_fractions, linewidth=4)
        handle.append(plt.scatter(0, 0, s=4, c=colour))

    plt.ylim(0, 0.2)
    plt.xlim(0.8, 1)
    plt.hlines(0.02, 0.8, 1.05, linestyles="--")
    plt.xlabel(fr"Time since Release / Median Lifetime")
    plt.ylabel("Fraction of Volume Remaining")
    plt.title("200 sccm Airflow")
    plt.legend(handle, ["D = 30.5 nm", "D = 86.1 nm", "D = 121.5 nm"], loc="upper right")
    plt.show()
if __name__ == '__main__':
    test_air_flow_size()
