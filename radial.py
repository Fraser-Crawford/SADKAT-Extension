from dataclasses import dataclass

import numpy as np
from numpy import typing as npt
from scipy.integrate import solve_ivp
from typing_extensions import Self
from droplet import Droplet
from suspension_droplet import crossing_rate
from uniform import UniformDroplet
from viscous_solution import ViscousSolution


layer_inertia = 1
stiffness = 100
damping = 2.0*np.sqrt(stiffness*layer_inertia)

@dataclass
class RadialDroplet(Droplet):

    @property
    def volume(self) -> float:
        true_boundaries = np.concatenate(([0],self.cell_boundaries))
        layer_volumes = np.array([4 / 3 * np.pi * (r1 ** 3 - r0 ** 3) for r0, r1 in zip(true_boundaries, true_boundaries[1:])])
        average_layer_concentration = self.layer_mass_solute[:-1]/layer_volumes
        layer_density = self.solution.concentration_to_solute_mass_fraction(average_layer_concentration)
        layer_mass = layer_volumes*layer_density
        layer_solvent_mass = layer_mass-self.layer_mass_solute()[:-1]
        outer_solvent = self.mass_solvent() - layer_solvent_mass
        outer_mfs = self.layer_mass_solute[-1]/(self.layer_mass_solute[-1]+outer_solvent)
        outer_density = self.solution.density(outer_mfs)
        outer_volume = (outer_solvent+self.layer_mass_solute[-1])/outer_density
        return np.sum(layer_volumes)+outer_volume

    @property
    def solute_sherwood_number(self) -> float:
        Sc = self.solution.viscosity(self.mass_fraction_solute,self.temperature) / (
                    self.solution.solvent.density(self.temperature) * self.solution.diffusion(self.mass_fraction_solute,self.temperature))
        Pe = self.peclet
        Re = Pe / Sc
        return 1 + 0.3 * np.sqrt(Re) * np.cbrt(Sc)

    def extra_results(self):
        return dict(
            layer_mass_solute=self.log_mass_solute[:],
            average_layer_concentrations=self.average_layer_concentration,
            layer_concentrations=self.linear_layer_concentrations(),
            layer_boundaries=self.cell_boundaries,
            surface_concentration=self.linear_layer_concentrations()[-1],
            average_diffusion = self.solution.diffusion(self.mass_fraction_solute,self.temperature),
            surface_diffusion = self.solution.diffusion(self.layer_mass_fraction_solute[-1],self.temperature),
            layer_mass_fraction_solute = self.layer_mass_fraction_solute[:],
            peclet = self.peclet,
            predicted_enrichment = self.predicted_enrichment,
            max_enrichment = self.max_enrichment,
            real_enrichment = self.linear_layer_concentrations()[-1]/self.concentration,
            predicted_surface_concentration = self.concentration*self.predicted_enrichment,
            layer_positions =  self.all_positions(),
        )

    solution: ViscousSolution
    total_mass_solvent: float
    cell_boundaries: npt.NDArray[np.float_]
    cell_velocities: npt.NDArray[np.float_]
    log_mass_solute: npt.NDArray[np.float_]

    @property
    def predicted_enrichment(self):
        pe = self.peclet
        return 1+pe/5+pe**2/100-pe**3/4000
    @property
    def max_enrichment(self):
        pe = self.surface_peclet
        return 1 + pe / 5 + pe ** 2 / 100 - pe ** 3 / 4000
    @property
    def surface_diffusion(self):
        return self.solution.diffusion(self.layer_mass_fraction_solute[-1], self.temperature)

    @property
    def average_diffusion(self):
        return self.solution.diffusion(self.mass_fraction_solute,self.temperature)

    def density_derivative(self):
        drho_dt = (self.solution.density(self.mass_fraction_solute+0.01)-self.solution.density(self.mass_fraction_solute-0.01))/0.02
        return drho_dt*(-self.dmdt())*self.mass_fraction_solute**2/self.mass_solute()

    @property
    def peclet(self):
        return (-self.dmdt()-self.density_derivative()*4.0/3.0*np.pi*self.radius**3)/(self.average_diffusion*4*np.pi*self.radius*self.density)

    @property
    def surface_peclet(self):
        return (-self.dmdt()-self.density_derivative()*4.0/3.0*np.pi*self.radius**3)/(self.surface_diffusion*4*np.pi*self.radius*self.density)

    @property
    def layers(self):
        return len(self.log_mass_solute)

    @staticmethod
    def from_mfs(solution, environment, gravity,
                 radius, mass_fraction_solute, temperature, layers=10,
                 velocity=np.zeros(3), position=np.zeros(3),stationary=True):
        """Create a droplet from experimental conditions.

        Args:
            solution: parameters describing solvent+solute
            environment: parameters of gas surrounding droplet
            gravity: body acceleration in metres/second^2 (3-dimensional vector)
            radius: in metres
            mass_fraction_solute: (MFS) (unitless)
            temperature: in K
            velocity: in metres/second (3-dimensional vector)
            position: in metres (3-dimensional vector)
            stationary: bool
        """
        volume = 4 * np.pi / 3 * radius ** 3
        mass = volume * solution.density(mass_fraction_solute)
        mass_solvent = (1 - mass_fraction_solute) * mass
        mass_solute = mass_fraction_solute * mass
        cell_boundaries = radius*np.array([i/layers for i in range(1,layers)])
        concentration = mass_solute/volume
        real_boundaries = np.concatenate(([0],cell_boundaries,[radius]))
        log_mass_solute = np.log(np.array([4/3*np.pi*(r1**3-r0**3)*concentration for r0,r1 in zip(real_boundaries,real_boundaries[1:])]))
        cell_velocities = np.zeros(len(cell_boundaries))
        return RadialDroplet(solution,environment,gravity,stationary,temperature,velocity,position,mass_solvent,cell_boundaries,cell_velocities,log_mass_solute)

    def state(self) -> npt.NDArray[np.float_]:
        return np.hstack((self.cell_boundaries, self.cell_velocities, self.log_mass_solute, self.total_mass_solvent, self.temperature, self.velocity, self.position))

    def split_state(self, state: npt.NDArray[np.float_]):
        cell_boundaries = state[:len(self.cell_boundaries)]
        cell_velocities = state[len(self.cell_boundaries):len(self.cell_velocities)+len(self.cell_boundaries)]
        n = len(self.cell_velocities) + len(self.cell_boundaries) + self.layers
        log_mass_solute = state[len(self.cell_velocities)+len(self.cell_boundaries):n]
        mass_solvent = state[n]
        temp = state[n+1]
        velocity = state[n+2:n+5]
        position = state[n+5:]
        return cell_boundaries, cell_velocities, log_mass_solute, mass_solvent, temp, velocity, position

    def set_state(self, state: npt.NDArray[np.float_]):
        self.cell_boundaries, self.cell_velocities, self.log_mass_solute, self.total_mass_solvent, self.temperature, self.velocity, self.position = self.split_state(state)

    def dxdt(self,time) -> npt.NDArray[np.float_]:
        return np.hstack((self.boundary_correction(),self.boundary_acceleration(),self.change_in_solute_mass(),self.dmdt(),self.dTdt(),self.dvdt(),self.drdt()))

    @property
    def deviation(self):
        return self.cell_boundaries - self.radius*np.array([i/self.layers for i in range(1,self.layers)])

    def boundary_acceleration(self):
        return (-self.cell_velocities*damping-stiffness*self.deviation/self.radius)/layer_inertia

    def boundary_correction(self):
        return self.cell_velocities

    @property
    def layer_volume(self):
        true_boundaries = self.all_positions()
        return np.array([4/3*np.pi*(r1**3-r0**3) for r0,r1 in zip(true_boundaries,true_boundaries[1:])])

    @property
    def average_layer_concentration(self):
        return self.layer_mass_solute/self.layer_volume

    def redistribute(self):
        sign = np.sign(self.cell_velocities)
        volume_corrections = 4*np.pi*(self.cell_boundaries**2*self.cell_velocities)
        concentrations = self.linear_layer_concentrations()[1:]
        result = np.zeros(self.layers)

        for i in range(len(volume_corrections)):
            if sign[i] < 0:
                value = volume_corrections[i]*concentrations[i]
                result[i] += value
                result[i+1] -= value
            else:
                value = volume_corrections[i] * concentrations[i+1]
                result[i] += value
                result[i+1] -= value

        return result

    @property
    def corrected_crossing_rate(self):
        radius = self.radius
        R = self.cell_boundaries / radius
        viscosity_ratio = self.solution.viscosity(self.mass_fraction_solute,self.temperature) / self.environment.dynamic_viscosity
        return crossing_rate(R, radius) * (self.relative_speed / 0.02) * (1 + 1e-3 / 1.81e-5) / (1 + viscosity_ratio)

    @property
    def circulate(self):
        crossing_rates = self.corrected_crossing_rate
        result = np.zeros(self.layers)
        for index, (m0, m1, rate) in enumerate(
                zip(self.layer_mass_solute, self.layer_mass_solute[1:], crossing_rates)):
            value = rate * (m0 - m1)
            result[index] -= value
            result[index + 1] += value
        return result

    @property
    def layer_density(self):
        return self.solution.concentration_to_solute_mass_fraction(self.average_layer_concentration)

    @property
    def layer_mass_solute(self):
        return np.exp(self.log_mass_solute)

    @property
    def layer_mass_fraction_solute(self):
        concentration = self.layer_mass_solute / self.layer_volume
        return self.solution.concentration_to_solute_mass_fraction(concentration)

    def get_gradients(self,normalised_boundaries):
        concentrations = self.linear_layer_concentrations()
        return np.array([(c2-c0)/(r2-r0) for r0,r2,c0,c2 in zip(normalised_boundaries[:-2],normalised_boundaries[2:],concentrations[:-2],concentrations[2:])])

    def refractive_index(self):
        return self.solution.refractive_index(self.mass_fraction_solute,self.temperature)

    def change_in_solute_mass(self):
        radius = self.radius
        redistribute = self.redistribute()
        layer_diffusion= self.solution.diffusion(self.layer_mass_fraction_solute,self.temperature)
        average_diffusion = [(d1+d2)/2 for d1,d2 in zip(layer_diffusion,layer_diffusion[1:])]
        normalised_boundaries = self.all_positions()/radius
        gradients = self.get_gradients(normalised_boundaries)
        diffusion = np.zeros(self.layers)
        circulation = self.circulate
        for i in range(self.layers-1):
            value = 4*np.pi*radius*average_diffusion[i]*gradients[i]*normalised_boundaries[i+1]**2
            diffusion[i] += value
            diffusion[i+1] -= value
        return (redistribute + diffusion + circulation)/self.layer_mass_solute

    def mass_solute(self):
        return np.sum(self.layer_mass_solute)

    def mass_solvent(self) -> float:
        return self.total_mass_solvent

    def linear_layer_concentrations(self):
        rs = np.concatenate((self.cell_boundaries,[self.radius]))
        r0s = rs[:-1]
        r1s = rs[1:]
        r03s = r0s**3
        r04s = r0s**4
        r13s = r1s**3
        r14s = r1s**4
        masses = self.layer_mass_solute
        c0 = 3*masses[0]/(4*np.pi*r03s[0])
        c = [c0,c0]

        for r03,r04,r13,r14,mass,r0,r1 in zip(r03s,r04s,r13s,r14s,masses[1:],r0s,r1s):
            numerator = mass/np.pi+4/3*c[-1]*(r03-r13)
            denominator = r14-r04+4/3*r0*(r03-r13)
            gradient = numerator/denominator
            c.append(gradient*(r1-r0)+c[-1])

        return c

    def all_positions(self):
        return np.concatenate(([0],self.cell_boundaries,[self.radius]))

    def surface_solvent_activity(self) -> float:
        return self.solution.activity(self.solution.concentration_to_solute_mass_fraction(self.linear_layer_concentrations()[-1]))

    def virtual_droplet(self, x) -> Self:
        cell_boundaries, cell_velocities, layer_mass_solute, total_mass_solvent, temperature, velocity, position = self.split_state(x)
        return RadialDroplet(self.solution,self.environment,self.gravity,self.stationary,temperature,velocity,position,total_mass_solvent,cell_boundaries,cell_velocities,layer_mass_solute)

    def convert(self, mass_solvent):
        return UniformDroplet(self.solution, self.environment, self.gravity,self.stationary, self.environment.temperature, self.velocity,
                       self.position, mass_solvent, self.mass_solute())

    def solver(self, dxdt, time_range, first_step, rtol, events):
        unstable = lambda time, x: self.virtual_droplet(x).radius - self.virtual_droplet(x).cell_boundaries[-1]
        unstable.terminal = True
        events.append(unstable)
        return solve_ivp(dxdt, time_range, self.state(), first_step=first_step, rtol=rtol, events=events, method="Radau")