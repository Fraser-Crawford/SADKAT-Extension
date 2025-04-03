from dataclasses import dataclass

import numpy as np
import pandas as pd
from rust_SADKAT import get_initial_state, y_prime, efflorescence,locking
from scipy.integrate import solve_ivp
from solution_definitions import aqueous_NaCl
from suspension import silica


@dataclass
class RustDroplet:
    solution:str
    suspension:str
    layers:int
    temperature:float
    relative_humidity:float
    air_speed:float
    particle_radius:float
    def starting_state(self,radius:float,solute_concentration:float,particle_concentration:float):
        return np.array(get_initial_state(self.solution,(self.temperature,self.relative_humidity,self.air_speed),
                                 self.suspension,self.particle_radius,radius,solute_concentration,
                                 particle_concentration,self.layers))

    def update_state(self,time, state):
        derivative = y_prime(state,self.solution,(self.temperature,self.relative_humidity,self.air_speed),
                       self.suspension,self.particle_radius,self.layers)
        #print(np.sum(derivative[1+2*self.layers:1+3*self.layers]*np.exp(state[1+2*self.layers:1+3*self.layers])))
        return np.array(derivative)

    def efflorescence(self,state):
        return efflorescence(state,self.solution,(self.temperature,self.relative_humidity,self.air_speed),
                       self.suspension,self.particle_radius,self.layers)

    def locking(self,state,locking_threshold):
        value= locking(state,self.solution,(self.temperature,self.relative_humidity,self.air_speed),
                       self.suspension,self.particle_radius,self.layers,locking_threshold)
        return value

    def equilibrate(self,time,state,threshold):
        dmdt = np.abs(self.update_state(time,state)[0])*np.exp(state[0])
        return dmdt - threshold

    def integrate(self,time:float,radius:float,solute_concentration:float,particle_concentration:float,rtol=1e-6,
                  terminate_on_equilibration=False, equ_threshold=1e-4,
                  terminate_on_efflorescence=False, eff_threshold=0.5,
                  terminate_on_locking=False, locking_threshold=400e-9):
        x0 = self.starting_state(radius,solute_concentration,particle_concentration)
        events = []

        if terminate_on_equilibration:
            m0 = np.exp(x0[0])
            equilibrated = lambda time, x:  self.equilibrate(time, x, equ_threshold*m0)
            equilibrated.terminal = True
            events += [equilibrated]

        if terminate_on_efflorescence:
            efflorescing = lambda time, x: self.efflorescence(x) - eff_threshold
            efflorescing.terminal = True
            events += [efflorescing]

        if terminate_on_locking:
            shell_formation = lambda time, x: self.locking(x,locking_threshold)
            shell_formation.terminal = True
            events += [shell_formation]

        dxdt = lambda time, x: self.update_state(time,x)
        trajectory = solve_ivp(dxdt, (0,time), x0, rtol=rtol, events=events, method="Radau")
        return trajectory

    def complete_trajectory(self, trajectory):
        """Get the trajectory of all variables (including dependent ones) from a simulation (i.e.
        the output of UniformDroplet.integrate).

        Args:
            trajectory: the output of UniformDroplet.integrate, which gives the trajectory of independent
                        variables only.
        Returns:
            A pandas dataframe detailing the complete droplet history.
        """
        labels = ["radius","surface_temperature","solvent_mass","density",
                  "mfs","activity","layer_positions","layer_concentrations",
                  "wet_layer_volumes","layer_mass_solute","true_boundaries",
                  "layer_particle_concentrations","particle_volume_fraction"]
        variables = {key: np.empty(trajectory.t.size, dtype=object) for key in labels}
        for i, state in enumerate(trajectory.y.T):
            earlier_droplet = DataDroplet(state, self.solution,self.suspension,self.particle_radius,self.layers)
            earlier_state = earlier_droplet.complete_state()
            for label, value in earlier_state.items():
                variables[label][i] = value

        variables['time'] = trajectory.t
        return pd.DataFrame(variables)

class DataDroplet:
    def __init__(self, state, solution, suspension, particle_radius, layers):
        state = np.array(state)
        self.solvent_mass = np.exp(state[0])
        self.temperatures = state[1:layers+1]
        self.layer_solute_mass = np.exp(state[layers+1:2*layers+1])
        self.solute_mass = np.sum(self.layer_solute_mass)
        self.layer_particle_mass = np.exp(state[2*layers+1:3*layers+1])
        self.particle_mass = np.sum(self.layer_particle_mass)
        self.layer_positions = state[3*layers+1:4*layers]*1e-6
        match solution:
            case "aqueous_NaCl": self.solution = aqueous_NaCl
        match suspension:
            case "silica": self.suspension = silica(particle_radius)
        self.mfs = self.solute_mass/(self.solvent_mass+self.solute_mass)
        self.density = self.solution.density(self.mfs)
        self.volume = (self.solute_mass+self.solvent_mass)/self.density + self.particle_mass/self.suspension.particle_density
        self.radius = np.cbrt(self.volume*3/(4*np.pi))
        self.true_boundaries = np.concatenate(([0], self.layer_positions, [self.radius]))
        self.layer_volumes = 4 / 3 * np.pi * (self.true_boundaries[1:] ** 3 - self.true_boundaries[:-1] ** 3)
        self.layer_particle_concentration = self.layer_particle_mass/self.layer_volumes
        self.wet_layer_volumes =  self.layer_volumes - self.layer_particle_mass/self.suspension.particle_density
        self.layer_concentrations = self.layer_solute_mass/self.layer_volumes
        self.activity = self.solution.activity(self.mfs)
        self.particle_volume_fraction = self.particle_mass/(self.volume*self.suspension.particle_density)
    def complete_state(self):
        return dict(radius=self.radius,surface_temperature=self.temperatures[-1],
                    solvent_mass=self.solvent_mass,density=self.density,mfs=self.mfs,activity=self.activity,
                    layer_positions=self.layer_positions,layer_concentrations=self.layer_concentrations,
                    wet_layer_volumes=self.wet_layer_volumes,layer_mass_solute=self.layer_solute_mass,
                    true_boundaries=self.true_boundaries,layer_particle_concentrations=self.layer_particle_concentration,particle_volume_fraction=self.particle_volume_fraction)
