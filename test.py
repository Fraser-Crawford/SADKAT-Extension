import unittest

import numpy as np
import inspect
from environment import Atmosphere
from radial import RadialDroplet
from solution_definitions import aqueous_NaCl
from uniform import UniformDroplet
from viscous_defintions import viscous_aqueous_NaCl
import matplotlib.pyplot as plt

class UnifromCase(unittest.TestCase):
    def setUp(self):
        self.droplet = UniformDroplet.from_mfs(aqueous_NaCl,Atmosphere(293),np.array([0,0,0]),50e-6,0.1,293)

    def test_copy(self):
        self.assertEqual(self.droplet.state().all(),self.droplet.copy().state().all(),"Copied droplet is not the same as original droplet")

    def test_complete_state(self):
        self.assertEqual(self.droplet.float_mass_solvent, self.droplet.complete_state["mass_solvent"])

    def test_integrate(self):
        trajectory = self.droplet.integrate(1)
        states = self.droplet.complete_trajectory(trajectory)

class RadialCase(unittest.TestCase):
    def setUp(self):
        self.droplet = RadialDroplet.from_mfs(viscous_aqueous_NaCl,Atmosphere(293),np.array([0,0,0]),50e-6,0.1,293,layers=10)

    def test_copy(self):
        self.assertEqual(self.droplet.state().all(),self.droplet.copy().state().all(),"Copied droplet is not the same as original droplet")

    def test_complete_state(self):
        self.assertEqual(self.droplet.float_mass_solvent, self.droplet.complete_state["mass_solvent"])

    def test_integrate(self):
        trajectory = self.droplet.integrate(1)
        states = self.droplet.complete_trajectory(trajectory)

if __name__ == '__main__':
    unittest.main()

