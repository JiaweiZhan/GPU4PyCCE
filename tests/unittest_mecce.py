import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pycce as pc
import unittest
pc.utilities.config_torch()

class GPU4PyCCETestCase(unittest.TestCase):
    def test_mecce_gpu(self):
        #bath spin array
        electrons = pc.random_bath('e', [5e3, 5e3, 5e3], density=1e16,
                                density_units='cm-3', seed=2)  # Density in 1/cm^3
        electrons.types.add_type('e', 1.0, pc.ci['e'].gyro /2)

        #Center array
        center = pc.CenterArray(spin=1/2, alpha=[1,0], beta=[0,1])

        #define simulator object with linbladian map
        calc = pc.Simulator(center, bath=electrons, order=2, r_bath=1.4e3, r_dipole=0.7e3,
                            pulses=1, magnetic_field=300, n_clusters=None, verbose=True)
        calc.bath.linmap = pc.LindbladianMap()


        # Define spin states
        plus_1 = np.array([1, 0, 0], dtype=complex)
        zero = np.array([0, 1, 0], dtype=complex)
        minus_1 = np.array([0, 0, 1], dtype=complex)

        #Spin-Matrices 
        p0_1 =np.outer(zero, plus_1) 
        m1_0=np.outer(plus_1, zero) 
        p0__1=np.outer(zero, minus_1) 
        m0__1=np.outer(minus_1, zero) 


        #Strength of pair-interaction
        def pos(coord_1, coord_2, r_h=200):
            pos = (np.abs(coord_1) - np.abs(coord_2))
            r = np.linalg.norm(pos)
            return r/r_h

        #Define the dissipator rates with four different dissipators  
        et1 = 0.5  # in ms
        decay_rate = 1 / et1 / 2  # in rad / ms
        for k in range(len(calc.clusters[1])):
                i, j = calc.clusters[2][k]
                n_1 = calc.bath[i]
                n_2 = calc.bath[j]
                hop_rate = decay_rate*np.exp(-np.abs(pos(n_1['xyz'], n_2['xyz'], )))
                #dis = pc.Dissipator(size=2,  left=p0_1, right=m1_0, symmetric=False)
                dis1 = pc.Dissipator(size=2,  symmetric=True)
                dis2 = pc.Dissipator(size=2,  symmetric=True)
                dis3 = pc.Dissipator(size=2,  symmetric=True)
                dis4 = pc.Dissipator(size=2,  symmetric=True)
                dis1.add_jump_operator(p0_1, 0, rate=hop_rate)
                dis1.add_jump_operator(m1_0, 1, rate=hop_rate)
                dis2.add_jump_operator(m1_0, 0, rate=hop_rate)
                dis2.add_jump_operator(p0_1, 1, rate=hop_rate)
                dis3.add_jump_operator(p0__1, 0, rate=hop_rate)
                dis3.add_jump_operator(m0__1, 1, rate=hop_rate)
                dis4.add_jump_operator(m0__1, 0, rate=hop_rate)
                dis4.add_jump_operator(p0__1, 1, rate=hop_rate)
                calc.bath.linmap.add_dissipator(k, dis1)
                calc.bath.linmap.add_dissipator(k, dis2)
                calc.bath.linmap.add_dissipator(k, dis3)
                calc.bath.linmap.add_dissipator(k, dis4)

        #Computation of the coherence        
        ts = np.linspace(0, 2, 51)
        calc.order = 1
        calc.to(f'cuda')

        lmecce = calc.compute(ts, method='mecce', parallel=True)

        lmecce_ref = np.load("./mecce_ref.npy")
        self.assertTrue(np.allclose(lmecce, lmecce_ref))

if __name__ == '__main__':
    unittest.main()
