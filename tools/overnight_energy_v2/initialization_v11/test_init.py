"""Independent constraint checks for the isolated joint-power solver."""
import unittest

import numpy as np

import wireless_init_v11 as joint


class JointPowerTests(unittest.TestCase):
    def check_design(self, mode, chi):
        channels = np.array([[0., 3., 6.], [3., 0., 5.], [6., 5., 0.]])
        rho = np.array([[.002, .004], [.004, .003], [.003, .002]])
        delta = np.full((3, 2), 1e-5)
        payload = np.array([[0, 0], [32, 32], [32, 32]])
        if mode == 'mixed_aircomp':
            payload[:] = 0; delta[:] = 0
        dims = np.array([4, 4]); symbols = np.array([32, 32])
        config = {'power_limit_w': .32, 'available_energy_j': .1,
                  'base_overhead_j': 1e-5, 'head_overhead_j': 5e-5,
                  'symbol_rate_hz': 1e6, 'round_uplink_deadline_s': .2,
                  'coherence_time_s': .01, 'max_ao_iterations': 5,
                  'receive_initializations': 4}
        epsilon = .004
        design = joint.optimize_head(0, channels, rho, delta, payload, dims,
                                     symbols, epsilon, config, chi, mode)
        self.assertIsNotNone(design, (mode, chi))
        a, d, s = design.analog_rms, design.digital_sqrt_power, design.receiver
        h = channels[design.users, 0, None]
        for block in range(2):
            active = np.flatnonzero(payload[design.users, block] > 0)
            active = active[np.argsort(-h[active, 0], kind='stable')]
            for rank, user in enumerate(active):
                interference = 0. if mode == 'orthogonal' else float((h[:, 0] @ a[:, block])**2)
                interference += sum(float((h[other, 0]*d[other, block])**2)
                    * (chi if i < rank else 1.) for i, other in enumerate(active) if other != user)
                sinr = float((h[user, 0]*d[user, block])**2 / (1 + interference))
                required = np.expm1(np.log(2)*payload[design.users[user], block]/symbols[block])
                self.assertGreaterEqual(sinr*(1+1e-6), required)
        duration = symbols/config['symbol_rate_hz']
        if mode == 'orthogonal':
            energy = duration*(a*a+d*d)
            peak = np.maximum(a*a,d*d)
        else:
            energy = duration*(a+d)**2
            peak = (a+d)**2
        self.assertLessEqual(float(peak.max()), config['power_limit_w']*(1+1e-6))
        self.assertTrue(np.all(energy.sum(axis=1)+config['base_overhead_j'] <= .1*(1+1e-6)))
        overhead = 3*config['base_overhead_j']+config['head_overhead_j']
        self.assertAlmostEqual(design.energy_j, float(energy.sum()+overhead), places=12)
        residual = 0 if mode == 'orthogonal' else chi
        bounds = np.abs(s*h*a-rho[design.users]).sum(axis=0)
        bounds += s*(np.sqrt(dims/(2*symbols))+(np.sqrt(residual)*h*d).sum(axis=0))
        bounds += delta[design.users].sum(axis=0)
        self.assertLessEqual(float(np.linalg.norm(bounds)), epsilon*(1+1e-5))
        self.assertAlmostEqual(design.error_bound, float(np.linalg.norm(bounds)), places=12)
        history = np.array(design.history)
        self.assertTrue(np.all(np.diff(history) <= np.maximum(1e-12, history[:-1]*1e-6)))

    def test_hybrid_with_ideal_sic(self):
        self.check_design('hybrid', 0.)

    def test_hybrid_with_residual_sic(self):
        self.check_design('hybrid', .001)

    def test_phase_and_mixed(self):
        self.check_design('orthogonal', .001)
        self.check_design('mixed_aircomp', 0.)

    def test_parent_incumbent_is_not_lost(self):
        import importlib.util
        from pathlib import Path
        import sys
        parent_path=Path(__file__).parent.parent/'joint_power_v3/wireless_joint_v3.py'
        spec=importlib.util.spec_from_file_location('frozen_parent_v3',parent_path)
        parent=importlib.util.module_from_spec(spec);sys.modules[spec.name]=parent;spec.loader.exec_module(parent)
        channels=np.array([[0.,3.,6.],[3.,0.,5.],[6.,5.,0.]])
        rho=np.array([[.002,.004],[.004,.003],[.003,.002]])
        delta=np.full((3,2),1e-5);payload=np.array([[0,0],[32,32],[32,32]])
        config=dict(power_limit_w=.32,available_energy_j=.1,base_overhead_j=1e-5,
                    head_overhead_j=5e-5,symbol_rate_hz=1e6,round_uplink_deadline_s=.2,
                    coherence_time_s=.01,max_ao_iterations=5,receive_initializations=4)
        for mode in ['hybrid','orthogonal']:
            args=(0,channels,rho,delta,payload,np.array([4,4]),np.array([32,32]),.004,config,.001,mode)
            old=parent.optimize_head(*args);new=joint.optimize_head(*args)
            self.assertIsNotNone(old);self.assertIsNotNone(new)
            self.assertLessEqual(new.energy_j,old.energy_j*(1+1e-6))


if __name__ == '__main__':
    unittest.main()
