"""Independent complex-waveform and Monte Carlo checks for the I/Q solver."""
import numpy as np
import manuscript_iq_radio_v1 as radio
from manuscript_iq_study_v1 import ROOT,OUT,write,sha
from pathlib import Path


def main():
    c=dict(block_dimension=128,relative_tolerance=.2,absolute_tolerance_floor=1e-8,
        symbol_rate_hz=1e6,power_limit_w=.32,available_energy_j=.5,
        round_uplink_deadline_s=1.31072,coherence_time_s=.01,max_ao_iterations=8)
    rng=np.random.default_rng(718);l=rng.normal(size=(10,512))*.001
    u=rng.normal(size=l.shape)*.0003
    h=np.full((10,10),3.);np.fill_diagonal(h,0)
    ell=np.array([[1.,2.,3.,4.],[4.,1.,2.,3.]])
    best,audit=radio.design(l,u,h,c,ell);assert best is not None
    ch=h*np.exp(1j*rng.uniform(-np.pi,np.pi,h.shape));head=best['head'];dm=128;n=best['n'];W=c['symbol_rate_hz']
    for block in range(4):
        x=best['x'][:,block];cr=1/np.sqrt(W*x);signal=np.zeros(dm,complex);energies=np.zeros(10)
        sl=slice(block*dm,(block+1)*dm)
        for user in range(10):
            if user==head:continue
            b=np.sqrt(W*x)/ch[user,head]
            waveform=b[0]*l[user,sl]+1j*b[1]*u[user,sl]
            expanded=np.tile(waveform,n)/np.sqrt(n)
            energies[user]=np.vdot(expanded,expanded).real/W
            signal+=ch[user,head]*waveform
        assert np.allclose(energies,best['energy'][:,block],atol=1e-14)
        remote=np.arange(10)!=head
        assert np.allclose(cr[0]*signal.real,l[remote,sl].sum(0),atol=1e-12)
        assert np.allclose(cr[1]*signal.imag,u[remote,sl].sum(0),atol=1e-12)
    # Draw once per trial without accepting/rejecting noise outcomes.
    error=rng.normal(size=(4000,2,4,dm))*np.sqrt(best['variance'][None,:,:,None]/dm)
    observed=np.mean(np.sum(error**2,axis=(2,3)),axis=0)
    assert np.allclose(observed,best['mse'],rtol=.01)
    z=np.zeros_like(l)
    _,zm=radio.analog(z,z,h,c,11,np.ones((1,4)))
    assert zm['energy_bound_j']==zm['uplink_s']==zm['conditional_rmse']==0
    _,lm=radio.analog(l,z,h,c,11,np.ones((1,4)))
    assert lm['component_mse'][1]==0
    result=dict(passed=True,phase_compensation=True,orthonormal_expansion_energy=True,
        separate_IQ_recovery=True,shared_power_verified=True,monte_carlo_trials=4000,
        analytic_MSE=best['mse'].tolist(),observed_MSE=observed.tolist(),zero_and_single_task=True,
        source_sha256={str(Path(p).relative_to(ROOT)):sha(Path(p)) for p in [__file__,radio.__file__]})
    write(OUT/'radio_checks.json',result);print(result)


if __name__=='__main__':main()
