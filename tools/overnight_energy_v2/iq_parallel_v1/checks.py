"""CPU mathematical/physical checks; no dataset/test/reference selection."""
import json
import numpy as np
import radio_iq as radio

def self_check():
    rng=np.random.default_rng(231026);L=rng.normal(size=(3,8));U=rng.normal(size=(3,8))
    h=rng.uniform(.8,2.,3)*np.exp(1j*rng.uniform(-3,3,3));cl=.7;cu=1.1
    tx=(L/cl+1j*U/cu)/h[:,None];rx=(h[:,None]*tx).sum(0)
    assert np.allclose(cl*rx.real,L.sum(0)) and np.allclose(cu*rx.imag,U.sum(0))
    e=np.square(np.abs(tx)).sum(1)
    assert np.allclose(e,((L/cl)**2+(U/cu)**2).sum(1)/np.abs(h)**2)
    raw=(L+U).sum(0);assert np.allclose(cl*rx.real+cu*rx.imag,raw)
    packed=(L+U)[:,::2]+1j*(L+U)[:,1::2]
    assert packed.shape[1]*2==L.shape[1]
    assert np.allclose((np.column_stack([packed.sum(0).real,packed.sum(0).imag])).ravel(),raw)
    v,meta=radio.controller(np.array([1.,2.]),np.array([-2.,0.]),True)
    assert np.allclose(v,[-2.,2.]) and abs(meta['adjusted_FL_dot_FU'])<1e-12
    v,_=radio.controller(np.array([1.,2.]),np.zeros(2),True);assert np.allclose(v,[1,2])
    n=(rng.normal(size=(30000,8))+1j*rng.normal(size=(30000,8)))/np.sqrt(2)
    empirical=float(np.square(cl*n.real+cu*n.imag).sum(1).mean())
    expected=8*(cl**2+cu**2)/2;assert abs(empirical/expected-1)<.02
    cfg=dict(block_dimension=8,symbol_rate_hz=1e6,relative_tolerance=.2,absolute_tolerance_floor=1e-8,
             power_limit_w=.32,available_energy_j=.5,coherence_time_s=.01,round_uplink_deadline_s=1.31072)
    channels=np.full((3,3),100.);np.fill_diagonal(channels,0.)
    checks={}
    for mode,project in [('iq',False),('iq',True),('packed_sum',False),('separate',True)]:
        value,m=radio.communicate(L,U,channels,cfg,mode,project,891)
        assert value is not None and m['max_power_w']<=.32*(1+3e-6)
        assert m['conditional_rmse']<=m['epsilon']*(1+3e-5)
        checks[mode+str(project)]={k:m[k] for k in ['uplink_s','energy_bound_j','conditional_rmse','epsilon','max_power_w']}
    # Invalid channel is reported without an invented feasible zero-cost result.
    invalid=radio._solve(0,np.zeros((3,3)),[np.linalg.norm(L[:,None,:],axis=-1)],np.array([8]),[np.array([4])],[.1],cfg,False)
    assert invalid is None
    return dict(passed=True,IQ_recovers_two_real_sums=True,RF_energy_adds_I_and_Q=True,
        packed_sum_uses_both_quadratures=True,ideal_plain_sum_equivalence=True,
        projection_is_first_order_only=True,noise_mc_relative_error=empirical/expected-1,radio=checks)

def fixed_update(l,u,h,c):
    results={};designs={}
    for label,mode,project in [('iq_sum','iq',False),('iq_project','iq',True),
                               ('packed_sum','packed_sum',False),('separate_project','separate',True)]:
        vector,meta,design=radio.communicate(l,u,h,c,mode,project,600000,True)
        assert vector is not None,(label,meta)
        assert meta['conditional_rmse']<=meta['epsilon']*(1+3e-5)+1e-10
        assert meta['energy_bound_j']<=.05*(1+3e-6)
        results[label]=meta;designs[label]=design
    a,b=results['iq_sum'],results['iq_project']
    assert a['energy_bound_j']==b['energy_bound_j'] and a['head']==b['head']
    # Plain I/Q and controlled I/Q have IDENTICAL PHY design on a fixed update.
    # Phase stress diagnostic is illustrative, not an optimized or learning run.
    z=designs['iq_sum'];users=z['users'];head=z['head'];cl,cu=z['c']
    lb,ub=radio._blocks(l,c),radio._blocks(u,c);stress=[]
    for deg in [0,1,3,5]:
        errors=[]
        for seed in range(12):
            delta=np.random.default_rng(700000+seed).normal(0,np.deg2rad(deg),(len(users),lb.shape[1],1))
            valid=(cl>0)&(cu>0)
            ratio=np.divide(cl,cu,out=np.zeros_like(cl),where=cu>0)[None,:,None]
            inv=np.divide(cu,cl,out=np.zeros_like(cu),where=cl>0)[None,:,None]
            a=lb[head]+(np.cos(delta)*lb[users]-ratio*np.sin(delta)*ub[users]).sum(0)
            b=ub[head]+(inv*np.sin(delta)*lb[users]+np.cos(delta)*ub[users]).sum(0)
            errors.append(float(np.linalg.norm((a+b-(lb+ub).sum(0))[:, :]))**2)
        stress.append(dict(phase_std_degrees=deg,raw_sum_bias_rms=float(np.sqrt(np.mean(errors)))))
    return dict(passed=True,scope='Actual full-dimensional training update; no final-reference/test selection.',
        fixed_transport=results,phase_only_sensitivity=stress,
        phase_scope='Frozen perfect-CSI design, independent Gaussian residual phase per user/block; excludes additive noise. No robust guarantee.')

if __name__=='__main__':print(json.dumps(self_check(),indent=2))
