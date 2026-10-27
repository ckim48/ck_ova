"""Fixed-route ports of ISCC-FL P2 and UAV-AFL normalized OTA transport.

No Proposed head enumeration, guard or OVA receiver is called here. Channels are
amplitudes normalized by full-band noise; ideal coding/CSI matches the simulator.
"""
import math
import numpy as np
from scipy.optimize import brentq


def allocate_fdma(gains, compute_s, bandwidth, bits, power):
    """Solve ISCC Eq.(24)/P2 for q_s=1, fixed batch/positions by bisection.

    x is a bandwidth fraction. R=B*x*log2(1+P*h^2/x); h already includes
    noise in the WHOLE B band. Each worker uploads after its compute completes.
    """
    h=np.asarray(gains,float);c=np.asarray(compute_s,float)
    if not len(h):return np.array([]),np.array([]),float(c.max(initial=0))
    a=power*h*h
    def rate(x):return bandwidth*x*np.log1p(a/x)/np.log(2.)
    equal=np.full(len(h),1/len(h));upper=float(np.max(c+bits/rate(equal)))
    lower=float(np.max(c+bits/rate(np.ones(len(h)))))
    def fractions(t):
        needed=bits/(t-c)
        out=[]
        for snr,r in zip(a,needed):
            fn=lambda x:bandwidth*x*np.log1p(snr/x)/np.log(2.)-r
            if fn(1.) < -1e-7:return None
            out.append(brentq(fn,1e-14,1.,xtol=1e-13))
        return np.array(out)
    for _ in range(55):
        middle=(lower+upper)/2;share=fractions(middle)
        if share is not None and share.sum()<=1:upper=middle
        else:lower=middle
    share=fractions(upper*(1+1e-12));share/=share.sum()
    seconds=bits/rate(share)
    assert abs(share.sum()-1)<1e-9 and np.all(share>0)
    return share,seconds,float(np.max(c+seconds))


def metadata(vector,head,users,power,seconds,c,mode):
    power=np.asarray(power,float);seconds=np.asarray(seconds,float)
    energy=power*seconds;uplink=float(seconds.max(initial=0))
    eps=max(c['absolute_tolerance_floor'],c['relative_tolerance']*float(np.linalg.norm(vector)))
    return dict(status='feasible',physical_mode=mode,head=int(head),active_users=list(map(int,users)),
        uplink_s=uplink,energy_bound_j=float(energy.sum()),expected_energy_j=float(energy.sum()),
        max_power_w=float(power.max(initial=0)),max_user_energy_j=float(energy.max(initial=0)),
        user_power_w=power.tolist(),user_uplink_s=seconds.tolist(),conditional_rmse=0.,
        realized_error_to_raw=0.,epsilon=eps,broadcast_required=bool(np.any(vector)),
        network_cap_overrun=bool(energy.sum()>.1*(1+1e-6)),
        round_deadline_overrun=bool(uplink>c['round_uplink_deadline_s']*(1+1e-6)),
        per_update_energy_overrun=bool(energy.max(initial=0)>c['available_energy_j']*(1+1e-6)),
        budget_policy='Complete full workloads; power is hard. Network/update energy and round deadline are reported QoS violations, not hidden or discarded.',
        circuit_policy='Common conservative 9TX+1RX during uplink, full 9-peer broadcast accounting even when only selected peers consume the model.')


def digital(updates,channels,c,head,compute_s):
    raw=updates.sum(0);users=np.flatnonzero(np.any(updates,axis=1)&(np.arange(len(updates))!=head))
    power=c['power_limit_w'];bits=32*updates.shape[1]+64
    shares,seconds,finish=allocate_fdma(channels[users,head],np.asarray(compute_s)[users],c['symbol_rate_hz'],bits,power)
    m=metadata(raw,head,users,np.full(len(users),power),seconds,c,'ISCC_P2_FP32_FDMA')
    m.update(bandwidth_fractions=shares.tolist(),payload_bits_per_user=bits,
        compute_upload_finish_s=max(finish,float(np.max(compute_s))),
        reference='ISCC-FL Eq.(4), Eq.(8), P2/Eq.(24); q_s=1 and fixed locations/batches',
        waveform='FP32 digital payload, ideal capacity-achieving coding; no extra Q6 compression.')
    return raw,m


def ota(updates,channels,c,head,seed):
    """UAV-AFL Eq.(11)-(16), unbiased channel inversion, coherent repetition.

    Each row already includes its equal-client weight and step size. The
    variance-derived coefficients absorb that weight. Noise is drawn ONCE after
    analytical repetition selection, never accepted/rejected by realization.
    """
    raw=updates.sum(0);users=np.flatnonzero(np.any(updates,axis=1)&(np.arange(len(updates))!=head))
    if not len(users):return raw,metadata(raw,head,users,[],[],c,'UAV_AFL_OTA_channel_inversion')
    d=updates.shape[1];assert d%2==0
    x=updates[users];mu=x.mean(1);sigma=x.std(1);h=np.abs(channels[users,head])
    zeta=float(np.max(sigma/(h*np.sqrt(c['power_limit_w']/2))))
    eps=max(c['absolute_tolerance_floor'],c['relative_tolerance']*float(np.linalg.norm(raw)))
    # Complex CN(0,1), so each real coordinate has variance1/2.
    repeats=max(1,int(math.ceil(d*zeta*zeta/(2*eps*eps))))
    coefficient=np.divide(sigma,zeta*h,out=np.zeros_like(sigma),where=zeta*h>0)
    powers=2*coefficient**2;seconds=np.full(len(users),repeats*(d/2)/c['symbol_rate_hz'])
    rng=np.random.default_rng(seed);noise=rng.normal(0.,zeta/math.sqrt(2*repeats),d)
    received=raw+noise
    m=metadata(raw,head,users,powers,seconds,c,'UAV_AFL_OTA_channel_inversion')
    m.update(conditional_rmse=math.sqrt(d/2/repeats)*zeta,realized_error_to_raw=float(np.linalg.norm(noise)),
        repetitions=repeats,receiver_scale=zeta,normalization_means=mu.tolist(),normalization_std=sigma.tolist(),
        metadata_bits=64*len(users),metadata_convention='Error-free mean/variance control side-channel as original paper; airtime/energy not modeled.',
        reference='UAV-AFL Eq.(11)-(16), Alg.1; fixed-route all-ready scheduler and channel inversion replace joint Alg.2 optimization.',
        symbols_per_repetition=d//2)
    assert m['conditional_rmse']<=eps*(1+1e-9)
    assert m['max_power_w']<=c['power_limit_w']*(1+1e-9)
    return received,m


def unit():
    h=np.array([1.,2.,3.]);c=np.array([.2,.01,.08]);B=1e6;bits=655424;p=.32
    share,seconds,t=allocate_fdma(h,c,B,bits,p)
    equal=c+bits/(B/3*np.log2(1+p*h*h*3))
    assert t<=equal.max()+1e-8 and np.ptp(c+seconds)<1e-7
    cfg=dict(symbol_rate_hz=B,power_limit_w=p,relative_tolerance=.2,absolute_tolerance_floor=1e-8,round_uplink_deadline_s=1.31072,available_energy_j=.5)
    x=np.random.default_rng(3).normal(size=(4,128))*1e-3;channels=np.full((4,4),3.)
    _,m=ota(x,channels,cfg,0,1);assert m['conditional_rmse']<=m['epsilon']
    # Huge-noise draw is retained; analytical feasibility never retries noise.
    outputs=[ota(x,channels,cfg,0,i)[0]-x.sum(0) for i in range(300)]
    assert abs(np.mean(np.sum(np.array(outputs)**2,axis=1))/m['conditional_rmse']**2-1)<.08
    return dict(passed=True,FDMA_equalized_finish_and_no_worse_than_equal_bandwidth=True,OTA_noise_variance_and_power=True)
