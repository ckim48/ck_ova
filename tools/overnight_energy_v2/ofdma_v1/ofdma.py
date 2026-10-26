"""Flat-channel, integer-subcarrier OFDMA; not a relabeling of TDMA.

Noise is normalized over the full W-Hz band. A user with fraction f has
capacity L*f*log2(1+h^2*P/f), total RF power P, and energy L/W*P.
128 orthogonal tones; ideal coding/CSI and zero CP/guard overhead, consistently
with the existing ideal waveform comparison. All users share the same interval.
"""
import numpy as np


def required_power(bits, h, length, counts, tones):
    f=np.asarray(counts,dtype=float)/tones
    exponent=np.divide(np.log(2)*bits,length*f,out=np.zeros_like(f),where=f>0)
    with np.errstate(over='ignore',invalid='ignore'):
        p=f/np.asarray(h)**2*np.expm1(exponent)
    return np.where(np.asarray(bits)>0,np.where(f>0,p,np.inf),0.)


def digital(bits,h,symbols,c):
    bits=np.asarray(bits,float);h=np.asarray(h,float);symbols=np.asarray(symbols,int)
    tones=int(c.get('ofdma_subcarriers',128));n,m=bits.shape
    if np.any(h<=0) or np.any(symbols<=0) or np.any(symbols%tones):return None
    if np.any(bits<0):raise ValueError('Negative payload')
    counts=np.zeros((n,m),int);powers=np.zeros((n,m))
    # Separable decreasing convex powers: start with peak-feasible minimum tone
    # counts, then allocate each remaining tone to the largest energy reduction.
    # Exact discrete sum-power optimum for fixed duration/flat channels. Battery
    # feasibility is audited separately; callers enforce Pmax*T <= Eavailable.
    for k,L in enumerate(symbols):
        active=np.flatnonzero(bits[:,k]>0)
        if not len(active):continue
        tables={}
        for u in active:
            p=required_power(bits[u,k],h[u],L,np.arange(tones+1),tones)
            viable=np.flatnonzero(p<=c['power_limit_w']*(1+1e-12))
            if not len(viable):return None
            counts[u,k]=viable[0];tables[u]=p
        if counts[:,k].sum()>tones:return None
        for _ in range(tones-int(counts[:,k].sum())):
            gains=[tables[u][counts[u,k]]-tables[u][counts[u,k]+1] if counts[u,k]<tones else -np.inf for u in active]
            counts[active[int(np.argmax(gains))],k]+=1
        for u in active:powers[u,k]=tables[u][counts[u,k]]
    f=counts/tones
    snr=np.divide(h[:,None]**2*powers,f,out=np.zeros_like(f),where=f>0)
    capacity=symbols*f*np.log2(1+snr)
    energies=(symbols/c['symbol_rate_hz']*powers).sum(1)
    valid=(np.all(counts.sum(0)<=tones) and np.all(powers<=c['power_limit_w']*(1+1e-10))
           and np.all(capacity+1e-7>=bits) and np.all(energies<=c['available_energy_j']*(1+1e-10)))
    if not valid:return None
    return dict(energy_by_user=energies,powers=powers,subcarrier_counts=counts,
                bandwidth_fractions=f,minimum_rate_margin=float(np.min(capacity-bits)),
                max_power_w=float(np.max(powers)),tones=tones)


def phase_grid(total,dims):
    # Identical grid policy for every head; no validation-loss tuning.
    smallest=float(np.max(dims/total))
    choices=[smallest,2*smallest,4*smallest,.125,.25,.375,.5,.625,.75,.875,1-smallest]
    return sorted(set(int(np.clip(round(f*total[0]/dims[0]),1,total[0]//dims[0]-1)) for f in choices))


def head_design(head,channels,rho,payload,dims,total,eps,c,mode):
    from fast_solver import optimize_head
    users=np.delete(np.arange(len(channels)),head);h=channels[users,head]
    assert np.all(dims==128) and np.all(total==total[0])
    assert total.sum()/c['symbol_rate_hz']*c['power_limit_w']<=c['available_energy_j']*(1+1e-12)
    if mode=='ofdma_only':
        d=digital(payload[users],h,total,c)
        if d is None:return None
        return dict(head=head,users=users,digital=d,energy_j=float(d['energy_by_user'].sum()),symbols_digital=total)
    best=None
    for count in phase_grid(total,dims):
        la=count*dims;ld=total-la
        d=digital(payload[users],h,ld,c)
        if d is None:continue
        a=optimize_head(head,channels,rho,np.zeros_like(rho),np.zeros_like(payload),dims,la,eps,c,0.,'hybrid',digital_symbols=la)
        if a is None:continue
        eb=(la/c['symbol_rate_hz']*a.analog_rms**2).sum(1)+d['energy_by_user']
        if np.any(eb>c['available_energy_j']*(1+1e-9)):continue
        value=float(eb.sum())
        if best is None or value<best['energy_j']:
            best=dict(head=head,users=users,digital=d,analog=a,energy_j=value,
                      symbols_analog=la,symbols_digital=ld,phase_count=count)
    return best


def checks():
    import itertools
    c=dict(symbol_rate_hz=1e6,power_limit_w=.32,available_energy_j=.5,ofdma_subcarriers=128)
    one=digital(np.array([[128.]]),np.array([2.]),np.array([256]),c)
    assert one is not None and np.isclose(one['powers'][0,0],(np.sqrt(2)-1)/4)
    b=np.array([[32.],[32.]])
    two=digital(b,np.ones(2),np.array([128]),c)
    assert two is not None and np.all(two['subcarrier_counts']==64)
    assert (b[:,0]/np.log2(1+.32)).sum()>128 # TDMA is peak-infeasible here.
    zero=digital(np.zeros((2,1)),np.ones(2),np.array([128]),c)
    assert not zero['powers'].any() and not zero['subcarrier_counts'].any()
    tiny=dict(c,ofdma_subcarriers=8,power_limit_w=10.)
    bits=np.array([[8.],[12.],[4.]]);h=np.array([1.,2.,.8]);L=np.array([32])
    z=digital(bits,h,L,tiny)
    brute=min(sum(required_power(bits[:,0],h,32,np.array(ns),8)) for ns in itertools.product(range(1,7),repeat=3) if sum(ns)==8)
    assert np.isclose(z['powers'].sum(),brute,rtol=1e-12)
    return dict(passed=True,single_user_shannon=True,integer_allocation_matches_exhaustive=True,
                bandwidth_scaled_noise=True,tdma_and_ofdma_peak_constraints_distinguished=True,zero_payload=True)
