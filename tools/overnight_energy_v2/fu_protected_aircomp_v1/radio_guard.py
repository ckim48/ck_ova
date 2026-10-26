"""FU-relative total-noise protection without changing the desired FL+FU sum."""
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tools/overnight_energy_v2/streaming_requests_v1'))
import radio_stream as original
base=original.base

def budget(l,u,c):
    nl,nu=original.norm_bound(l,c),original.norm_bound(u,c)
    old=max(c['absolute_tolerance_floor'],c['relative_tolerance']*np.hypot(nl,nu))
    target=min(old,max(c['absolute_tolerance_floor'],c['relative_tolerance']*nu)) if np.any(u) else old
    return nl,nu,float(old),float(target)

def analog(l,u,channels,c,seed):
    nl,nu,old,target=budget(l,u,c)
    if not np.any(u) or len(np.flatnonzero(np.any(l+u,axis=1)))<=1:
        vector,meta=original.analog(l,u,channels,c,seed)
    else:
        ll,uu,denom=(l,u,nl) if np.any(l) else (u,np.zeros_like(u),nu)
        vector,meta=base.communicate(ll,uu,channels,dict(c,relative_tolerance=target/denom),'joint',seed)
    meta.update(FL_norm_bound=nl,FU_norm_bound=nu,original_epsilon=old,protected_epsilon=target,
        fu_noise_guard_active=bool(np.any(u)),
        numerical_floor_active=bool(np.any(u) and c['relative_tolerance']*nu<c['absolute_tolerance_floor']),
        component_budget_rule='FU active: total RMS <= min(original joint RMS budget, max(numerical floor, original relative coefficient * FU block-norm bound)); otherwise unchanged',
        protection_scope='Total FL+FU reconstruction noise relative to an available FU norm bound; not independent FU reconstruction, per-request accuracy, or deletion certification')
    return vector,meta
