"""Use the original 0.1 J design first; allow logged network-cap overruns.

Peak power, individual device energy, deadline, coherence and quantization
error constraints stay unchanged. The fallback uses the largest permitted
frame count and the minimum-energy head at that duration.
"""
from types import SimpleNamespace
import json
import numpy as np

ORIGINAL_CAP_J=.1

def build(base):
    original=base.base.parent.digital
    cache={}
    raw=original.base.old.ofdma.digital
    def cached(bits,h,symbols,config):
        arrays=[np.ascontiguousarray(v) for v in [bits,h,symbols]]
        key=tuple((a.dtype.str,a.shape,a.tobytes()) for a in arrays)+(json.dumps(config,sort_keys=True),)
        if key not in cache:cache[key]=raw(bits,h,symbols,config)
        return cache[key]
    ofdma=SimpleNamespace(**original.base.old.ofdma.__dict__);ofdma.digital=cached
    old=SimpleNamespace(**original.base.old.__dict__);old.ofdma=ofdma
    limited=SimpleNamespace(**original.base.__dict__);limited.old=old;limited.CAP=ORIGINAL_CAP_J
    normal=base.repeat.clone_function(original.analog,[],{'base':limited})

    class SequentialRadio:
        @staticmethod
        def analog(l,u,channels,config,seed):
            assert not (np.any(l) and np.any(u))
            phase='FU' if np.any(u) else 'FL'
            vector,meta=normal(l,u,channels,config,seed)
            failure=None
            if vector is None:
                failure=dict(meta)
                relaxed=SimpleNamespace(**limited.__dict__)
                # This bound is implied by the retained per-user limit; it is
                # not a replacement experimental network-energy budget.
                relaxed.CAP=(len(channels)-1)*config['available_energy_j']
                relaxed.COUNTS=tuple(reversed(limited.COUNTS))
                fallback=base.repeat.clone_function(original.analog,[],{'base':relaxed})
                vector,meta=fallback(l,u,channels,config,seed)
            energy=meta.get('energy_bound_j')
            excess=None if energy is None else max(0.,energy-ORIGINAL_CAP_J)
            meta.update(physical_mode='digital_sequential_ofdma',phase=phase,
                        packet='Q6('+phase+' only); one payload per transmitting UAV',FL_FU_temporally_overlapped=False,
                        network_tx_cap_j=None,original_network_tx_cap_j=ORIGINAL_CAP_J,
                        network_cap_is_hard_constraint=False,network_cap_excess_j=excess,
                        original_network_cap_satisfied=None if energy is None else energy<=ORIGINAL_CAP_J*(1+2e-6),
                        overrun_fallback_used=failure is not None,
                        energy_policy='Original cap-feasible minimum-delay design preferred; if unavailable, longest permitted frame and minimum-energy head, with original per-user power/energy and deadline limits.',
                        original_cap_attempt=failure)
            if vector is not None:
                assert meta['max_power_w']<=config['power_limit_w']*(1+2e-6)
                assert meta.get('max_user_energy_j',0)<=config['available_energy_j']*(1+2e-6)
                assert meta['uplink_s']<=config['round_uplink_deadline_s']+1e-10
                assert meta['conditional_rmse']<=meta['epsilon']*(1+3e-5)+1e-9
            return vector,meta
    return SequentialRadio
