"""Centroid head and uniform receive-noise allocation; same protected target."""
import numpy as np
def pool_init(cap):
    import radio_parallel_cap
    radio_parallel_cap.CAP=cap

def build(base,geometry,channels,cap):
    common=base.base.parent.common;guard=base.base.parent.guard
    heads=[int(np.argmin(((np.asarray(s['positions_xy_m'])-np.mean(s['positions_xy_m'],axis=0))**2).sum(1))) for s in geometry['stops']]
    class Heuristic:
        @staticmethod
        def analog(l,u,ch,c,seed):
            if len(np.flatnonzero(np.any(l+u,axis=1)))<=1:return guard.analog(l,u,ch,c,seed)
            matches=[i for i,h in enumerate(channels) if np.array_equal(h,ch)];assert len(matches)==1
            head=heads[matches[0]];users=np.delete(np.arange(len(l)),head)
            d=c['block_dimension'];v=l+u
            padded=np.pad(v,((0,0),(0,(-v.shape[1])%d))).reshape(len(v),-1,d)
            dims=np.minimum(d,v.shape[1]-np.arange(padded.shape[1])*d)
            rho=np.linalg.norm(padded,axis=-1);coeff=rho[users]**2/ch[users,head,None]**2
            active=np.any(coeff>0,axis=0);nl,nu,old,eps=guard.budget(l,u,c);w=c['symbol_rate_hz']
            x=float(dims[active].sum()/(2*w*eps**2)*(1+1e-8))
            eu=coeff.sum(1)*x;energy=float(eu.sum());selected=None
            if energy<=cap*(1+1e-9) and eu.max()<=c['available_energy_j']*(1+1e-9):
                for n in common.base.COUNTS:
                    peak=float((coeff*x*w/(n*dims)).max())
                    if peak<=c['power_limit_w']*(1+1e-9):selected=(n,peak);break
            m=dict(status='no_feasible_design_found',head=head,head_candidates=1,epsilon=eps,
                protected_epsilon=eps,original_epsilon=old,FL_norm_bound=nl,FU_norm_bound=nu,
                network_tx_cap_j=cap,heuristic='Nearest formation-centroid head; uniform inverse receive-noise scale across nonzero blocks; first feasible common frame count; no optimized fallback.')
            if selected is None:return None,m
            n,peak=selected;raw=padded.sum(0);received=raw.copy();rng=np.random.default_rng(seed)
            sd=1/np.sqrt(2*w*x)
            for j in range(len(dims)):
                received[j]+=rng.normal(0,sd if active[j] else 0.,d)
            rms=float(np.sqrt(dims[active].sum()*sd**2))
            m.update(status='feasible',physical_mode='heuristic_joint',energy_bound_j=energy,expected_energy_j=energy,
                uplink_s=float(n*dims.sum()/w),frame_count=n,max_user_energy_j=float(eu.max()),max_power_w=peak,
                conditional_rmse=rms,error_bound=rms,quantization_error_bound=0.,fu_noise_guard_active=bool(np.any(u)),
                realized_error_to_raw=float(np.linalg.norm(received-raw)))
            assert rms<=eps*(1+1e-6)
            return received.reshape(-1)[:v.shape[1]],m
    return Heuristic
