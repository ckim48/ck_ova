"""Independent complex-waveform check of the mapped Combined construction."""
from pathlib import Path
import numpy as np
import combined_iq_diagnostic_v1 as study


def main():
    rng=np.random.default_rng(673104);rows=[]
    for path in sorted((study.OUT/'snapshots').glob('*/*.npz')):
        z=np.load(path);info=study.read(path.with_suffix('.json'));c=info['config']
        l,u,ell=z['l'],z['u'],z['ell'];W=c['symbol_rate_hz'];dm=c['block_dimension']
        h=z['channels']*np.exp(1j*rng.uniform(-np.pi,np.pi,z['channels'].shape))
        best,_=study.transport('iq',1.).design(l,u,h,c,ell);assert best is not None
        head=best['head'];m=l.shape[1]//dm;n=best['n']
        l=l.reshape(10,m,dm);u=u.reshape(10,m,dm)
        va=best['variance'];ca=np.sqrt(2*va/dm);cc=np.sqrt(2*va.sum(0)/dm)
        assert np.all(cc>0)
        iq_received=np.zeros((m,dm),complex);combined_received=np.zeros_like(iq_received)
        iq_energy=np.zeros((10,m));combined_energy=np.zeros_like(iq_energy)
        for user in range(10):
            if user==head:continue
            bl=np.divide(1.,ca[0]*h[user,head],out=np.zeros(m,complex),where=ca[0]>0)
            bu=np.divide(1.,ca[1]*h[user,head],out=np.zeros(m,complex),where=ca[1]>0)
            bc=1/(cc*h[user,head])
            wi=bl[:,None]*l[user]+1j*bu[:,None]*u[user]
            wc=bc[:,None]*(l[user]+u[user])
            iq_received+=h[user,head]*wi;combined_received+=h[user,head]*wc
            # An orthonormal n-fold expansion preserves this waveform energy.
            iq_energy[user]=np.sum(np.abs(wi)**2,axis=1)/W
            combined_energy[user]=np.sum(np.abs(wc)**2,axis=1)/W
        assert np.allclose(iq_energy,best['energy'],rtol=1e-10,atol=1e-18)
        assert np.all(combined_energy<=iq_energy*(1+1e-10)+1e-18)
        ni=(rng.normal(size=(m,dm))+1j*rng.normal(size=(m,dm)))/np.sqrt(2)
        # The real component has variance1/2. Its independent imaginary part
        # makes this a valid proper complex Gaussian noise for one real stream.
        nc_real=(ca[0,:,None]*ni.real+ca[1,:,None]*ni.imag)/cc[:,None]
        nc=nc_real+1j*rng.normal(size=(m,dm))/np.sqrt(2)
        received_iq=ca[0,:,None]*(iq_received+ni).real+ca[1,:,None]*(iq_received+ni).imag+l[head]+u[head]
        received_combined=cc[:,None]*(combined_received+nc).real+l[head]+u[head]
        assert np.allclose(received_combined,received_iq,rtol=1e-10,atol=1e-15)
        blocks=np.array([0,m//2,m-1]);trials=2000
        normals=rng.normal(size=(trials,2,len(blocks),dm))/np.sqrt(2)
        noise=ca[0,blocks][None,:,None]*normals[:,0]+ca[1,blocks][None,:,None]*normals[:,1]
        empirical=np.mean(np.sum(noise**2,axis=2),axis=0);target=va[:,blocks].sum(0)
        assert np.allclose(empirical,target,rtol=.025,atol=1e-30)
        rows.append(dict(dataset=info['dataset'],joint_index=info['joint_index'],
            physical_waveform_match=True,random_channel_phase=True,joint_power_inequality=True,
            independent_Combined_waveform_computed=True,
            max_update_difference=float(np.max(np.abs(received_combined-received_iq))),
            IQ_RF_energy_j=float(iq_energy.sum()),Combined_RF_energy_j=float(combined_energy.sum()),
            Monte_Carlo_trials=trials,analytic_block_MSE=target.tolist(),empirical_block_MSE=empirical.tolist()))
    result=dict(passed=True,snapshots=len(rows),checks=rows,source_sha256=study.sha(Path(__file__)),
        interpretation='The end-to-end mapped run deliberately couples the final Gaussian update to I/Q, so its tensor equality is a controlled consequence, not an independent quality win. This independent waveform computation verifies that the coupling corresponds to a physically valid single-stream Combined signal under the stated phase/noise model.')
    study.write(study.OUT/'mapped_waveform_check.json',result);print({k:v for k,v in result.items() if k!='checks'})


if __name__=='__main__':main()
