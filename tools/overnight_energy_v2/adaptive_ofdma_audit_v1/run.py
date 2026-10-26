"""Independent CPU audit; leaves registered training sources and outputs intact."""
from pathlib import Path
import sys, os, json, hashlib, time, traceback
os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'tools/overnight_energy_v2/adaptive_ofdma_v2'))
import adaptive_radio as radio
import radio_ofdma as old
import numpy as np
OUT = ROOT/'results/overnight_energy_v2/adaptive_ofdma_audit_v1'

def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def ledger(folder, proxy=False):
    folder=ROOT/folder
    rows=[read(p) for p in sorted(folder.glob('round_*.json'))] if proxy else read(folder/'prefix_rounds.json')
    assert [r['round'] for r in rows]==list(range(1,21))
    sums={k:sum(r['radio'][k] for r in rows) for k in ['energy_bound_j','expected_energy_j','uplink_s']}
    target=read(folder/'complete.json') if proxy else rows[-1]
    mapping={k:k for k in sums} if proxy else dict(energy_bound_j='cumulative_tx_energy_bound_j',expected_energy_j='cumulative_tx_energy_expected_j',uplink_s='cumulative_uplink_s')
    for k,v in sums.items(): assert abs(v-target[mapping[k]])<1e-9, (str(folder),k)
    if 'ofdma_v1' in str(folder):
        for r in rows:
            a=r['radio'];n=np.asarray(a['subcarrier_counts']);f=np.asarray(a['bandwidth_fractions'])
            assert a['access']=='integer_subcarrier_OFDMA' and a['ofdma_subcarriers']==128
            assert np.max(n.sum(0))<=128 and np.allclose(n/128,f)
            assert a['maximum_digital_power_w']<=.32*(1+1e-9) and a['rate_margin_bits']>=-1e-6
    return dict(rounds=len(rows),costs=sums,checkpoint_sha256=sha(folder/'round_020.pt'))

def main():
    OUT.mkdir(parents=True,exist_ok=True);start=time.time()
    protocol=read(ROOT/'results/overnight_energy_v2/adaptive_ofdma_v2/protocol.json')
    verified={p:sha(ROOT/p)==s for p,s in protocol['source_hashes'].items()}
    assert all(verified.values())
    ledgers={}
    for name,path,proxy in [('analog_prefix','retrieval_long200_v3/proposed',False),('ofdma_prefix','ofdma_v1/learning/ofdma_initial20',False),('analog_proxy','proxy_objective_v2/proxies/analog',True),('ofdma_proxy','ofdma_v1/learning/proxies/oma',True)]:
        ledgers[name]=ledger('results/overnight_energy_v2/'+path,proxy)
    print('LEDGERS_VERIFIED',flush=True)
    c=read(ROOT/'results/overnight_energy_v2/retrieval_long200_v3/plan.json')['config']
    z=np.load(ROOT/'results/overnight_energy_v2/retrieval_fourway_v2/fixed_updates.npz')
    l,u,h=z['learning'],z['unlearning'],z['channels'];zero=np.zeros_like(u)
    av,ar=radio.communicate(l,zero,h,c,'ofdma_only',48123)
    bv,br=old.communicate(l,zero,h,c,'ofdma_only',48123)
    assert av is not None and bv is not None
    assert np.allclose(av,bv,rtol=1e-9,atol=1e-12)
    for key in ['energy_bound_j','uplink_s','error_bound']: assert abs(ar[key]-br[key])<1e-10,key
    prefix=dict(coordinates=l.shape[1],maximum_vector_difference=float(np.max(np.abs(av-bv))),head_new=ar['head'],head_old=br['head'],learning_bits=ar['learning_bits'],energy_j=ar['energy_bound_j'],uplink_s=ar['uplink_s'])
    print('DIGITAL_PREFIX_MATCH',flush=True)
    # Independently calculate exact conditional MSE for the selected preflight designs.
    c=dict(c,quantization_bits=6,ofdma_subcarriers=128)
    ll,uu,q,delta,payload,rho,dims,total=radio.make_inputs(l,u,c)
    eps=max(c['absolute_tolerance_floor'],c['relative_tolerance']*float(np.linalg.norm(rho.sum(0))))
    pre=read(ROOT/'results/overnight_energy_v2/adaptive_ofdma_v2/preflight.json')['results'];mse={}
    for mode,pr in pre.items():
        d=radio.design(pr['head'],h,ll,uu,q,delta,payload,rho,dims,total,eps,c,mode)
        assert d is not None
        head,users=d['head'],d['users'];a=d['analog'];variance=0.
        if mode=='ofdma_only': mean=ll[head]+d['quantized_learning'][users].sum(0)+uu[head]+q[users].sum(0)
        else:
            lb=ll+uu if mode=='aircomp_aircomp' else ll
            rb=np.linalg.norm(lb,axis=-1)
            gain=a.receiver*h[users,head,None]*a.analog_rms/np.where(rb[users]>0,rb[users],1.)
            mean=lb[head]+(gain[:,:,None]*lb[users]).sum(0)
            if mode!='aircomp_aircomp':mean+=uu[head]+q[users].sum(0)
            variance=float(np.sum(dims*a.receiver**2/(2*d['analog_symbols'])))
        bias=mean-(ll+uu).sum(0);exact=float(np.sqrt(np.sum(bias**2)+variance))
        assert exact<=d['raw_error_bound']*(1+3e-5)<=eps*(1+3e-5)
        assert abs(d['energy_j']-pr['energy_bound_j'])<1e-8
        mse[mode]=dict(exact_conditional_rms=exact,raw_rms_bound=d['raw_error_bound'],epsilon=eps,bias_norm=float(np.linalg.norm(bias)),noise_variance=variance,selected_boundary=d['metadata'].get('selected_boundary'))
        print('EXACT_MSE_VERIFIED '+mode,flush=True)
    assert all(sha(ROOT/p)==s for p,s in protocol['source_hashes'].items())
    out=dict(passed=True,unix=time.time(),runtime_s=time.time()-start,source_hashes_verified=verified,ledgers=ledgers,digital_no_fu_reuse=prefix,conditional_raw_mse=mse,
        scope='CPU fixed-update integration and accounting audit only. No new learning or model-selection evidence; no frozen source changes.',source_sha256=sha(__file__))
    (OUT/'report.json').write_text(json.dumps(out,indent=2)+'\n')
    print('AUDIT_PASSED',flush=True)

if __name__=='__main__':
    try: main()
    except Exception as e:
        OUT.mkdir(parents=True,exist_ok=True)
        (OUT/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2)+'\n')
        raise
    finally:radio.close();old.close()
