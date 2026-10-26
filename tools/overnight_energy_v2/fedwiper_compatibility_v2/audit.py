"""CPU compatibility and ideal-code packet budget audit; no learning ranking."""
from pathlib import Path
import os, sys, json, math, hashlib, time
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/overnight_energy_v2/fedwiper_compatibility_v2'
sys.path.insert(0,str(ROOT/'tools/overnight_energy_v2/full_system_baselines_v1'))
import run_memprune as run
import numpy as np


def ofdma_min_time(bits, gains, W, P, energy=None):
    """Minimize common upload T with optimized constant bandwidth fractions.

    gains = |h|^2 / full-band noise. Noise PSD is constant across fractions.
    Equal per-node bit payload; ideal coding; full-band total bandwidth W.
    Power <= P and optional per-node TX energy <= energy. No RX/CPU energy.
    """
    if bits==0:return {'time_s':0.,'fractions':[0.]*len(gains),'energy_per_user_j':0.}
    if energy is not None and np.any(bits>=W*energy*gains/math.log(2)):
        return None  # finite-energy wideband-capacity upper bound
    def fractions(T):
        power=P if energy is None else min(P,energy/T)
        need=bits/(W*T)
        if np.any(np.log2(1+power*gains)<need):return None
        lo=np.zeros(len(gains));hi=np.ones(len(gains))
        for _ in range(65):
            f=(lo+hi)/2
            rate=f*np.log2(1+power*gains/f)
            small=rate<need;lo=np.where(small,f,lo);hi=np.where(small,hi,f)
        return hi
    low=0.;high=1.
    for _ in range(80):
        f=fractions(high)
        if f is not None and f.sum()<=1:break
        high*=2
    else:raise RuntimeError('No finite numerical bracket')
    for _ in range(70):
        middle=(low+high)/2;f=fractions(middle)
        if f is not None and f.sum()<=1:high=middle
        else:low=middle
    T=high;f=fractions(T);p=P if energy is None else min(P,energy/T)
    delivered=T*W*f*np.log2(1+p*gains/f)
    assert f.sum()<=1+1e-10 and np.all(delivered>=bits*(1-1e-10))
    assert p<=P and (energy is None or p*T<=energy*(1+1e-10))
    before=fractions(T*(1-1e-6))
    assert before is None or before.sum()>1
    return dict(time_s=float(T),fractions=f.tolist(),power_per_user_w=float(p),
                energy_per_user_j=float(p*T),total_energy_j=float(len(gains)*p*T),
                boundary_checked=True)


def main():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from train import Engine
    from clip_model import DeepPromptCLIP
    from dataset import RetrievalData
    from objectives import contrastive
    torch.set_num_threads(1);torch.manual_seed(823)
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'results.json').exists(),'Do not overwrite completed audit'
    run.core.write(OUT/'started.json',dict(unix=time.time(),pid=os.getpid(),device='cpu'))
    class ForgetLayer(nn.Module):
        # Exact arithmetic/parameterization of official module/module.py.
        def __init__(self,channel):
            super().__init__()
            self.layer1=nn.Conv2d(channel,channel*2,kernel_size=1)
            self.relu=nn.ReLU()
            self.layer2=nn.Conv2d(channel*2,channel,kernel_size=1)
        def forward(self,x):
            identity=x;x=self.layer1(x);x=self.relu(x);x=self.layer2(x)
            x+=identity;x=self.relu(x);return x
    plan=json.loads((run.core.SOURCE/'plan.json').read_text())
    records=json.loads((run.core.SOURCE/'records.json').read_text());c=plan['config']
    e=object.__new__(Engine);e.data=RetrievalData();e.plan=plan;e.device=torch.device('cpu')
    e.model=DeepPromptCLIP.from_local_checkpoint(ROOT/'checkpoints/clip-vit-base-patch16',
        prompt_layers=8,prompt_tokens=2,checkpoint_layers=False).eval()
    e.teacher=None;e.teacher_images={};e.teacher_texts={}
    theta=run.core.load_theta(run.PREFIX/'round_020.pt',e.device)
    ids=records[20]['retained'][0][:4];cids=e.data.training_caption_ids(ids,21)
    assert set(ids).isdisjoint(plan['validation_images']+plan['official_test_images'])
    assert set(ids).isdisjoint(plan['forgotten_images'])
    with torch.no_grad():im,tx,scale=e.features(theta,ids,cids)
    C=im.shape[1];adapter=ForgetLayer(C)
    assert sum(p.numel() for p in adapter.parameters())==4*C*C+3*C
    zero=ForgetLayer(C)
    with torch.no_grad():
        zero.layer2.weight.zero_();zero.layer2.bias.zero_()
        out=zero(im[:,:,None,None]).squeeze(-1).squeeze(-1)
        assert torch.equal(out,im.clamp_min(0))
        positive=im.abs()
        assert torch.equal(zero(positive[:,:,None,None]).flatten(1),positive)
    adapted=F.normalize(adapter(im[:,:,None,None]).flatten(1),dim=-1)
    owner=torch.tensor(ids);loss=contrastive(adapted,tx,scale,owner,owner)
    loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in adapter.parameters())
    shape=dict(training_images=len(ids),width=C,parameters=sum(p.numel() for p in adapter.parameters()),
        negative_fraction=float((im<0).float().mean()),
        zero_branch_relative_change=float((out-im).norm()/im.norm()),
        zero_branch_mean_cosine=float(F.cosine_similarity(out,im).mean()),
        positive_activation_identity_check=True,finite_adapter_gradients=True,
        original_relu_preserved=True,
        interpretation='A post-embedding CLIP insertion is not neutral even with a zero residual branch. This is an insertion compatibility diagnostic, not a trained FedWiper performance result.')
    # Independent closed-form symmetric-channel validation.
    g=np.full(9,12.);B=30000;W=1e6;P=.32
    exact=B/(W/9*math.log2(1+P*12*9))
    control=ofdma_min_time(B,g,W,P)
    assert abs(control['time_s']/exact-1)<1e-10
    double=ofdma_min_time(2*B,g,W,P)
    assert abs(double['time_s']/control['time_s']-2)<1e-10
    channels=np.load(run.core.SOURCE/'channels.npy');rows=[]
    for adapters in [1,2]:
        d=(4*C*C+3*C)*adapters
        for bits in [8,32]:
            payload=bits*d+(32*math.ceil(d/128) if bits==8 else 0)
            for cap in [None,float(c['available_energy_j'])]:
                candidates=[]
                for head in range(c['users']):
                    gains=np.delete(channels[:,head],head)**2
                    sol=ofdma_min_time(payload,gains,c['symbol_rate_hz'],c['power_limit_w'],cap)
                    if sol is not None:candidates.append(dict(head=head,**sol))
                best=min(candidates,key=lambda x:x['time_s']) if candidates else None
                rows.append(dict(adapters=adapters,parameters_per_user=d,bits=bits,payload_bits_per_user=payload,
                    metadata='32 bits per128-coordinate block' if bits==8 else 'native float32, no quantizer scale',
                    energy_cap_per_user_j=cap,feasible_heads=len(candidates),best=best,
                    coherence_frames=math.ceil(best['time_s']/c['coherence_time_s']) if best else None))
    result=dict(status='cpu_audit_completed',ranking_permitted=False,device='cpu',shape_probe=shape,radio=rows,
        radio_checks=dict(symmetric_closed_form=True,payload_linear_scaling_without_energy_cap=True,
                          returned_constraints_and_previous_time_boundary=True),
        official_source='https://github.com/grey1989/FedWiper/blob/master/module/module.py',
        source_notes=['Official MyRes.forward activates only forget4; forget1..3 exist but are bypassed.',
          'Official CNN classifier is task-specific and absent from this optimistic retrieval payload count.',
          'Official run serializes full state_dict; this audit permits only active trainable adapter transmission, a stronger efficient baseline.',
          'Official voting evaluator averages class probabilities and updates BN on test images before scoring. Such test adaptation is excluded from our validation protocol and must be disclosed in any VLM port.'],
        unresolved=['VLM insertion and retrieval ensemble are adaptations, not exact reproduction.',
          'Affected shard retraining from pre-data initialization and own matched reference remain unimplemented.',
          'OFDMA calculation assumes ideal coding, static channels across frames, zero pilot/control cost. Frame count is an accounting plan, not a packet-level decoder simulation.',
          'Q8 reconstruction quality for a trained adapter has not been verified; no energy-to-target result.',
          'Time minima use the per-node power/energy caps; reported energy is energy at minimum time, not minimum-energy optimization.'],
        input_hashes={str(p.relative_to(ROOT)):run.core.sha(p) for p in [run.core.SOURCE/'plan.json',run.core.SOURCE/'channels.npy',run.PREFIX/'round_020.pt']})
    run.core.write(OUT/'results.json',result)
    print('AUDIT_SUMMARY'+json.dumps(result)+'AUDIT_END',flush=True)

if __name__=='__main__':main()
