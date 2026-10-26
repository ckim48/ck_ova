from pathlib import Path
import sys,json,time,importlib.util
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
OUT=ROOT/'results/overnight_energy_v2/parallel_digital_v1'
sys.path.insert(0,str(HERE));import radio_digital as radio
spec=importlib.util.spec_from_file_location('parallel_digital_clip_parent',ROOT/'tools/overnight_energy_v2/cliperase_fu_v1/run.py')
clip=importlib.util.module_from_spec(spec);spec.loader.exec_module(clip)
def write(name,obj):clip.core.write(OUT/name,obj)
def prepare():
    plan,requests=clip.prepare();OUT.mkdir(parents=True,exist_ok=True)
    parent=ROOT/'results/overnight_energy_v2/fu_protected_aircomp_v1/protocol.json'
    hashes=dict(json.loads(parent.read_text())['source_hashes']);hashes[str(parent.relative_to(ROOT))]=clip.core.sha(parent)
    for p in HERE.glob('*.py'):hashes[str(p.relative_to(ROOT))]=clip.core.sha(p)
    protocol=dict(version='parallel_digital_v1',registered_unix=time.time(),source_hashes=hashes,
      authorization='User asked to add an all-digital parallel scheme to the three main comparisons.',
      algorithm='Same CLIPErase-inspired FL/FU vectors and immediate parallel request scheduler as the proposed method. Each UAV forms its local FL+FU sum then quantizes once atQ6. Individually decode each UAV sum through OFDMA and sum at head.',
      multiplexing='All UAV packets share the same block interval on disjoint integer subcarriers, total128tones. FL and FU are contained in a single local-sum packet, not independently recoverable component streams. Not NOMA, and no duplicate bandwidth.',
      fairness='Same data, common initial20-round checkpoint and prefix,45 exogenous requests,5 FU steps/request,200 total FL steps,losses,learning rates,seed,channels,power/energy limits,frame grid and FU-relative RMS target as proposed. Local pre-addition allowed because the desired server operation is the same sum.',
      costs='Q6 payload plus original32bit block scale metadata. Ideal successful decoding under rate constraints; quantization distortion is included relative to raw FL+FU. Count all digital FL+FU transmissions; pure uplink excludes compute/control/recovery/downlink consistently.',
      parent_comparisons='Do not modify or restart separate AirComp or AirComp+OFDMA. Their existing FU objectives differ from this matched-loss digital comparison and that distinction remains reported.',
      gate='Existing OFDMA kernel analytical/exhaustive tests plus full20480-coordinate actual training snapshot, FL-only and FU-only feasibility,raw quantization bound,payload/tone/power/battery/delay validation. No reference/test tuning.',
      evaluation='Same retained validation mR/loss,mean/worst request retrainKL,prediction differences,MIA,request procedure/deadline counts,energy and pure uplink. Record incomplete physical feasibility or missed joint quality; no success-cost rank for failures.',
      scope='One all-digital trajectory. Existing campaigns and results preserved. This is our controlled digital transport baseline, not a published federated unlearning algorithm reproduction.')
    if (OUT/'protocol.json').exists():protocol=json.loads((OUT/'protocol.json').read_text())
    else:write('protocol.json',protocol);write('requests.json',requests)
    for n,h in protocol['source_hashes'].items():assert clip.core.sha(ROOT/n)==h,n
    return plan,requests
def main():
    plan,_=prepare();c=plan['config'];z=np.load(clip.OUT/'fixed_training_updates.npz');l,u,ch=z['FL'],z['FU'],z['channels']
    checks=radio.base.old.ofdma.checks();assert checks['passed'];records={}
    for name,ll,uu in [('joint',l,u),('FL_only',l,np.zeros_like(u)),('FU_only',np.zeros_like(l),u),('zero',np.zeros_like(l),np.zeros_like(u))]:
        v,m=radio.analog(ll,uu,ch,c,500000);records[name]=m
        assert v is not None and m['status']=='feasible',name
        assert m['conditional_rmse']<=m['epsilon']*(1+1e-5)+1e-9
        assert m['energy_bound_j']<=radio.base.CAP*(1+1e-7)
        assert m['max_power_w']<=c['power_limit_w']*(1+1e-6)
        assert m['max_user_energy_j']<=c['available_energy_j']*(1+1e-7)
        assert m['uplink_s']<=c['round_uplink_deadline_s']+1e-12
        if 'subcarrier_counts' in m:assert np.all(np.asarray(m['subcarrier_counts']).sum(0)<=128)
        assert np.isfinite(v).all()
        print(name,{k:m.get(k) for k in ['energy_bound_j','uplink_s','conditional_rmse','epsilon','total_payload_bits']},flush=True)
    # A lone active UAV can be its own head; exact local sum has no upload cost.
    ll=np.zeros_like(l);uu=np.zeros_like(u);ll[2]=l[2];uu[2]=u[2]
    v,m=radio.analog(ll,uu,ch,c,1);assert np.array_equal(v,(ll+uu).sum(0)) and m['uplink_s']==0
    write('preflight.json',dict(passed=True,unix=time.time(),dimension=l.shape[1],OFDMA_kernel=checks,records=records,
      single_active_local_exact=True,physical_power_energy_delay_and_tones_checked=True,config=c))
    print('PREFLIGHT_PASSED',flush=True)
if __name__=='__main__':
    try:main()
    finally:radio.base.close()
