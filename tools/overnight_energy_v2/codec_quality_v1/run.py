"""Predeclared Q6 quality screen, serialized after the existing GPU campaign.

No physical-radio ranking: ideal analog transport with native blockwise FU
quantization. Q8 results are reused only after identical source/protocol checks.
"""
from pathlib import Path
import os, sys, json, time, fcntl, traceback, subprocess, importlib.util
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=ROOT/'results/overnight_energy_v2/codec_quality_v1'
BASE=ROOT/'results/overnight_energy_v2/head_persistent_v1'
QOUT=ROOT/'results/overnight_energy_v2/full_system_baselines_v1'
spec=importlib.util.spec_from_file_location('unchanged_head_pilot',ROOT/'tools/overnight_energy_v2/head_persistent_v1/run_pilot.py')
hp=importlib.util.module_from_spec(spec);spec.loader.exec_module(hp)
core=hp.core
STOP=60
ARMS=['q6','raw_ideal']

def write(name,data):core.write(OUT/name,data)
def status(**data):write('status.json',dict(unix=time.time(),pid=os.getpid(),**data))
def identity(pid):
    try:
        s=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()
        return dict(pid=pid,start=s[19],state=s[0])
    except FileNotFoundError:return None
def same_live(x):
    y=identity(x['pid'])
    return y is not None and y['start']==x['start'] and y['state']!='Z'

def encode(l,u,c,bits):
    if bits is None:return u.copy(),dict(payload_bits=0,identity=True,ideal_no_packet_cost_claim=True)
    from phase_split_grid import make_inputs
    x=make_inputs(l,u,dict(c,quantization_bits=bits))
    q=x[2].reshape(len(u),-1)[:,:u.shape[1]]
    true_block_error=np.linalg.norm(x[2]-x[1],axis=-1)
    assert np.all(true_block_error<=x[3]+1e-12)
    assert np.isfinite(q).all()
    return q,dict(bits=bits,payload_bits=int(x[4].sum()),
        active_blocks=int(np.count_nonzero(x[4])),
        aggregate_error_l2=float(np.linalg.norm((q-u).sum(0))),
        aggregate_error_triangle_bound=float(np.linalg.norm(x[3].sum(0))),
        max_client_relative_l2=float(max(np.linalg.norm(q[i]-u[i])/max(np.linalg.norm(u[i]),1e-30) for i in range(len(u)))))

def check_codec(plan):
    from wireless import quantize
    z=np.load(BASE/'common_transport_snapshot.npz')
    l=z['learning'];u=z['unlearning'];c=plan['config'];b=c['block_dimension']
    results={}
    for bits in [4,6,8]:
        q,diag=encode(l,u,c,bits)
        padded=np.pad(u,((0,0),(0,(-u.shape[1])%b))).reshape(len(u),-1,b)
        levels=2**(bits-1)-1
        scale=(np.max(np.abs(padded),axis=-1)/levels).astype(np.float32).astype(np.float64)
        reconstructed=np.clip(np.rint(padded/np.where(scale>0,scale,1)[...,None]),-levels,levels)*scale[...,None]
        assert np.array_equal(q,reconstructed.reshape(len(u),-1)[:,:u.shape[1]])
        native,bound,payload=quantize(padded,bits,c['scale_metadata_bits'])
        assert np.array_equal(q,native.reshape(len(u),-1)[:,:u.shape[1]])
        assert diag['payload_bits']==int(np.count_nonzero(np.max(np.abs(padded),axis=-1))*(b*bits+c['scale_metadata_bits']))
        if bits==8:assert np.array_equal(q,z['canonical_unlearning'])
        zero,zz=encode(l,np.zeros_like(u),c,bits);assert not np.any(zero) and zz['payload_bits']==0
        odd,oo=encode(l[:,:131],u[:,:131],c,bits);assert odd.shape==(len(u),131)
        results[str(bits)]=diag
    q,diag=encode(l,u,c,None);assert np.array_equal(q,u)
    result=dict(passed=True,native_float32_scale=True,q8_exact_previous_snapshot=True,
        zero_payload_and_zero_update=True,padding_shape=True,raw_identity=True,quantizers=results,
        caveat='Training-update reconstruction/payload checks only; no VLM quality or transmission guarantee.')
    write('codec_checks.json',result)
    return result

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    plan=json.loads((core.SOURCE/'plan.json').read_text())
    for k,v in plan['sources'].items():assert core.sha(ROOT/k)==v,k
    for k,v in plan['frozen_engine_hashes'].items():assert core.sha(core.FROZEN/k)==v,k
    for k,v in plan['runner_hashes'].items():assert core.sha(core.LONG/k)==v,k
    old=json.loads((BASE/'pilot_protocol.json').read_text())
    for k,v in old['sources'].items():assert core.sha(ROOT/k)==v,k
    assert old['stop']==60 and old['seed']==0
    assert json.loads((BASE/'status.json').read_text())['stage']=='completed_pilot'
    checks=check_codec(plan)
    queue=json.loads((QOUT/'queue_started.json').read_text())
    qidentity=identity(queue['pid'])
    launch=OUT/'registered_queue_identity.json'
    if launch.exists():qidentity=json.loads(launch.read_text())
    else:
        assert qidentity is not None and 'full_system_audit_v1/queue.py' in Path(f'/proc/{qidentity["pid"]}/cmdline').read_bytes().replace(b'\0',b' ').decode()
        write('registered_queue_identity.json',qidentity)
    paths=[Path(__file__),ROOT/'wireless.py',core.SOURCE/'plan.json',core.SOURCE/'records.json',
           core.SOURCE/'proposed/round_020.pt',hp.PRIOR/'proxies/ideal/round_020.pt',
           BASE/'pilot_protocol.json',BASE/'pilot_results.json']
    protocol=dict(version='codec_quality_v1',seed=0,arms=ARMS,q8_baseline='reuse unchanged head_persistent_v1/sum at round25..60',
        stop=STOP,unlearning_rounds=list(range(21,26)),eval_rounds=list(range(25,61,5)),
        objective='Identical repartitioned retain CE + retained relation KL + forget-proxy KL; no head control.',
        teacher='Existing retain-only20-step proxy; no final-reference parameters used for updates.',
        codec='Native wireless.quantize:128 coordinates per block, symmetric Q6 with float32 scale and32 metadata bits. Same update partition as completed Q8 pilot.',
        selection='Q6 fixed in advance from training-update CPU cost/fidelity diagnostics. No hyperparameter sweep driven by official test or final-reference outputs.',
        gate='Report predeclared Q6-vs-Q8 quality margins at40 and60: loss+0.01,mR-0.005,mean/worst forget KL<=1.01 times Q8. Exploratory validation audit only; no automatic tuning or long-run promotion.',
        data='Official test untouched. Existing fixed retained validation and deleted training probes, same references and schedule for all arms.',
        equivalence='Ideal analog plus identical canonical FU applies to NOMA and Air+OFDMA equally. raw_ideal is an upper-quality transport control, not energy-zero winner.',
        costs='Log quantized FU payload only; full physical energy and uplink time are not simulated. CPU block frontiers cannot be extrapolated to time-to-target.',
        queue='Start only after existing MemPrune and matched-reference parent exits successfully, then claim shared GPU mutex. No SIGSTOP and no source edits.',
        predecessor_identity=qidentity,source_hashes={str(p.relative_to(ROOT)):core.sha(p) for p in paths},
        inherited_source_hashes=old['sources'])
    p=OUT/'protocol.json'
    if p.exists():assert json.loads(p.read_text())==protocol,'Protocol changed'
    else:write('protocol.json',protocol)
    return plan,protocol

def verify(protocol):
    for group in ['source_hashes','inherited_source_hashes']:
        for k,v in protocol[group].items():assert core.sha(ROOT/k)==v,k

def run(plan,protocol):
    import torch
    from train import Engine,update_theta
    from dataset import RetrievalData
    engine=Engine(RetrievalData(),dict(plan))
    initial=core.load_theta(core.SOURCE/'proposed/round_020.pt',engine.device)
    engine.set_teacher(initial)
    teacher=core.Teacher(engine,core.load_theta(hp.PRIOR/'proxies/ideal/round_020.pt',engine.device))
    recs=json.loads((core.SOURCE/'records.json').read_text());forget=recs[20]['forget']
    forbidden=set(i for pool in forget for i in pool)
    assert forbidden.isdisjoint(plan['validation_images'])
    assert forbidden.isdisjoint(plan['official_test_images'])
    for rec in recs[20:STOP]:assert all(not forbidden.intersection(ids) for ids in rec['retained'])
    baseline=json.loads((BASE/'pilot_results.json').read_text())['results']['sum']
    results={'q8_reused':baseline};diagnostics={};completed={}
    write('q8_provenance.json',dict(source=str(BASE/'pilot_results.json'),sha256=core.sha(BASE/'pilot_results.json'),
        reason='Same source function, initial model, teacher, records, LR, Q8 codec and uncorrected sum; nativeQ8 snapshot exact equality tested.'))
    for arm in ARMS:
        theta=initial.clone();results[arm]={};payload=0;active_requests=set();finite=True
        for k in range(21,STOP+1):
            t0=time.time();status(stage='learning',arm=arm,round=k,stop=STOP)
            engine.plan['learning_rate']=core.lr_at(k)
            l,u,stats,err=hp.repartitioned_updates(engine,theta,recs[k-1],k,forget,teacher)
            q,codec=encode(l,u,plan['config'],6 if arm=='q6' else None)
            payload+=codec['payload_bits'];active_requests.update(i for i,v in enumerate(u) if np.any(v))
            if k>25:assert not np.any(u) and codec['payload_bits']==0
            theta=update_theta(theta,(l+q).sum(0));assert torch.isfinite(theta).all()
            row=dict(round=k,arm=arm,codec=codec,cumulative_fu_payload_bits=payload,local=stats,
                     repartition_relative_error=err,finite=True)
            if k%5==0:
                e=hp.evaluation(engine,theta,k,forget);row['evaluation']=e;results[arm][str(k)]=e
                core.save_theta(OUT/arm/f'round_{k:03d}.pt',theta,k)
            row['wall_s']=time.time()-t0;core.write(OUT/arm/'rounds'/f'{k:03d}.json',row)
            print(json.dumps(dict(arm=arm,round=k,wall_s=row['wall_s'])),flush=True)
        mia,_=engine.mia_scores(theta);core.write(OUT/arm/'mia.json',mia)
        completed[arm]=dict(rounds=STOP,requests=len(active_requests),payload_bits=payload,
            requests_completed=active_requests=={i for i,x in enumerate(forget) if x})
        write('interim_results.json',results)
    for k in ['40','60']:
        a=results['q8_reused'][k];b=results['q6'][k]
        diagnostics[k]=dict(retained_loss=b['retained']['loss']<=a['retained']['loss']+.01,
            retained_mR=b['retained']['mR']>=a['retained']['mR']-.005,
            mean_KL=b['forget']['KL_retrain_to_candidate']<=1.01*a['forget']['KL_retrain_to_candidate'],
            worst_KL=hp.worst(b)<=1.01*hp.worst(a))
    verify(protocol)
    write('results.json',dict(results=results,completed=completed,predeclared_validation_margins=diagnostics,
        q6_within_margins=all(all(v.values()) for v in diagnostics.values()) and all(v['requests_completed'] for v in completed.values()),
        no_automatic_promotion=True,interpretation='Codec quality screen only; no physical or NOMA-specific superiority claim.'))
    write('finished.json',dict(unix=time.time(),status='completed',stop=STOP));status(stage='completed')

def main():
    plan,protocol=prepare()
    if '--prepare' in sys.argv:print('CODEC_PREFLIGHT_PASSED');return
    claim=(OUT/'pilot.lock').open('a');fcntl.flock(claim,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'started.json').exists(),'Duplicate pilot'
    status(stage='waiting_for_existing_campaign',predecessor=protocol['predecessor_identity'])
    while same_live(protocol['predecessor_identity']):time.sleep(30)
    end=json.loads((QOUT/'queue_status.json').read_text())
    if end.get('status')!='completed':
        status(stage='blocked_predecessor_not_completed',predecessor=end);return
    lock=(ROOT/'results/overnight_energy_v2/association_objective_gpu.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX)
    for _ in range(30):
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
        if not apps.strip():break
        time.sleep(2)
    else:raise RuntimeError('Unexpected GPU user after acquiring mutex')
    verify(protocol);write('started.json',dict(unix=time.time(),pid=os.getpid()))
    status(stage='loading_model');run(plan,protocol)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        OUT.mkdir(parents=True,exist_ok=True);status(stage='failed',error=repr(exc));traceback.print_exc();raise
