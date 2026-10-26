"""Add one paired cross-modal ablation after the frozen main serial GPU queue."""
from pathlib import Path
import copy,hashlib,json,os,subprocess,time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
MAIN=ROOT/'results/overnight_energy_v2/partial_overlap_tx_v1/learning_comparison'
RUNNERS=ROOT/'tools/overnight_energy_v2/partial_overlap_tx_v1'
OUT=ROOT/'results/overnight_energy_v2/partial_overlap_ablation_v1'
PY=ROOT/'.venv/bin/python'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def status(**kw):write(OUT/'queue_status.json',dict(updated_unix=time.time(),**kw))

def main():
    import fcntl
    OUT.mkdir(exist_ok=False)
    lock=(OUT/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    original=json.loads((MAIN/'plan.json').read_text())
    for name,expected in original['runner_sha256'].items():assert sha(RUNNERS/name)==expected
    for name,expected in original['source_sha256'].items():assert sha(ROOT/name)==expected
    code=(RUNNERS/'run_overlap.py').read_text()
    assert "if args.method == 'proposed' and batch:" in code
    assert "mode = 'hybrid' if args.method in ('proposed','hybrid_no_coupling')" in code
    plan=copy.deepcopy(original)
    plan['methods']=['hybrid_no_coupling']
    plan['scope']='Paired ablation of proposed partial-overlap NOMA-AirComp: only projected visual-text cross-modal preconditioning is skipped. The VLM, prompts, shared parameter, gradients, data schedule, checkpoint, quantizer, receiver controller, wireless optimization and transmit-only costs stay identical.'
    plan['paired_reference']=str((MAIN/'proposed').relative_to(ROOT))
    plan['reference_plan_sha256']=sha(MAIN/'plan.json')
    plan['effective_crossmodal_preconditioner']='Identity (equivalent to kappa=0), implemented by the existing hybrid_no_coupling method switch. Stored config is deliberately identical to the paired reference.'
    plan['ablation_runner_sha256']={p.name:sha(p) for p in HERE.glob('*.py')}
    (OUT/'records.json').write_bytes((MAIN/'records.json').read_bytes())
    assert sha(OUT/'records.json')==plan['records_sha256']
    assert plan['config']==original['config']
    write(OUT/'plan.json',plan)
    write(OUT/'paired_design_audit.json',dict(status='passed',same_config=True,same_records_sha256=plan['records_sha256'],
          same_warm_checkpoint_sha256=plan['warm_checkpoint_sha256'],same_original_runner_hashes=True,
          only_behavioral_switch='Skip projected_coupling; both methods call the same hybrid communicate mode and receiver controller.'))
    env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    while True:
        q=json.loads((MAIN/'queue_status.json').read_text())
        if q['status'] in ('completed','failed'):break
        status(status='waiting_for_main_serial_queue',prior_method=q.get('method'),prior_status=q['status']);time.sleep(30)
    mf=MAIN/'proposed/manifest.json'
    if not mf.exists() or json.loads(mf.read_text())['status']!='screen_completed':
        status(status='blocked_reference_not_complete');return
    while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        status(status='waiting_for_gpu');time.sleep(30)
    with (OUT/'hybrid_no_coupling.log').open('w') as log:
        child=subprocess.Popen([str(PY),'-u',str(RUNNERS/'run_overlap.py'),'--plan',str(OUT/'plan.json'),
                                '--method','hybrid_no_coupling'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        status(status='running',method='hybrid_no_coupling',pid=child.pid);rc=child.wait()
    manifest=OUT/'hybrid_no_coupling/manifest.json'
    observed=json.loads(manifest.read_text())['status'] if manifest.exists() else 'failed_before_manifest'
    with (OUT/'report.log').open('w') as log:
        rr=subprocess.run([str(PY),str(HERE/'report_five_methods.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    status(status='completed' if rc==0 and observed=='screen_completed' and rr.returncode==0 else 'failed',
           returncode=rc,run_status=observed,report_returncode=rr.returncode)

if __name__=='__main__':main()
