"""Run the revised transmission model in a separate, serial GPU campaign."""
from pathlib import Path
import copy,hashlib,json,os,subprocess,time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
OUT=ROOT/'results/overnight_energy_v2/partial_overlap_tx_v1/learning_comparison'
PY=ROOT/'.venv/bin/python'  # Explicit environment wrapper; sys.executable bypasses it on this host.

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def status(**kw):write(OUT/'queue_status.json',dict(updated_unix=time.time(),**kw))

def main():
    import fcntl
    assert json.loads((OUT.parent/'transport_validation.json').read_text())['status']=='passed'
    OUT.mkdir(exist_ok=False)
    lock=(OUT/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    old=ROOT/'results/overnight_energy_v2/deletion_window_screen_v1'
    continuation=ROOT/'results/overnight_energy_v2/energy_to_target_v1'
    records=json.loads((old/'records.json').read_text())+json.loads((continuation/'records.json').read_text())
    assert [x['round'] for x in records]==list(range(61,81))
    first=records[0]
    for r in records:
        for u in range(10):
            assert not set(r['batches'][u]).intersection(first['forget'][u])
            if r['round']>61:assert not r['forget'][u]
    write(OUT/'records.json',records)
    plan=copy.deepcopy(json.loads((old/'plan.json').read_text()))
    plan['config'].update(base_overhead_j=0.,head_overhead_j=0.)
    plan['methods']=['proposed','aircomp_oma','aircomp_only','oma_only']
    plan['records_sha256']=sha(OUT/'records.json')
    plan['scope']='Twenty rounds (61-80) from common historical round 60; one K=5 deletion burst. Partial-overlap transmission and uplink-only cost definition requested by user. Baselines remove crossmodal coupling. Utility comparisons do not certify equal forgetting.'
    plan['timing_scope']='Primary total_delay_s is cumulative occupied uplink transmission time only. All computation/reconstruction/SIC/control/downlink excluded; diagnostic wall times retained separately.'
    plan['energy_scope']='Uplink waveform transmit energy only, robust bound and uncorrelated-codeword expectation. No receiver/decoding/optimization/training/fixed overhead cost.'
    plan['transmission_policy']='Request round: proposed overlaps analog first 2*d symbols inside digital 14*d symbols; strictly positive digital overlap. AirComp+OMA optimizes its split within 16*d symbols. AirComp-only and nonrequest analog rounds use 2*d symbols. All-digital OMA uses 16*d. Same bandwidth, per-user power/battery, noise, RMS quality budget, canonical Q8 deletion payload and common deadline.'
    plan['metric_change_audit']='Old 336ms trajectories are not relabeled as 294ms. New model trajectories are evaluated separately.'
    plan['fixed_forget_probe_ids']=json.loads((continuation/'plan.json').read_text())['fixed_forget_probe_ids']
    plan['source_sha256'].update({str(p.relative_to(ROOT)):sha(p) for p in [
        ROOT/'tools/overnight_energy_v2/deletion_window_screen_v1/trajectory_transport.py',
        ROOT/'tools/overnight_energy_v2/deadline_frontier_v1/run_frontier.py']})
    plan['runner_sha256']={p.name:sha(p) for p in HERE.glob('*.py')}
    write(OUT/'plan.json',plan)
    env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    results=[]
    for method in plan['methods']:
        while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            status(status='waiting_for_gpu',method=method,methods=results);time.sleep(30)
        with (OUT/f'{method}.log').open('w') as log:
            child=subprocess.Popen([str(PY),'-u',str(HERE/'run_overlap.py'),'--plan',str(OUT/'plan.json'),'--method',method],
                                   cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            status(status='running',method=method,pid=child.pid,methods=results);rc=child.wait()
        mf=OUT/method/'manifest.json'
        state=json.loads(mf.read_text())['status'] if mf.exists() else 'failed_before_manifest'
        results.append(dict(method=method,returncode=rc,status=state))
        with (OUT/'report.log').open('a') as log:
            subprocess.run([str(PY),str(HERE/'report_tx_only.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        if rc!=0:
            status(status='failed',method=method,methods=results)
            return
    status(status='completed',methods=results)

if __name__=='__main__':main()
