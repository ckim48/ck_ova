"""One proposed-only trajectory; only the radio RMS constraint changes."""
import os,sys,json,time,fcntl,subprocess,traceback
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from preflight import prepare,write,OUT,ROOT,clip,guard
stream=clip.stream;core=clip.core
ARMS={'parallel_cliperase_fu_guard':dict(parallel=True,crossmodal=True,period=0,digital=False,main=True)}

def status(**obj):write('status.json',dict(unix=time.time(),pid=os.getpid(),**obj))
def main():
    plan,requests=prepare()
    own=(OUT/'launch.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'started.json').exists(),'Duplicate campaign'
    assert json.loads((OUT/'preflight.json').read_text())['passed']
    assert (clip.OUT/'finished.json').exists(),'Parent not completed'
    gpu=(ROOT/'results/overnight_energy_v2/association_objective_gpu.lock').open('a');fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU occupied'
    write('started.json',dict(unix=time.time(),identity=stream.base.identity(os.getpid())))
    status(stage='loading_model')
    from train import Engine
    from dataset import RetrievalData
    engine=Engine(RetrievalData(),dict(plan));channels=np.load(core.SOURCE/'channels.npy')
    p=ROOT/'results/overnight_energy_v2/proxy_objective_v2/proxies/analog'
    proxy_theta=core.load_theta(p/'round_020.pt',engine.device)
    prefix=json.loads((clip.SOURCE/'common_prefix_cost.json').read_text());write('common_prefix_cost.json',prefix)
    # Reporting reference is read only after training-only preflight and source hashes are frozen.
    ref_theta=core.load_theta(core.SOURCE/'retrain/round_200.pt',engine.device)
    ref_eval,ref_probs=engine.evaluate(ref_theta,None);ref_mia,_=engine.mia_scores(ref_theta)
    threshold=.95*ref_eval['retained']['mR']
    assert abs(threshold-json.loads((clip.SOURCE/'reference_metrics.json').read_text())['retained_mR_threshold'])<1e-7
    write('reference_metrics.json',dict(metrics=ref_eval,retained_mR_threshold=threshold,reporting_only=True));write('reference_mia.json',ref_mia)
    stream.OUT=OUT;stream.ARMS=ARMS;stream.vectors=clip.vectors;stream.radio=guard
    stream.report()
    for name,opt in ARMS.items():
        stream.run_arm(engine,plan,requests,proxy_theta,ref_probs,threshold,channels,name,opt,prefix)
        path=OUT/name/'manifest.json';manifest=json.loads(path.read_text())
        if manifest['status']=='completed_200FL_and_requests':
            e=manifest['evaluation']
            manifest['final_joint_quality_pass']=bool(manifest['requests']['procedure_completed']==45 and e['retained']['mR']>=threshold and e['forget']['KL_retrain_to_candidate']<=.01 and e['worst_request_KL']<=.01)
            manifest['radio_note']='FU-relative total-noise cap; desired FL+FU sum unchanged. No claim of independent FU recovery or privacy certification.'
            core.write(path,manifest)
    prepare();stream.OUT=OUT;stream.report()
    write('finished.json',dict(unix=time.time(),status='registered_proposed_candidate_finished_or_reported_incomplete'))
    status(stage='completed')

if __name__=='__main__':
    try:main()
    except Exception as e:OUT.mkdir(parents=True,exist_ok=True);status(stage='failed',error=repr(e));traceback.print_exc();raise
    finally:guard.base.close()
