"""Detached resumable queue. Independent arms fail visibly; no result tuning."""
from common import *
import argparse, fcntl, subprocess, traceback

def publish():
    git=['git','--git-dir=.git-push-ready/code-charts.git','--work-tree=.']
    def call(args):return subprocess.check_output(git+args,cwd=ROOT,text=True).strip()
    assert call(['remote','get-url','origin'])=='https://github.com/ckim48/ck_ova.git'
    if call(['diff','--cached','--name-only']):return dict(status='deferred_existing_staged_changes')
    paths=list(HERE.glob('*.py'))+[OUT/'protocol.json',OUT/'dataset_sources.json',OUT/'README.md',OUT/'preflight.json']+list((OUT/'figures').glob('*'))
    paths=[str(p.relative_to(ROOT)) for p in paths if p.is_file()]
    subprocess.run(git+['add','--']+paths,cwd=ROOT,check=True,capture_output=True)
    if not call(['diff','--cached','--name-only']):return dict(status='unchanged')
    subprocess.run(git+['commit','-m','Add paired UAV sensitivity sweeps and separated completion delays'],cwd=ROOT,check=True,capture_output=True)
    subprocess.run(git+['push','origin','main'],cwd=ROOT,check=True,capture_output=True,timeout=180)
    commit=call(['rev-parse','HEAD']);assert call(['ls-remote','origin','refs/heads/main']).split()[0]==commit
    return dict(status='pushed',commit=commit)

def register():
    path=OUT/'protocol.json'
    if path.exists():return read(path)
    hashes={}
    for dataset in ['RSITMD','CapERA']:
        dd=SOURCE/(dataset.lower()+'_seed0')
        for name in ['plan.json','requests.json','feature_cache/features.pt','initial20/final.pt','proxy20/final.pt','reference200/final.pt']:
            f=dd/name;hashes[str(f.relative_to(ROOT))]=sha(f)
    p=dict(version='iot_sensitivity_v1_20261027',registered_unix=time.time(),seed=0,
           datasets=['RSITMD','CapERA'],additional_datasets=EXTRA_DATASETS,methods=METHODS,cases=cases(),
           axes={k:dict(label=v[0],values=v[1],baseline=v[2]) for k,v in AXES.items()},
           arms_per_pair=410,total_requested_arms=820,
           input_sha256=hashes,source_sha256={str(f.relative_to(ROOT)):sha(f) for f in HERE.glob('*.py')},
           authorization='User requested overnight 10-axis sweeps, a Proposed-matched heuristic head/OVA ablation, separate FL/FU and mission completion delays, and two additional UAV VLM datasets.',
           comparison='One-factor, paired seed0, fixed 200 FL and same 45 requests. H-OVA shares every Proposed learning/FU setting; centroid head + uniform inverse noise scale replace optimized head and SOCP.',
           initialization='Conditional post-initialization sensitivity: shared cold embeddings, initial20, proxy20 and reporting-only reference200 per dataset. No baseline final learned checkpoint reused. Prefix radio costs remain reference-condition costs; CPU/downlink changes recost prefix operations.',
           propagation='Bandwidth sweep keeps noise PSD fixed by normalizing channel amplitude by sqrt(B/1MHz). Gain sweep scales all h by 10^(offset/20). Other geometry held fixed.',
           delays='FL/FU completion is takeoff to final work broadcast, including preceding movement/collection. Mission also includes return. Total also includes preflight initialization. Active service and occupied uplink additionally reported.',
           failure_policy='Infeasible radio or collection points remain gaps, all raw partial records retained. No successful-run retries or outcome-based adjustment.',
           source_campaign=str(SOURCE.relative_to(ROOT)),additional_data_policy='User explicitly authorized other VLM datasets if unavailable. Official GeoText-1652 drone views (ECCV2024, also used by ACM MM2025 HCCM; Google Earth virtual drone views) and UAVReason captions on real UAVScenes RGB (ICCV2025 imagery; caption extension is a2026 preprint). Fixed subsets for system sensitivity, not full official benchmark reproduction. Native3/1 captions, no duplication. ERA/UDV original captions obtained but Baidu images unavailable; AeroCaps gated, not used.')
    write(path,p);return p

def main():
    OUT.mkdir(parents=True,exist_ok=True);lock=(OUT/'campaign.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=register()
    for f,h in p['input_sha256'].items():assert sha(ROOT/f)==h,f
    write(OUT/'launch.json',dict(pid=os.getpid(),unix=time.time(),protocol_sha256=sha(OUT/'protocol.json')))
    from report import generate
    runs=[]
    def run_case(case,pair):
        state(stage='scenario',case=case['id'],datasets=pair)
        log=OUT/'logs'/('_'.join(d.lower() for d in pair)+'_'+case['id']+'.log');log.parent.mkdir(exist_ok=True)
        with log.open('a') as f:
            result=subprocess.run([sys.executable,str(HERE/'worker.py'),'--case',case['id'],'--datasets',*pair],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        runs.append(dict(case=case['id'],datasets=pair,returncode=result.returncode));write(OUT/'runs.json',runs)
        report=generate();state(stage='scenario_terminal',case=case['id'],report=report)
    run_case(p['cases'][0],p['datasets'])
    from prepare_extra import prepare
    ready=[]
    for dataset in p['additional_datasets']:
        try:
            result=prepare(dataset)
            if result['status']=='ready':ready.append(dataset)
            write(OUT/(dataset.lower()+'_data_status.json'),result)
        except Exception as e:write(OUT/(dataset.lower()+'_data_status.json'),dict(status='data_preparation_failed',error=repr(e),traceback=traceback.format_exc()))
    if ready:run_case(p['cases'][0],ready)
    for i,case in enumerate(p['cases'][1:],1):
        for pair in [p['datasets'],ready]:
            if pair:run_case(case,pair)
        # Four nonbaseline points finish each axis; publish the completed figures.
        if i%4==0:
            try:publication=publish()
            except Exception as e:publication=dict(status='failed',error=str(e))
            write(OUT/'publication.json',publication)
    report=generate();pending=[d for d in p['additional_datasets'] if not (OUT/'prepared'/(d.lower()+'_seed0')/'ready.json').exists()]
    write(OUT/'finished.json',dict(unix=time.time(),report=report,pending_datasets=pending,status='available_datasets_finished_pending_images' if pending else 'finished'))
    state(stage='terminal',pending_datasets=pending,report=report)

if __name__=='__main__':
    try:main()
    except Exception as e:
        write(OUT/'campaign_failure.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
