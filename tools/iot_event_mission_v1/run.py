"""Fresh eight-arm completion-driven mission; frozen common prefixes reused."""
from pathlib import Path
import sys,os,time,json,fcntl,shutil,traceback,importlib.util,argparse
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tools/iot_vlm_energy_overrun_v1'))
spec=importlib.util.spec_from_file_location('registered_energy_overrun',ROOT/'tools/iot_vlm_energy_overrun_v1/run.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
recover=parent.recover
from event_arm import make_arm
from event_checks import preflight,validate
from event_energy import method_energy,PARAMETERS
from overrun_radio import build
read,write,sha=recover.read,recover.write,recover.sha
SOURCE=parent.OUT
OUT=ROOT/'results/iot_uav_remoteclip_v4_event_mission'
GEOMETRY=recover.GEOMETRY

def state(dest=OUT,**kwargs):
    recover.original.state(dest,**kwargs)
    if dest!=OUT:recover.original.state(OUT,dataset=Path(dest).name,**kwargs)

def register():
    parent.verify();assert read(SOURCE/'finished.json')['report']['status']=='completed'
    p=read(SOURCE/'protocol.json');inputs={};reused={}
    for key,digest in p['input_sha256'].items():
        source=ROOT/key
        if source.is_relative_to(SOURCE):
            target=OUT/source.relative_to(SOURCE);target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,target);key=str(target.relative_to(ROOT))
        inputs[key]=digest
    for dataset in p['datasets']:
        src=SOURCE/(dataset.lower()+'_seed0');dest=OUT/src.name
        for name in ['feature_cache','initial20','proxy20','reference200']:shutil.copytree(src/name,dest/name)
        for name in ['common_prefix_cost.json','reference_metrics.json','reference_mia.json','common_checkpoint_hashes.json']:shutil.copy2(src/name,dest/name)
        for file in dest.rglob('*'):
            if file.is_file():reused[str(file.relative_to(ROOT))]=sha(file)
    p.update(version='iot_uav_v4_event_mission',registered_unix=time.time(),input_sha256=inputs,reused_artifact_sha256=reused,
             source_campaign=str(SOURCE.relative_to(ROOT)),source_protocol_sha256=sha(SOURCE/'protocol.json'),energy_model=PARAMETERS,
             authorization='User identified fixed residence after work completion as the modeling problem; implement immediate departure after all required collection and FL/FU work. Prior approval to allow S-OF energy-cap overruns persists.',
             mission_policy='Five zones, same physical path/speeds/IoT demand. Each zone: 10s IoT collection, 36 FL updates and all assigned five-step FU requests, then immediate departure. No 60s minimum. Return travel retained.',
             clock_policy='Every transaction: maximum modeled device compute duration, uplink and complete model downlink. Queue triggers and learning updates run on this full clock, rather than postprocessing old timestamps.',
             arrival_policy='Same FL-round-triggered requests as the prior campaign; no external-arrival load test and no active deadlines. Five FU steps are procedural completion, not certified deletion.',
             common='Reuse hash-checked common cold features, initial20, proxy20 and reporting-only reference200. Rerun all eight adaptation arms from common initial state under the new clock.',
             recovery_policy='Historical failure records remain intact; no comparison-arm final checkpoint is reused.',
             energy_policy='Communication plus executed device compute plus actual hover and unchanged travel energy; collection charged once. S-OF+ retains original-cap-preferred fallback and logs overruns.',
             comparison_scope='Completion-driven mission sensitivity, one seed. Model/objectives differ across method families; real data remain a logical replay on synthetic IoT demand.')
    for file in HERE.glob('*.py'):p['source_sha256'][str(file.relative_to(ROOT))]=sha(file)
    write(OUT/'protocol.json',p);return p

def verify():
    p=read(OUT/'protocol.json')
    for group in ['input_sha256','source_sha256','frozen_source_sha256','reused_artifact_sha256']:
        for key,digest in p[group].items():assert sha(ROOT/key)==digest,key
    return p

def plot_fu_latency(rows):
    import matplotlib.pyplot as plt
    labels=read(OUT/'protocol.json')['plotting']['labels']
    fig,axes=plt.subplots(1,2,figsize=(7.16,2.7),layout='constrained')
    for ax,dataset in zip(axes,['RSITMD','CapERA']):
        subset={r['method']:r for r in rows if r['dataset']==dataset};values=[subset[m]['FU_mean_completion_latency_s'] for m in labels]
        bars=ax.bar(range(4),values,width=.6,color=['#167D8D','#4978A8','#D69B48','#8A769F'],edgecolor='white',linewidth=.5)
        ax.set_xticks(range(4),list(labels.values()));ax.set_title(dataset);ax.set_ylabel('Mean FU completion latency (s)')
        ax.set_ylim(0,max(values)*1.2);ax.spines[['top','right']].set_visible(False)
        ax.bar_label(bars,labels=[f'{v:.3f}' for v in values],padding=3,fontsize=7,color='black')
    for extension in ['png','pdf','svg']:fig.savefig(OUT/'figures'/f'fu_completion_latency_1x2.{extension}',dpi=450,bbox_inches='tight')
    plt.close(fig)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight-only',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    own=(OUT/'campaign.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'started.json').exists()
    protocol=read(OUT/'protocol.json') if (OUT/'protocol.json').exists() else register();verify()
    base=recover.original.baseline_module();base.base.SequentialRadio=build(base)
    geometry=read(GEOMETRY/'mission_spec.json');deployment=read(GEOMETRY/'deployment.json')
    arm=make_arm(base,deployment,state)
    report={d:preflight(base,read(OUT/(d.lower()+'_seed0')/'requests.json'),geometry,deployment) for d in protocol['datasets']}
    write(OUT/'preflight.json',dict(passed=True,datasets=report,arm_compiled=True))
    if args.preflight_only:state(stage='preflight_passed');base.base.parent.common.base.close();return
    import torch,numpy as np
    from accounting import instrumented_engine
    import iot_report
    assert torch.cuda.is_available()
    gpu=(ROOT/'results/overnight_energy_v2/association_objective_gpu.lock').open('a');fcntl.flock(gpu,fcntl.LOCK_EX)
    write(OUT/'started.json',dict(pid=os.getpid(),unix=time.time(),gpu=torch.cuda.get_device_name(),protocol_sha256=sha(OUT/'protocol.json')))
    Engine=instrumented_engine(base.Engine);channels=np.load(GEOMETRY/'channels_by_site.npy');results=[]
    try:
        for dataset in protocol['datasets']:
            verify();dest=OUT/(dataset.lower()+'_seed0')
            plan=read(dest/'plan.json');requests=read(dest/'requests.json');payload=torch.load(dest/'feature_cache/features.pt',map_location='cpu',weights_only=False)
            data=recover.Data(dataset);engine=Engine(data,plan,payload)
            warm=base.core.load_theta(dest/'initial20/final.pt',engine.device);proxy=base.core.load_theta(dest/'proxy20/final.pt',engine.device)
            ref=base.core.load_theta(dest/'reference200/final.pt',engine.device);engine.set_teacher(warm);_,refprob=engine.evaluate(ref,None)
            threshold=read(dest/'reference_metrics.json')['retained_mR_threshold']
            for method in protocol['plotting']['labels']:
                verify();state(stage='running_arm',dataset=dataset,method=method);started=time.time()
                try:
                    arm(engine,plan,requests,geometry,channels,method,base.base.ARMS[method],read(dest/'common_prefix_cost.json'),warm,proxy,refprob,threshold,dest,0)
                    checked=validate(dest/method,base.base.ARMS[method]['parallel'],geometry)
                    write(dest/method/'transmission_verification.json',checked)
                    metrics=method_energy(dest,method,geometry);write(dest/method/'energy_and_delay.json',metrics)
                    results.append(dict(dataset=dataset,method=method,status='completed'))
                except Exception as error:
                    write(dest/method/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),unix=time.time()))
                    results.append(dict(dataset=dataset,method=method,status='failed',error=repr(error)))
                write(dest/method/'runtime_seconds.json',dict(elapsed_server_wall_seconds=time.time()-started));write(OUT/'dataset_results.json',results)
            del engine,data,payload,refprob
            import gc;gc.collect();torch.cuda.empty_cache()
        generate=base.repeat.clone_function(iot_report.generate,[],{'method_energy':method_energy,'PARAMETERS':PARAMETERS})
        final=generate(OUT)
        if final['status']=='completed' and all(r['status']=='completed' for r in results):
            plot_fu_latency(read(OUT/'figures/metrics.json')['rows'])
            from publish import publish_figures
            final['publication']=publish_figures(ROOT,OUT)
        write(OUT/'finished.json',dict(results=results,report=final,unix=time.time()));state(stage='terminal',results=results,report=final)
    finally:base.base.parent.common.base.close()

if __name__=='__main__':
    try:main()
    except Exception as error:
        write(OUT/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),unix=time.time()))
        state(stage='failed',error=repr(error));raise
