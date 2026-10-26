"""User-authorized energy-overrun continuation after the constrained campaign."""
from pathlib import Path
import sys,json,os,time,fcntl,shutil,traceback,subprocess
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tools/iot_vlm_recovery_v1'))
import recover
from overrun_radio import build,ORIGINAL_CAP_J
read,write,sha=recover.read,recover.write,recover.sha
SOURCE=recover.OUT
OUT=ROOT/'results/iot_uav_remoteclip_v3_energy_overrun'
METHOD='sequential_digital_periodic'

def state(dest=OUT,**kwargs):recover.original.state(dest,**kwargs)

def register():
    recover.verify()
    assert (SOURCE/'finished.json').exists(),'Wait for constrained campaign to terminate'
    p=read(SOURCE/'protocol.json');inputs={}
    for key,digest in p['input_sha256'].items():
        source=ROOT/key
        if source.is_relative_to(SOURCE):
            target=OUT/source.relative_to(SOURCE);target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,target);key=str(target.relative_to(ROOT))
        inputs[key]=digest
    reused={}
    for dataset in p['datasets']:
        src=SOURCE/(dataset.lower()+'_seed0');dest=OUT/src.name
        for name in ['feature_cache','initial20','proxy20','reference200']:
            marker=src/name/('manifest.json' if name=='feature_cache' else 'finished.json')
            assert marker.exists(),marker
            shutil.copytree(src/name,dest/name)
        for name in ['common_prefix_cost.json','reference_metrics.json','reference_mia.json','common_checkpoint_hashes.json']:
            shutil.copy2(src/name,dest/name)
        for method in p['plotting']['labels']:
            if method==METHOD:continue
            manifest=src/method/'manifest.json'
            assert manifest.exists() and read(manifest)['status']=='completed_200FL_and_requests',(dataset,method)
            shutil.copytree(src/method,dest/method)
        for file in dest.rglob('*'):
            if file.is_file():reused[str(file.relative_to(ROOT))]=sha(file)
    p.update(version='iot_uav_v3_energy_overrun',registered_unix=time.time(),input_sha256=inputs,
             reused_artifact_sha256=reused,continuation_of=str(SOURCE.relative_to(ROOT)),
             authorization='User explicitly requested continuation even when the 0.100 J network transmission cap is exceeded.',
             energy_policy='S-OF first attempts the original constrained design. On failure only the network sum-energy cap is relaxed: choose the longest permitted frame and its minimum-energy head. Peak power, per-user energy, coherence, quantization error and uplink deadline remain hard limits. Each transaction records original-cap compliance and excess. Other three methods reuse their completed original-cap runs.',
             comparison_scope='Energy-overrun sensitivity comparison. S-OF+ may violate the original shared network-energy cap; this is not an all-method original-cap-feasible comparison.')
    p['plotting']['labels'][METHOD]='S-OF+'
    for file in HERE.glob('*.py'):p['source_sha256'][str(file.relative_to(ROOT))]=sha(file)
    write(OUT/'protocol.json',p);return p

def verify():
    p=read(OUT/'protocol.json')
    for group in ['input_sha256','source_sha256','frozen_source_sha256','reused_artifact_sha256']:
        for key,digest in p[group].items():assert sha(ROOT/key)==digest,key
    return p

def energy_with_overruns(dataset,method,geometry):
    from iot_energy import method_energy
    result=method_energy(dataset,method,geometry)
    rows=[read(p)['radio'] for p in sorted((Path(dataset)/method/'transactions').glob('*.json'))]
    energies=[float(r['energy_bound_j']) for r in rows]
    result.update(original_network_tx_cap_j=ORIGINAL_CAP_J,
                  network_cap_overrun_transactions=sum(e>ORIGINAL_CAP_J*(1+2e-6) for e in energies),
                  network_cap_total_excess_j=sum(max(0.,e-ORIGINAL_CAP_J) for e in energies),
                  max_transaction_tx_energy_j=max(energies,default=0.),
                  network_cap_policy='soft, original-feasible preferred' if method==METHOD else 'original hard cap')
    return result

def make_arm(base):
    cap_edit='''return repeat.clone_function(old.run_arm,[
  ("assert rm['energy_bound_j']<=.1*(1+2e-6)","assert np.isfinite(rm['energy_bound_j']) and rm['energy_bound_j']>=0",1),'''
    adapted=base.repeat.clone_function(base.adapted_arm,[
        ('frozen_CLIP_B32_residual_rank10','RemoteCLIP_B32_mean8_global_IoT_five_regions_rank10_energy_overrun',1),
        ('site=min(fl//45,3)','site=min(fl//36,4)',1),
        ('return repeat.clone_function(old.run_arm,[',cap_edit,1)])
    arm=adapted();assert callable(arm)
    return arm

def main():
    import torch
    from accounting import instrumented_engine
    import iot_report
    OUT.mkdir(parents=True,exist_ok=True)
    own=(OUT/'campaign.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'started.json').exists()
    state(stage='queued_after_constrained_campaign',source=str(SOURCE.relative_to(ROOT)))
    # The active recovery keeps this lock until its outputs and status are saved.
    gpu=(ROOT/'results/overnight_energy_v2/association_objective_gpu.lock').open('a');fcntl.flock(gpu,fcntl.LOCK_EX)
    protocol=register();verify();assert torch.cuda.is_available()
    base=recover.original.baseline_module();base.base.SequentialRadio=build(base)
    arm=make_arm(base)
    Engine=instrumented_engine(base.Engine)
    write(OUT/'started.json',dict(pid=os.getpid(),unix=time.time(),gpu=torch.cuda.get_device_name(),protocol_sha256=sha(OUT/'protocol.json')))
    results=[]
    try:
        for dataset in protocol['datasets']:
            verify();dest=OUT/(dataset.lower()+'_seed0');state(stage='running_energy_overrun_arm',dataset=dataset,method=METHOD)
            started=time.time()
            try:
                plan=read(dest/'plan.json');requests=read(dest/'requests.json')
                geo=read(recover.GEOMETRY/'mission_spec.json')
                import numpy as np
                channels=np.load(recover.GEOMETRY/'channels_by_site.npy')
                payload=torch.load(dest/'feature_cache/features.pt',map_location='cpu',weights_only=False)
                data=recover.Data(dataset);engine=Engine(data,plan,payload)
                warm=base.core.load_theta(dest/'initial20/final.pt',engine.device)
                proxy=base.core.load_theta(dest/'proxy20/final.pt',engine.device)
                ref=base.core.load_theta(dest/'reference200/final.pt',engine.device)
                engine.set_teacher(warm);_,refprob=engine.evaluate(ref,None)
                threshold=read(dest/'reference_metrics.json')['retained_mR_threshold']
                arm(engine,plan,requests,geo,channels,METHOD,base.base.ARMS[METHOD],read(dest/'common_prefix_cost.json'),warm,proxy,refprob,threshold,dest,0)
                base.validate_arm(dest/METHOD,base.base.ARMS[METHOD])
                results.append(dict(dataset=dataset,method=METHOD,status='completed'))
                del engine,data,payload,refprob
                import gc;gc.collect();torch.cuda.empty_cache()
            except Exception as error:
                write(dest/METHOD/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),unix=time.time()))
                results.append(dict(dataset=dataset,method=METHOD,status='failed',error=repr(error)))
            write(dest/METHOD/'runtime_seconds.json',dict(elapsed_server_wall_seconds=time.time()-started))
            write(OUT/'dataset_results.json',results)
        generate=base.repeat.clone_function(iot_report.generate,[],{'method_energy':energy_with_overruns})
        report=generate(OUT)
        if report['status']=='completed':
            from publish import publish_figures
            report['publication']=publish_figures(ROOT,OUT)
        write(OUT/'finished.json',dict(results=results,report=report,unix=time.time()))
        state(stage='terminal',results=results,report=report)
    finally:base.base.parent.common.base.close()

if __name__=='__main__':
    try:main()
    except Exception as error:
        write(OUT/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),unix=time.time()))
        state(stage='failed',error=repr(error));raise
