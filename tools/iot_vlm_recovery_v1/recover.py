"""Resume valid work, fix video seeking, and isolate transport failures by arm.

No numerical radio constraint or learning setting is relaxed. Failed artifacts
remain in the original run; this run has a separate protocol and directory.
"""
from pathlib import Path
import importlib.util,sys,json,time,os,fcntl,shutil,traceback
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tools/iot_vlm_v3_global_iot'))
spec=importlib.util.spec_from_file_location('failed_iot_campaign',ROOT/'tools/iot_vlm_v3_global_iot/experiment.py')
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
original=prior.original
read,write,sha=prior.read,prior.write,prior.sha
SOURCE=prior.OUT
OUT=ROOT/'results/iot_uav_remoteclip_v3_recovery'
GEOMETRY=prior.GEOMETRY
import numpy as np

class Data(original.Data):
    def frames(self,index):
        if self.dataset=='RSITMD':return super().frames(index)
        import cv2
        from PIL import Image
        path=self.media[int(index)];capture=cv2.VideoCapture(str(path))
        count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        assert capture.isOpened() and count>=8,str(path)
        indices=np.linspace(0,count-1,8).round().astype(int).tolist()
        selected=set(indices);frames={}
        try:
            for frame in range(indices[-1]+1):
                okay,pixels=capture.read()
                assert okay,(str(path),'sequential_decode',frame,count)
                if frame in selected:frames[frame]=Image.fromarray(cv2.cvtColor(pixels,cv2.COLOR_BGR2RGB))
        finally:capture.release()
        return [frames[i] for i in indices],indices

def state(dest=OUT,**kwargs):original.state(dest,**kwargs)

def register():
    assert not (OUT/'protocol.json').exists()
    prior.verify()
    assert read(SOURCE/'finished.json')['report']['status']=='incomplete'
    p=read(SOURCE/'protocol.json');hashes={}
    for key,digest in p['input_sha256'].items():
        source=ROOT/key
        if source.is_relative_to(SOURCE):
            target=OUT/source.relative_to(SOURCE);target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,target);key=str(target.relative_to(ROOT))
        hashes[key]=digest
    p['input_sha256']=hashes
    reused={}
    src=SOURCE/'rsitmd_seed0';dest=OUT/'rsitmd_seed0'
    for name in ['feature_cache','initial20','proxy20']:
        assert (src/name/('manifest.json' if name=='feature_cache' else 'finished.json')).exists()
        shutil.copytree(src/name,dest/name)
        for file in (dest/name).rglob('*'):
            if file.is_file():reused[str(file.relative_to(ROOT))]=sha(file)
    p.update(version='iot_uav_v3_recovery',recovery_unix=time.time(),recovery_of=str(SOURCE.relative_to(ROOT)),
        recovery_policy='Preserve all failed records. Reuse hash-checked completed RSITMD feature and common-prefix artifacts only. Decode CapERA sequentially at the same eight uniform frame indices. Per-case preflight failures are recorded without aborting unrelated arms; every actual arm still enforces all original constraints. No numerical setting changes.',
        reused_artifact_sha256=reused)
    for file in HERE.glob('*.py'):p['source_sha256'][str(file.relative_to(ROOT))]=sha(file)
    write(OUT/'protocol.json',p)
    return p

def verify():
    p=read(OUT/'protocol.json')
    for group in ['input_sha256','source_sha256','frozen_source_sha256','reused_artifact_sha256']:
        for key,digest in p[group].items():assert sha(ROOT/key)==digest,key
    return p

def prepare_features(data,plan,checkpoint,dest,writer,status):
    if (dest/'features.pt').exists():
        import torch
        assert data.dataset=='RSITMD'
        verify();return torch.load(dest/'features.pt',map_location='cpu',weights_only=False),read(dest/'manifest.json')
    return original.prepare_features(data,plan,checkpoint,dest,writer,status)

BASE=None
PURE=None

def pure_trajectory(engine,plan,records,ch,kind,dest):
    folder=dest/kind
    if (folder/'finished.json').exists():
        assert plan['dataset']=='RSITMD' and kind in ['initial20','proxy20']
        verify();m=read(folder/'finished.json');assert m['status']=='completed' and m['rounds']==len(records)
        return BASE.core.load_theta(folder/'final.pt',engine.device),dict(energy_j=m['TX_energy_j'],uplink_s=m['uplink_s'],server_compute_seconds=m['server_compute_seconds'])
    return PURE(engine,plan,records,ch,kind,dest)

def physical_checks(l,u,old,ch,c,label):
    b=BASE.base;p=b.parent
    cases=[('proposed',p.guard.analog,l,u),('separate_FL',p.common.analog,l,np.zeros_like(l)),
           ('separate_FU',p.common.analog,np.zeros_like(l),old),
           ('hybrid_FU',lambda a,v,ch,c,s:p.common.digital_fu(v,ch,c),np.zeros_like(l),old),
           ('digital_FL',b.SequentialRadio.analog,l,np.zeros_like(l)),('digital_FU',b.SequentialRadio.analog,np.zeros_like(l),u)]
    results={}
    for name,fn,a,v in cases:
        vector,m=fn(a,v,ch,c,500000)
        if vector is None or m['status']!='feasible':
            results[name]=dict(passed=False,radio=m);continue
        assert np.isfinite(vector).all() and m['conditional_rmse']<=m['epsilon']*(1+3e-5)+1e-9
        assert m['energy_bound_j']<=.1*(1+2e-6) and m['max_power_w']<=c['power_limit_w']*(1+2e-6)
        assert m['max_user_energy_j']<=c['available_energy_j']*(1+2e-6) and m['uplink_s']<=c['round_uplink_deadline_s']+1e-10
        results[name]=dict(passed=True,radio=m)
    b.write(label,dict(all_cases_passed=all(x['passed'] for x in results.values()),records=results,
                       policy='Failed feasibility probes do not abort unrelated arms. Actual arms enforce original radio constraints and retain failures.'))

def main():
    global BASE,PURE
    import torch
    from accounting import instrumented_engine
    from iot_report import generate
    from iot_energy import check_five_site_accounting
    OUT.mkdir(parents=True,exist_ok=True)
    own=(OUT/'campaign.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'started.json').exists()
    protocol=read(OUT/'protocol.json') if (OUT/'protocol.json').exists() else register()
    verify();assert torch.cuda.is_available();assert check_five_site_accounting()['passed']
    gpu=(ROOT/'results/overnight_energy_v2/association_objective_gpu.lock').open('a');fcntl.flock(gpu,fcntl.LOCK_EX)
    BASE=base=original.baseline_module();base.base.physical_checks=physical_checks
    PURE=base.repeat.clone_function(base.pure_trajectory,[],{'state':state})
    adapted=base.repeat.clone_function(base.adapted_arm,[('frozen_CLIP_B32_residual_rank10','RemoteCLIP_B32_mean8_global_IoT_five_regions_rank10',1),('site=min(fl//45,3)','site=min(fl//36,4)',1)])
    adapted()
    run=base.repeat.clone_function(base.run_dataset,
        [("data=(RetrievalData if dataset=='RSITMD' else base.RetrievalData)()",'data=Data(dataset)',1)],
        {'OUT':OUT,'OLD':GEOMETRY,'Data':Data,'prepare_features':prepare_features,'verify':verify,'state':state,
         'adapted_arm':adapted,'pure_trajectory':pure_trajectory,'Engine':instrumented_engine(base.Engine)})
    write(OUT/'started.json',dict(pid=os.getpid(),unix=time.time(),gpu=torch.cuda.get_device_name(),protocol_sha256=sha(OUT/'protocol.json')))
    results=[]
    try:
        for dataset in protocol['datasets']:
            state(stage='running_dataset',dataset=dataset)
            try:
                arms=run(dataset,protocol);results.append(dict(dataset=dataset,arms=arms))
                prior.OUT=OUT;prior.actual_head_figure(dataset)
            except Exception as error:
                write(OUT/(dataset.lower()+'_seed0')/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),unix=time.time()))
                results.append(dict(dataset=dataset,status='failed',error=repr(error)))
            write(OUT/'dataset_results.json',results)
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
