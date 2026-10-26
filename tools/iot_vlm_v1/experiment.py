"""Fresh four-method VLM evaluation on common optimized IoT-driven geometry."""
from pathlib import Path
import fcntl
import importlib.util
import json
import os
import sys
import time
import traceback
import subprocess

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/iot_uav_remoteclip_v1'
GEOMETRY=ROOT/'results/iot_uav_deployment_v1'
sys.path.insert(0,str(ROOT/'tools/uav_real_vlm_v1'))
spec=importlib.util.spec_from_file_location('iot_original_vlm',ROOT/'tools/uav_real_vlm_v1/experiment.py')
original=importlib.util.module_from_spec(spec);sys.modules[spec.name]=original;spec.loader.exec_module(original)
read,write,sha=original.read,original.write,original.sha


def state(dest=OUT,**kwargs):
    original.state(dest,**kwargs)


def register(base):
    if (OUT/'protocol.json').exists():return read(OUT/'protocol.json')
    from iot_energy import PARAMETERS
    base.OLD=GEOMETRY
    registration=base.repeat.clone_function(original.register,
        [('uav_remoteclip_capera_v1','iot_uav_remoteclip_v1',2)],
        {'OUT':OUT,'HERE':HERE})
    protocol=registration(base)
    protocol.update(geometry_directory=str(GEOMETRY.relative_to(ROOT)),energy_model=PARAMETERS,
                    geometry_policy='Same precomputed IoT-driven UAV positions and inter-UAV channel matrices for all four methods and both datasets. Nominal planning heads are not fixed: actual transport solvers enumerate/reselect heads on each update.',
                    iot_data_scope='Synthetic ground IoT packet demand defines deployment and collection cost. Real CapERA/RSITMD samples are a logical VLM replay, not geotagged observations of these ground devices. No claimed physical mapping of videos to IoT sensors.',
                    failed_prior_run='Original RemoteCLIP campaign stopped after six initial FL rounds; no final comparison results used. This independent geometry campaign starts fresh with the same registered model/training settings.')
    for path in (ROOT/'tools/uav_real_vlm_v1').glob('*.py'):
        protocol['source_sha256'][str(path.relative_to(ROOT))]=sha(path)
    for path in (ROOT/'tools/iot_uav_deployment_v1').glob('*.py'):
        protocol['source_sha256'][str(path.relative_to(ROOT))]=sha(path)
    for name in ['mission_spec.json','channels_by_site.npy','deployment.json','registered_inputs.json','trajectory.csv']:
        path=GEOMETRY/name;protocol['input_sha256'][str(path.relative_to(ROOT))]=sha(path)
    for dataset in protocol['datasets']:
        dest=OUT/(dataset.lower()+'_seed0')
        plan=read(dest/'plan.json');plan['physical_scenario']='iot_uav_deployment_v1'
        write(dest/'plan.json',plan)
        protocol['input_sha256'][str((dest/'plan.json').relative_to(ROOT))]=sha(dest/'plan.json')
        requests=read(dest/'requests.json');geo=read(GEOMETRY/'mission_spec.json')
        for request in requests:
            burst=geo['request_bursts'][request['burst']]
            request['old_mission_arrival_s']=burst['arrival_s']
            request['old_mission_deadline_s']=burst['arrival_s']+30.
        write(dest/'requests.json',requests)
        protocol['input_sha256'][str((dest/'requests.json').relative_to(ROOT))]=sha(dest/'requests.json')
    plot=ROOT/'tools/plot_mission_map.py'
    protocol['source_sha256'][str(plot.relative_to(ROOT))]=sha(plot)
    write(OUT/'protocol.json',protocol)
    return protocol


def verify():
    protocol=read(OUT/'protocol.json')
    for group in ['input_sha256','source_sha256','frozen_source_sha256']:
        for path,digest in protocol[group].items():assert sha(ROOT/path)==digest,path
    return protocol


def actual_head_figure(dataset):
    folder=OUT/(dataset.lower()+'_seed0')/'proposed_fu_guard_cliperase'
    if not (folder/'manifest.json').exists():return
    if read(folder/'manifest.json')['status']!='completed_200FL_and_requests':return
    subprocess.run([sys.executable,str(ROOT/'tools/plot_mission_map.py'),'--mode','iot',
                    '--deployment',str(GEOMETRY),'--transactions',str(folder/'transactions')],
                   cwd=ROOT,check=True,env=dict(os.environ,MPLCONFIGDIR='/tmp/ova_ck_mpl'))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    own=(OUT/'campaign.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'started.json').exists(),'Refuse duplicate campaign'
    original.prepare_videos()
    base=original.baseline_module()
    try:
        sys.path.insert(0,str(HERE))
        from accounting import instrumented_engine
        from iot_report import generate
        from iot_energy import base as energy_base
        import torch
        assert torch.cuda.is_available()
        protocol=register(base);verify()
        write(OUT/'preflight.json',dict(energy=energy_base.self_check(),geometry=read(GEOMETRY/'completed.json')))
        gpu=(ROOT/'results/overnight_energy_v2/association_objective_gpu.lock').open('a')
        state(stage='waiting_for_shared_GPU_lock');fcntl.flock(gpu,fcntl.LOCK_EX)
        adapted=base.repeat.clone_function(base.adapted_arm,
            [('frozen_CLIP_B32_residual_rank10','RemoteCLIP_B32_mean8_IoT_geometry_rank10',1)])
        pure=base.repeat.clone_function(base.pure_trajectory,[],{'state':state})
        run_dataset=base.repeat.clone_function(base.run_dataset,
            [("data=(RetrievalData if dataset=='RSITMD' else base.RetrievalData)()",'data=Data(dataset)',1)],
            {'OUT':OUT,'OLD':GEOMETRY,'Data':original.Data,'prepare_features':original.prepare_features,
             'verify':verify,'state':state,'adapted_arm':adapted,'pure_trajectory':pure,
             'Engine':instrumented_engine(base.Engine)})
        write(OUT/'started.json',dict(pid=os.getpid(),unix=time.time(),gpu=torch.cuda.get_device_name(),protocol_sha256=sha(OUT/'protocol.json')))
        results=[]
        for dataset in protocol['datasets']:
            state(stage='running_dataset',dataset=dataset)
            try:
                result=run_dataset(dataset,protocol)
                results.append(dict(dataset=dataset,arms=result))
                actual_head_figure(dataset)
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
