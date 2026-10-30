"""Check immutable baselines, full reporting, actual radio caps and conclusions."""
import csv
import json
from pathlib import Path
import numpy as np
import joint_quality_control_v1 as study
import joint_quality_audit_v1 as audit
import joint_quality_report_v1 as report

ROOT,read,sha=study.ROOT,study.read,study.sha


def rows(name):
    with (report.DEST/'data'/name).open() as f:return list(csv.DictReader(f))


def main():
    immutable=read(study.audit.OUT/'protocol.json')['input_sha256']
    for name,digest in immutable.items():assert sha(ROOT/name)==digest,name
    screens=rows('all_122_exact_screens.csv');physical=rows('all_physical_candidates.csv')
    assert len(screens)==122 and len(physical)==6
    assert len({(r['family'],r['dataset'],r['variant']) for r in screens})==122
    assert sum(r['exact_control_reproduced']=='True' for r in screens)==8
    for r in screens:
        path=ROOT/r['source'];x=read(path)
        for key in ['retained_mR','common_KL','worst_KL']:assert float(r[key])==x[key]
        assert sha(path.parent/'final.pt')==r['checkpoint_sha256']
    for family,path in report.FAMILIES.items():
        protocol=read(path/'protocol.json')
        source=dict(first='control_v1',context='context_v2',focused='focused_v3',loss='loss_v4')[family]
        assert sha(ROOT/'tools/uav_convergence_v2'/('joint_quality_'+source+'.py'))==protocol['source_sha256']
    transactions=0
    for r in physical:
        arm=ROOT/r['source'];assert sha(arm/'final.pt')==r['checkpoint_sha256']
        costs=read(arm/'energy_and_delay.json');verify=read(arm/'verification.json')
        assert verify['passed'] and verify['FL_rounds']==200 and verify['FU_steps']==225
        assert costs['network_cap_overrun_transactions']==0
        assert np.isclose(costs['total_energy_j'],sum(costs[k] for k in ['energy_computing_j','energy_communication_j','energy_hover_j']))
        for txpath in sorted((arm/'transactions').glob('*.json')):
            tx=read(txpath);radio=tx['radio'];transactions+=1
            assert radio['status']=='feasible' and radio['max_power_w']<=.32+1e-6
            assert radio['energy_bound_j']<=.1+1e-6
            if radio.get('access')=='zero_update_silence':
                assert radio['conditional_rmse']==0 and radio['FL_norm_bound']==0 and radio['FU_norm_bound']==0
            else:assert radio['error_bound']<=radio['epsilon']*(1+1e-5)+1e-9
            assert tx['end_s']>tx['start_s']
        # Every evaluation uses the complete original gallery and all native captions.
        plan=read(arm.parent/'plan.json')
        dest=audit.OUT/r['dataset'].lower()/'physical'/r['stage']/r['variant']
        per=np.load(dest/'per_video.npz');assert per['video_ids'].tolist()==plan['gallery_images']
        assert len(read(dest/'request_kl.json'))==45
        idx=study.audit.indices(plan,'forget');ret=study.audit.indices(plan,'retained')
        assert abs(float(r['common_KL'])-per['common_KL'][idx].mean())<1e-12
        assert abs(float(r['retained_mR'])-100*per['mR'][ret].mean())<1e-12
    assert transactions==1080
    improvements=read(report.DEST/'data/improvements.json')
    assert all(r['leads_three_baselines_mR'] and not r['leads_three_baselines_common_KL'] for r in improvements)
    assert all(r['common_KL_reduction_percent']>40 for r in improvements)
    assert all(r['loss_change']>0 for r in improvements)
    for ds in study.audit.DATASETS:
        data=read(audit.OUT/ds.lower()/'summary.json')['paired']
        for key in ['common_KL','matched_KL','mR']:
            assert data['exact_fu']['metrics']['forget'][key]>data['exact_no_fu']['metrics']['forget'][key]
    geometry=read(report.DEST/'data/geometry.json');assert len(geometry)==6
    for figure in geometry:
        assert figure['canvas_in']==[7.4,3.735]
        for panel in figure['panels']:
            assert np.isclose(panel['width_pt'],205.128)
            assert np.isclose(panel['height_pt'],174.798)
            assert min(panel['category_gaps_pt'])>5
        for ext in ['pdf','png','svg']:assert (report.DEST/(figure['figure']+'.'+ext)).stat().st_size>1000
    for path,digest in read(report.DEST/'data/provenance.json').items():assert sha(ROOT/path)==digest,path
    result=dict(passed=True,unchanged_input_files=len(immutable),exact_runs=len(screens),
        reproduced_controls=8,physical_runs=6,checked_radio_transactions=transactions,figures=6,
        MR_lead_preserved_both=True,common_KL_lead=False,FU_benefit_established=False)
    study.write(report.DEST/'data/verification.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
