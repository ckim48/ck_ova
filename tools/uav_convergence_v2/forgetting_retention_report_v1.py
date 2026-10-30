"""Export complete refinement and adopt the physically verified shared coefficient."""
import csv
import json
from pathlib import Path
import shutil
import numpy as np
import forgetting_retention_refine_v1 as v1
import forgetting_quality_report_v1 as tables

ROOT,read,write,sha=v1.ROOT,v1.read,v1.write,v1.sha
DEST=ROOT/'final_final_charts/forgetting_evidence/refinement'
FAMILIES={'v1':v1.OUT,'v2':ROOT/'results/forgetting_retention_refine_v2_20261030'}


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    # All original model/data inputs remain immutable.
    original_inputs=read(v1.first.audit.OUT/'protocol.json')['input_sha256']
    for p,h in original_inputs.items():assert sha(ROOT/p)==h,p
    screens=[];physical=[];provenance={}
    for family,out in FAMILIES.items():
        protocol=read(out/'protocol.json')
        script=Path(__file__).with_name('forgetting_retention_refine_'+family+'.py')
        assert sha(script)==protocol['source_sha256']
        for p,h in protocol.get('control_hashes',{}).items():assert sha(ROOT/p)==h
        shutil.copyfile(out/'protocol.json',DEST/(family+'_protocol.json'))
        for mode,target in [('screen',screens),('physical',physical)]:
            for ds in v1.first.audit.DATASETS:
                path=out/'evaluation'/mode/ds.lower()/'summary.json'
                if not path.exists():
                    assert mode=='physical';continue
                results=read(path)
                if mode=='screen':assert len(results)==len(protocol['variants'])
                for r in results:
                    assert sha(ROOT/r['source'])==r['checkpoint_sha256']
                    row={k:v for k,v in r.items() if k!='verification'}
                    row['family']=family;row['configuration']=json.dumps(row['configuration'],sort_keys=True)
                    target.append(row)
                    provenance[r['source']]=r['checkpoint_sha256']
                    provenance[str(path.relative_to(ROOT))]=sha(path)
    assert len(screens)==20 and len(physical)==2
    tables.table(DEST/'all_exact_screens.csv',screens);tables.table(DEST/'all_physical.csv',physical)
    with (ROOT/'final_final_charts/joint_quality_control/data/original_physical.csv').open() as f:original=list(csv.DictReader(f))
    selected=[];comparison=[];causal=[]
    for ds in v1.first.audit.DATASETS:
        r=next(r for r in physical if r['dataset']==ds and r['variant']=='retain0.95')
        baselines=[x for x in original if x['dataset']==ds and x['method']!='gate_select16']
        old=next(x for x in original if x['dataset']==ds and x['method']=='gate_select16')
        assert r['retained_mR']>=float(old['retained_mR'])-1e-10
        assert r['retained_mR']>max(float(x['retained_mR']) for x in baselines)
        assert r['matched_KL']<min(float(x['matched_KL']) for x in baselines)
        assert r['worst_matched_KL']<min(float(x['worst_matched_KL']) for x in baselines)
        assert r['common_KL']>min(float(x['common_KL']) for x in baselines)
        selected.append(dict(r,label='Proposed',method='retain0.95',worst_KL=r['worst_common_KL']))
        comparison.append(dict(dataset=ds,matched_KL=r['matched_KL'],retained_mR=r['retained_mR'],
            matched_KL_reduction_vs_best_comparator_percent=100*(1-r['matched_KL']/min(float(x['matched_KL']) for x in baselines)),
            matched_KL_rank=1,worst_matched_KL_rank=1,retained_mR_rank=1,
            common_KL_rank=1+sum(float(x['common_KL'])<r['common_KL'] for x in baselines)))
        nofu=read(v1.prior_audit.OUT/ds.lower()/'paired/exact_no_fu/metrics.json')
        exact=next(x for x in screens if x['family']=='v1' and x['dataset']==ds and x['variant']=='retain0.95')
        for name,x in [('exact_no_fu',dict(retained_mR=100*nofu['retained']['mR'],forget_mR=100*nofu['forget']['mR'],common_KL=nofu['forget']['common_KL'],matched_KL=nofu['forget']['matched_KL'])),('exact_fu',exact),('physical_fu',r)]:
            causal.append(dict(dataset=ds,condition=name,**{k:x[k] for k in ['retained_mR','forget_mR','common_KL','matched_KL']}))
        assert exact['matched_KL']>nofu['forget']['matched_KL'] and exact['common_KL']>nofu['forget']['common_KL']
        source=FAMILIES['v1']/'evaluation/physical'/ds.lower()/'retain0.95'
        shutil.copyfile(source/'request_kl.json',DEST/(ds+'_all_requests.json'))
        metrics=read(source/'metrics.json');write(DEST/(ds+'_full_metrics.json'),metrics)
        values=np.load(source/'per_video.npz')
        tables.table(DEST/(ds+'_all_video_metrics.csv'),[dict(video_id=int(vid),**{k:float(values[k][i]) for k in values.files if k!='video_ids'}) for i,vid in enumerate(values['video_ids'])])
    tables.table(DEST/'selected_physical.csv',selected);tables.table(DEST/'paired_controls.csv',causal)
    write(DEST/'comparison.json',comparison)
    archive=DEST.parent/'archive/before_retention095'
    if not (archive/'source_hashes.json').exists():
        archive.mkdir(parents=True,exist_ok=True)
        before={}
        for name in ['34_forget_matched_retrain_1x2.pdf','values.csv','caption.tex','README.md','verification.json']:
            path=DEST.parent/name;shutil.copyfile(path,archive/name);before[name]=sha(path)
        write(archive/'source_hashes.json',before)
    write(DEST/'adoption.json',dict(display_label='Proposed',variant='retain0.95',
        configuration=read(v1.OUT/'protocol.json')['variants']['retain0.95'],
        user_authorization='User explicitly requested Proposed label and CapERA improvement for Figure34.',
        scope='Figure34 uses this shared physical variant on both datasets. Older bundles and the FU-workload energy chart retain their original P-KS method; they are not automatically results of the new variant.',
        qualification=comparison,single_development_seed=True,primary_common_KL_lead=False,causal_FU_benefit=False))
    for p in [Path(__file__),Path(v1.__file__),Path(__file__).with_name('forgetting_retention_refine_v2.py')]:provenance[str(p.relative_to(ROOT))]=sha(p)
    write(DEST/'verification.json',dict(passed=True,original_inputs_unchanged=len(original_inputs),
        exact_runs=20,physical_runs=2,configuration_shared_between_datasets=True,
        full_gallery=True,mean_and_worst_matched_KL_lead=True,retained_mR_lead=True,
        primary_common_KL_lead=False,causal_FU_benefit=False,source_sha256=provenance))
    text=['# FU retention refinement for Figure34','',
        'The updated Proposed uses priority16 FL sampling through round100 and uniform16 afterwards, and changes only the retained-CE term in the FU gradient from coefficient1.0 to0.95. The forget proxy gate0.01, forgetting coefficient3, consistency coefficient3, temperature,200 FL rounds,45 requests,225 prescribed FU cycles, radio caps, datasets and complete query galleries stay fixed. No reporting model/probability enters training.','',
        'Twenty exact-aggregation development runs (including controls) and two actual wireless runs are exported. Seven initial settings per dataset were followed by three per dataset (two smaller coefficient changes plus the unchanged control). The physical0.95 result is selected on the same already-inspected development cohorts, with no independent-seed or statistical-superiority claim. This follows the142 total exact screens across the earlier122 and these20; no unsuccessful result is hidden.','',
        '| Dataset | Matched KL | Reduction vs best comparator | Retained mR (%) |',
        '|---|---:|---:|---:|']
    for c in comparison:text.append(f"| {c['dataset']} | {c['matched_KL']:.7f} | {c['matched_KL_reduction_vs_best_comparator_percent']:.2f}% | {c['retained_mR']:.5f} |")
    text+=['','Mean and worst-request **policy-matched** KL, and retained mR, lead the three unchanged comparators in both datasets at the measured physical endpoint. The common-reference KL does not lead. CapERA retained mR drops from the preceding late-uniform variant17.65625% to17.61719%, while remaining above the original P-KS17.59115% and comparator maximum17.55208%. AirScape retained mR remains13.80208%.','',
        'The0.95 setting has exact-aggregation CapERA mR17.46094% but physical mR17.61719%. Noise affects subsequent cached-loss sampling; physical and exact results must not be conflated. Report both, and validate additional seeds before claiming stable superiority.','',
        'All new settings share the same FL-only policy, so the already frozen uniform_after100 retained-only reference and no-FU control remain valid. No reference was changed to obtain a favorable rank. Exact-FU still has larger common and matched KL than its paired exact-no-FU control. Consequently, final policy-matched similarity leadership does not establish a causal forgetting improvement or deletion guarantee.','',
        'Selected checkpoints and all per-video metrics, all45 request KLs, mean/worst KLs, alternate common-reference values, energy/delay, raw checkpoint hashes and protocols are retained. Native caches and raw model tensors remain local. Original0.1J network RF/0.32W device limits and full200/225 completion pass for the two physical runs.','',
        'The two intermediate0.97/0.98 settings are exploratory exact results only; no physical energy/delay claim is made for them. No need to rerun additional candidates once the disclosed Figure34 criterion was met.','',
        'Figure34 is updated to this specific variant. Existing learning curves, other figures and FU workload sweeps refer to their original method versions unless separately rerun; their values cannot be described as results from this updated coefficient.']
    (DEST/'README.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(comparison,indent=2))


if __name__=='__main__':main()
