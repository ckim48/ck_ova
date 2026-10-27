"""Independent cross-method input, provenance, cost and pipeline audit."""
import argparse
import collections
import math
from pathlib import Path
import numpy as np
import paper_uav_charts as charts
from adaptive_fu import ROOT,read,write,sha


def audit(complete=True):
    study=charts.study
    protocols=[study.register(),charts.proposed.register(),charts.timing.register()]
    for protocol in protocols:
        for name,digest in protocol['source_sha256'].items():assert sha(ROOT/name)==digest,name
    execution=study.OUT/'execution_batching.json'
    if execution.exists():
        e=read(execution);assert sha(ROOT/e['source'])==e['sha256']
    parallel=study.OUT/'execution_parallel.json'
    if parallel.exists():
        e=read(parallel);assert sha(ROOT/e['source'])==e['sha256']
        assert sha(ROOT/e['batch_source'])==e['batch_sha256']
    if complete:
        fixed=charts.correction.register();verified=read(charts.correction.OUT/'verification.json');assert verified['passed']
        assert sha(Path(charts.correction.__file__))==verified['correction_source_sha256']
    old=study.lib.adaptive.register()
    for name,digest in old['input_sha256'].items():assert sha(ROOT/name)==digest,name
    cache={};rows=[];input_hashes={}
    def digest(path):
        key=str(path.resolve())
        if key not in cache:cache[key]=sha(path)
        input_hashes[str(path.relative_to(ROOT))]=cache[key]
        return cache[key]
    for c in (study.cases() if complete else [dict(id='baseline')]):
        for ds in study.DATASETS:
            paths=[charts.source(c['id'],ds,m) for m in charts.METHODS]
            expected=paths[0]
            for arm in paths:
                for name in ['plan.json','requests.json','initial20/final.pt','feature_cache/features.pt','reference200/final.pt','reference_metrics.json','reference_mia.json']:
                    assert digest(arm.parent/name)==digest(expected.parent/name),(arm,name)
                assert read(arm.parents[1]/'settings.json')['energy_model']==read(expected.parents[1]/'settings.json')['energy_model']
                metric=charts.ledger(c['id'],ds,arm.name)
                ev=read(arm/'manifest.json')['evaluation'];assert metric['retained_mR_percent']==100*ev['retained']['mR']
                row=dict(case=c['id'],dataset=ds,method=arm.name,inputs_match=True)
                if arm.name in study.METHODS:
                    plan=read(arm.parent/'plan.json');requests=read(arm.parent/'requests.json');geo=read(arm.parents[1]/'mission_spec.json')
                    v,tl,tx=study.validate(arm,requests,geo,plan['config'],.1)
                    original=read(arm/'verification.json');assert v==original
                    energy=read(arm/'energy_and_delay.json')
                    for k in ['total_energy_j','delay_s']:assert math.isclose(metric[k],energy[k],rel_tol=1e-11)
                    jobs={r['job']:r for r in [read(p) for p in (arm/'jobs').glob('*.json')]}
                    for t in tx:
                        if t['kind']!='fl' or arm.name!='uav_afl_ota_gaft':continue
                        m=t['radio'];tx_start=t['start_s']+t['timing']['compute_s']
                        for j in m['selected_job_ids']:
                            job=jobs[j];assert job['ready_s']<=tx_start+1e-8
                            assert t['sequence']-1-job['model_version']==m['staleness'][str(job['user'])]
                            intervening=[z for z in tx if z['kind']=='fu' and job['model_version']<z['sequence']<t['sequence']]
                            assert not intervening,'Pre-FU model update reapplied'
                    if arm.name=='uav_afl_ota_gaft':
                        job_macs=np.zeros(10)
                        for j in jobs.values():job_macs[j['user']]+=j['MACs']
                        total=np.sum([read(p)['compute']['device_macs'] for p in (arm/'work_attempts').glob('*.json')],axis=0)
                        assert np.all(total>=job_macs)
                        row.update(max_model_version_lag=v['max_staleness'],local_jobs=v['local_jobs'])
                rows.append(row)
    result=dict(passed=True,complete=complete,compared_arms=len(rows),rows=rows,input_sha256=input_hashes,
        source_sha256={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))},
        checks=['All methods share byte-identical plans/requests/features/warm/reference inputs per condition',
            'Registered source and inherited data hashes unchanged','Plot ledger agrees with executed jobs and actual critical-path clocks',
            'Every applied async cache is ready and no pre-FU cache is reused','Paper transports respect power and analytical error bounds; QoS overruns retained'])
    write(study.OUT/('comparison_audit.json' if complete else 'baseline_audit.json'),result)
    print(dict(passed=True,complete=complete,compared_arms=len(rows)),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--baseline-only',action='store_true');a=p.parse_args();audit(not a.baseline_only)
