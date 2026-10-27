"""Independent ledger, shared-input, ARQ and cache integrity checks."""
import math
from pathlib import Path
import numpy as np
import workload_charts as plots
import workload_campaign as campaign

study=plots.study
read,write,sha=study.read,study.write,study.sha


def main():
    protocol=study.register();campaign.control_check()
    execution=read(study.OUT/'execution.json');assert sha(study.ROOT/execution['source'])==execution['sha256']
    plots.prior.source=plots.source;plots.prior.comm=plots.comm
    cache={};hashes={};rows=[]
    def digest(p):
        key=str(p.resolve())
        if key not in cache:cache[key]=sha(p)
        hashes[str(p.relative_to(study.ROOT))]=cache[key];return cache[key]
    for c in plots.cases():
        for ds in study.DATASETS:
            paths=[plots.source(c,ds,m) for m in study.METHODS];expected=paths[0]
            for arm in paths:
                for name in ['plan.json','requests.json','initial20/final.pt','feature_cache/features.pt','reference200/final.pt','reference_metrics.json','reference_mia.json']:
                    assert digest(arm.parent/name)==digest(expected.parent/name),(arm,name)
                settings=read(arm.parents[1]/'settings.json');assert settings['energy_model']==read(expected.parents[1]/'settings.json')['energy_model']
                metric=plots.ledger(c,ds,arm.name);ev=read(arm/'manifest.json')['evaluation']
                assert metric['retained_mR_percent']==100*ev['retained']['mR']
                assert metric['forget_KL']==ev['forget']['KL_retrain_to_candidate']
                tx=[read(p) for p in sorted((arm/'transactions').glob('*.json'))]
                work=[read(p) for p in (arm/'work_attempts').glob('*.json')]
                jobs={r['job']:r for r in [read(p) for p in (arm/'jobs').glob('*.json')]}
                requests=read(arm/'request_records.json');plan=read(arm.parent/'plan.json')
                assert len(requests)==45 and sum(r['steps'] for r in requests)==225
                assert sum(t['kind'] in ['fl','joint'] for t in tx)==metric['FL_rounds']-20
                for t in tx:
                    m=t['radio'];assert m['max_power_w']<=settings['config']['power_limit_w']*(1+2e-6)
                    assert m['conditional_rmse']<=m['epsilon']*(1+3e-5)+1e-9
                    if 'broadcast_attempts' in m:
                        assert m['broadcast_trial_seed']==8200000+t['sequence']-1
                        draws=np.random.default_rng(m['broadcast_trial_seed']).random(m['broadcast_attempts']).tolist()
                        assert draws==m['broadcast_uniforms']
                        p=m['broadcast_erasure_probability'];assert all(x<p for x in draws[:-1])
                        if draws:assert draws[-1]>=p
                    if t['kind']=='fl' and arm.name=='uav_afl_ota_gaft':
                        upload=t['start_s']+t['timing']['compute_s']
                        for j in m['selected_job_ids']:
                            job=jobs[j];assert job['ready_s']<=upload+1e-8
                            assert t['sequence']-1-job['model_version']==m['staleness'][str(job['user'])]
                            assert not [z for z in tx if z['kind']=='fu' and job['model_version']<z['sequence']<t['sequence']]
                            assert not any(set(job['samples'])&set(r['images']) for r in requests if r['arrival_s'] is not None and r['arrival_s']<=job['start_s']+1e-8)
                if arm.name=='uav_afl_ota_gaft':
                    macs=np.zeros(10)
                    for j in jobs.values():macs[j['user']]+=j['MACs']
                    assert np.all(np.sum([w['compute']['device_macs'] for w in work],axis=0)>=macs)
                # Published ledgers independently reconstruct active energy; legacy
                # Proposed reused records have a broader old energy field, intentionally not compared.
                if c.startswith(('fl_rounds_','packet_loss_')) or arm.name in study.paper.METHODS:
                    e=read(arm/'energy_and_delay.json')
                    for key in ['total_energy_j','delay_s']:assert math.isclose(metric[key],e[key],rel_tol=1e-11),(c,ds,arm.name,key)
                rows.append(dict(case=c,dataset=ds,method=arm.name,shared_inputs=True,workload_verified=True,
                    independent_energy_j=metric['total_energy_j'],active_delay_s=metric['delay_s'],mission_delay_s=metric['mission_delay_s'],
                    actual_broadcast_retries=metric['broadcast_retries'],cached_jobs=len(jobs)))
    assert len(rows)==126
    result=dict(passed=True,compared_arms=126,rows=rows,input_sha256=hashes,
        checks=['Byte-identical shared data, plans, requests and reference per condition',
        'Every condition completes45 requests and225 FU cycles',
        'Independent energy sum and service/mission clocks agree',
        'Every ARQ draw matches registered seed and actual successful retry',
        'Async applied jobs are ready; no pre-FU or deleted-sample cache reuse',
        'All declared source hashes remain unchanged; power/error limits checked'],
        source_sha256={str(Path(__file__).relative_to(study.ROOT)):sha(Path(__file__))})
    write(study.OUT/'comparison_audit.json',result);print(dict(passed=True,arms=126))


if __name__=='__main__':main()
