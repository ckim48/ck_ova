"""Replay the frozen retraining data protocol without training or inference."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True)
    root=p.parse_args().project.resolve();suite=root/'results/comparisons_seed0_solver_v2'
    source=suite/'coupled_hybrid';reference=suite/'retrain_reference_ideal'
    expected=json.loads((suite/'suite_metadata.json').read_text())['source_sha256']
    assert all(sha(root/n)==h for n,h in expected.items())
    manifest=json.loads((reference/'manifest.json').read_text());assert manifest['status']=='completed'
    assert manifest['algorithm']=='retrain_reference' and manifest['transport']=='ideal'
    assert manifest['source_sha256']==expected
    c=manifest['config'];assert c==json.loads((suite/'config.json').read_text())
    assert sha(reference/'initial.pt')==sha(source/'initial.pt')
    assert sha(reference/'schedule.json')==sha(source/'schedule.json')
    schedule=json.loads((source/'schedule.json').read_text());deleted=set(schedule[-1]['deleted'])
    original_counts={i:0 for i in deleted};reference_counts={i:0 for i in deleted}
    reference_seen=set();original_seen=set();draws=0;max_batch=0;minimum_retained=None
    rows=[]
    for rec in schedule:
        retained=[sorted(set(ids)-deleted) for ids in rec['retained']]
        batches=[np.random.default_rng(seed).choice(ids,min(c['batch_size'],len(ids)),replace=False).tolist()
                 for ids,seed in zip(retained,rec['batch_seeds'])]
        count=sum(map(len,retained));assert count>0
        assert abs(sum(len(ids)/count for ids in retained)-1)<1e-12
        for original,ref,allowed in zip(rec['batches'],batches,retained):
            original_seen.update(original);reference_seen.update(ref);draws+=len(ref)
            assert set(ref)<=set(allowed) and not set(ref)&deleted
            assert len(ref)==min(c['batch_size'],len(allowed)) and len(ref)==len(set(ref))
            max_batch=max(max_batch,len(ref));minimum_retained=min(minimum_retained,len(allowed)) if minimum_retained is not None else len(allowed)
            for i in original:
                if i in original_counts:original_counts[i]+=1
            for i in ref:
                if i in reference_counts:reference_counts[i]+=1
        rows.append({'round':rec['round'],'retained_population':count,'reference_batch_draws':sum(map(len,batches)),
                     'deleted_ids_in_replayed_reference_batches':sum(i in deleted for batch in batches for i in batch)})
    assert all(x>0 for x in original_counts.values()) and not any(reference_counts.values())
    result={'at_kst':datetime.now().isoformat(timespec='seconds'),'status':'deterministic_protocol_replay_passed',
            'final_deleted_samples':len(deleted),'deleted_samples_used_in_original_schedule':sum(x>0 for x in original_counts.values()),
            'deleted_samples_used_in_replayed_retraining':sum(x>0 for x in reference_counts.values()),
            'original_deleted_use_count_median':float(np.median(list(original_counts.values()))),
            'original_deleted_use_count_max':max(original_counts.values()),
            'reference_batch_draws':draws,'reference_unique_samples':len(reference_seen),'original_unique_samples':len(original_seen),
            'reference_max_batch':max_batch,'minimum_client_retained_population':minimum_retained,
            'rounds':rows,'source_sha256':expected,'schedule_sha256':sha(source/'schedule.json'),
            'reference_manifest_sha256':sha(reference/'manifest.json'),'script_sha256':sha(__file__),
            'limits':['Replays deterministic minibatch construction in the frozen source; no independent runtime sample trace was recorded.',
                      'All final deleted IDs are excluded in every replayed reference round, including before request arrival.',
                      'Reference resamples retained populations using original batch seeds; it is one retraining realization, not a distributional certificate.',
                      'Positive prior exposure does not establish memorization or certified forgetting. No inference/training/test-set selection performed.']}
    out=root/'results/overnight_energy_v2/reference_protocol_audit_v1';out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['rounds','source_sha256']}))


if __name__=='__main__':main()
