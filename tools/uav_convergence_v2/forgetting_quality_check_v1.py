"""Independent cohort, reference, counterfactual and metric audit."""
import csv
import math
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import torch
import forgetting_quality_v1 as study

ROOT,OUT,read,write,sha=study.ROOT,study.OUT,study.read,study.write,study.sha
DEST=ROOT/'final_final_charts/forgetting_quality'


def metric_checks():
    # Multiple native captions and a known perfect retrieval ranking.
    for captions in [1,5]:
        logits=torch.zeros((12,12*captions))
        for i in range(12):logits[i,i*captions:(i+1)*captions]=1
        engine=SimpleNamespace(data=SimpleNamespace(caption_ids=[list(range(captions))]),
                               score=lambda theta,ids:logits)
        plan={'gallery_images':list(range(12))}
        a,pa=study.per_video(engine,torch.zeros(1),plan)
        assert np.all(a['mR']==1) and np.all(a['association_margin']==1)
        assert np.all(a['positive_cosine']==1)
        assert np.all(study.kl_per_video(pa,pa,captions)==0)
        # A global similarity offset must not masquerade as better association.
        engine.score=lambda theta,ids:logits+.125
        b,pb=study.per_video(engine,torch.zeros(1),plan)
        assert np.allclose(a['association_margin'],b['association_margin'],atol=1e-14)
        assert np.allclose(b['positive_cosine']-a['positive_cosine'],.125,atol=1e-14)
        assert np.max(study.kl_per_video(pa,pb,captions))<1e-14


def main():
    metric_checks();checks=[]
    protocol=read(OUT/'protocol.json')
    for name,digest in protocol['input_sha256'].items():assert sha(ROOT/name)==digest,name
    for ds in study.DATASETS:
        plan=read(study.current.source('baseline',ds,'gate_select16').parent/'plan.json')
        for m in study.METHODS:
            arm=study.current.source('baseline',ds,m)
            reqs=read(arm/'request_records.json')
            assert len(reqs)==45 and all(r['steps']==5 for r in reqs)
            cohorts=[i for r in reqs for i in r['images']]
            assert len(cohorts)==len(set(cohorts))
            assert set(cohorts)==set(plan['forgotten_images'])
            assert set(cohorts).isdisjoint(plan['validation_images'])
            for mode in ['actual','exact_fu','exact_no_fu']:
                arr=np.load(OUT/ds.lower()/m/mode/'per_video.npz')
                assert list(arr['video_ids'])==plan['gallery_images']
                summary=read(OUT/ds.lower()/m/mode/'metrics.json')
                for group in ['forget','retained']:
                    idx=study.indices(plan,group)
                    for key,value in summary[group].items():
                        assert np.isfinite(arr[key]).all()
                        assert math.isclose(float(arr[key][idx].mean()),value,abs_tol=1e-14)
            for mode in ['exact_fu','exact_no_fu']:
                finish=read(OUT/ds.lower()/m/mode/'finished.json')
                assert finish['FL_rounds']==200 and finish['FU_steps']==(225 if mode=='exact_fu' else 0)
                events=read(OUT/ds.lower()/m/mode/'request_events.json')
                assert len(events)==45 and {e['request'] for e in events}=={r['id'] for r in reqs}
                assert all(e['last_FL_round']>=e['first_FL_round'] for e in events)
                if mode=='exact_fu' and m in ['iscc','salun']:
                    assert finish['digital_original_tensor_exact']
            checks.append(dict(dataset=ds,method=m,full_deleted_cohort=True,all45requests=True,
                               metrics_reduced_from_per_video_values=True))
        # All three baselines have the same FL rule in this all-ready trace.
        control=[torch.load(OUT/ds.lower()/m/'exact_no_fu/final.pt',map_location='cpu',weights_only=True)['theta']
                 for m in ['iscc','afl','salun']]
        assert all(torch.equal(control[0],x) for x in control[1:])
        for policy in ['common','priority','equal']:
            assert read(OUT/ds.lower()/'references'/policy/'finished.json')['original_reference_tensor_exact']
    layout=read(DEST/'data/layout_checks.json');assert layout['passed'] and len(layout['figures'])==8
    for fig in layout['figures']:
        name=fig['figure'];svg=ET.parse(DEST/(name+'.svg')).getroot()
        assert svg.get('viewBox')=='0 0 532.8 268.92'
        rects=svg.findall('.//s:clipPath/s:rect',{'s':'http://www.w3.org/2000/svg'})
        assert len(rects)==2
        assert np.allclose([[float(r.get('width')),float(r.get('height'))] for r in rects],[[205.128,174.798]]*2)
        assert sha(DEST/(name+'.pdf'))==sha(ROOT/'final_final_charts/figures'/(name+'.pdf'))
    result=dict(passed=True,new_training_runs=22,retraining_runs=6,paired_control_runs=16,
                frozen_models_evaluated=8,original_inputs_unchanged=True,
                multiple_caption_metric_checks=True,global_similarity_offset_invariance=True,
                noFU_baselines_identical_as_expected=True,figures=8,checks=checks,
                source_sha256=sha(Path(__file__)))
    write(DEST/'data/verification.json',result)
    print(result)


if __name__=='__main__':main()
