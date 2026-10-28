"""Evaluate every retained query at exactly recovered energy-limited models."""
import fcntl
import math
from pathlib import Path
import time

import numpy as np
import limited_energy_checkpoints_v1 as checkpoint
import tgcn_retrieval_examples_v1 as retrieval

ROOT, read, write, sha = checkpoint.ROOT, checkpoint.read, checkpoint.write, checkpoint.sha
OUT = ROOT/'results/limited_energy_retrieval_v1_20261028'
METHODS, DATASETS = checkpoint.current.METHODS, checkpoint.current.DATASETS


def run():
    import torch
    from airscape import extra
    protocol = checkpoint.register()
    assert read(checkpoint.OUT/'finished.json')['passed']
    assert not (OUT/'started.json').exists(), 'Preserve previous inference attempts.'
    assert torch.cuda.is_available()
    torch.set_num_threads(1)
    source_map = {(r['dataset'],r['method']):ROOT/r['arm'] for r in protocol['original_sources']}
    with (ROOT/'results/overnight_energy_v2/association_objective_gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        write(OUT/'started.json',dict(unix=time.time()))
        summaries, input_hashes = [], {}
        for ds in DATASETS:
            data,base = retrieval.data_for(ds)
            arm = source_map[(ds,'gate_select16')]
            plan = read(arm.parent/'plan.json')
            gallery,validation = plan['gallery_images'],set(plan['validation_images'])
            assert not validation & set(plan['forgotten_images'])
            count = len(data.caption_ids[0])
            feature = arm.parent/'feature_cache/features.pt'
            payload = torch.load(feature,map_location='cpu',weights_only=False)
            Engine = base.Engine if ds=='CapERA' else extra.native_engine(base,count)
            engine = Engine(data,plan,payload)
            caption_ids = [cid for vid in gallery for cid in data.caption_ids[vid]]
            queries = [(ci,cid,gallery[ci//count]) for ci,cid in enumerate(caption_ids) if gallery[ci//count] in validation]
            previous = read(ROOT/'final_final_charts/downlink_bandwidth/data'/f'retrieval_energy_{ds.lower()}_query_ranks.json')
            assert [(r['gallery_caption_index'],r['caption_id'],r['ground_truth_video']) for r in previous]==queries
            for p in [feature,arm.parent/'plan.json']:
                input_hashes[str(p.relative_to(ROOT))]=sha(p)
            for budget in checkpoint.BUDGETS:
                rows = [dict(dataset=ds,budget_kJ=budget,caption_id=cid,gallery_caption_index=ci,ground_truth_video=vid)
                        for ci,cid,vid in queries]
                arrays = {}
                for method,label in METHODS.items():
                    dest=checkpoint.snapshot(ds,method,budget)
                    meta=read(dest/'metadata.json')
                    assert sha(dest/'model.pt')==meta['checkpoint_sha256']
                    input_hashes[str((dest/'model.pt').relative_to(ROOT))]=meta['checkpoint_sha256']
                    assert sha(source_map[(ds,method)].parent/'plan.json')==sha(arm.parent/'plan.json')
                    theta=base.core.load_theta(dest/'model.pt',engine.device)
                    with torch.no_grad():
                        actual,_=engine.evaluate(theta,None)
                        retained=actual['retained']
                        expected=meta['expected_evaluation']
                        if expected is not None:
                            for key in ['mR','loss','i2t_R1','i2t_R5','i2t_R10','t2i_R1','t2i_R5','t2i_R10']:
                                assert math.isclose(retained[key],expected['retained'][key],abs_tol=2e-7),(ds,method,budget,key)
                        score=engine.score(theta,gallery).cpu()
                        assert tuple(score.shape)==(len(gallery),len(gallery)*count)
                        order=torch.argsort(score.T,dim=1,descending=True)
                        ranks=torch.argsort(order,dim=1)+1
                        for row,(ci,cid,vid) in zip(rows,queries):
                            row[method+'_rank']=int(ranks[ci,ci//count])
                        observed=np.array([r[method+'_rank'] for r in rows])
                        for k in [1,5,10]:
                            assert math.isclose(float((observed<=k).mean()),retained[f't2i_R{k}'],abs_tol=2e-7)
                        arrays[method]=score.numpy()
                        summary={k:meta[k] for k in ['dataset','method','budget_kJ','stage','sequence','FL_round',
                            'FU_cycles','FU_arrived','FU_completed','energy_kJ','active_s','next_completion_energy_kJ']}
                        summary.update(label=label,retained_mR_percent=100*retained['mR'],retained_loss=retained['loss'],
                            median_GT_rank=float(np.median(observed)),mean_GT_rank=float(observed.mean()),
                            q1_GT_rank=float(np.quantile(observed,.25)),q3_GT_rank=float(np.quantile(observed,.75)),
                            text_queries=len(rows),gallery_videos=len(gallery),
                            reproduced_original_evaluation=expected is not None,
                            **{k:retained[k] for k in ['i2t_R1','i2t_R5','i2t_R10','t2i_R1','t2i_R5','t2i_R10']})
                        summaries.append(summary)
                write(OUT/f'{budget}kJ'/f'{ds.lower()}_query_ranks.json',rows)
                np.savez_compressed(OUT/f'{budget}kJ'/f'{ds.lower()}_scores.npz',**arrays)
                print(dict(dataset=ds,budget_kJ=budget,models=4,queries=len(rows)),flush=True)
            del data,engine,payload,theta,score,order,ranks,arrays
            torch.cuda.empty_cache()
        write(OUT/'summary.json',summaries)
        write(OUT/'finished.json',dict(passed=True,models=len(summaries),budgets_kJ=checkpoint.BUDGETS,
            original_query_ids_and_gallery_preserved=True,all_ranks_from_actual_scores=True,
            source_sha256={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))},input_sha256=input_hashes))


if __name__=='__main__':run()
