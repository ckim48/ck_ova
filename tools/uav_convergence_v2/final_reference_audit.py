"""Evaluate the existing pre-FU and retrained checkpoints without changing them."""
import fcntl
import math
from pathlib import Path
from final_request_sweep import ROOT, OUT, BASELINE, DATASETS, read, write, sha, load_base
from airscape import VideoData, extra


def run():
    import torch
    torch.set_num_threads(1)
    assert torch.cuda.is_available()
    lock=(ROOT/"results/overnight_energy_v2/association_objective_gpu.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX)
    sources={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))}
    event,base=load_base();rows=[]
    try:
        for dataset in DATASETS:
            directory=BASELINE/"cases/baseline"/(dataset.lower()+"_seed0")
            for name in ["plan.json","initial20/final.pt","initial20/rounds.json","reference200/final.pt",
                         "reference_metrics.json","reference_mia.json","feature_cache/features.pt"]:
                p=directory/name;sources[str(p.relative_to(ROOT))]=sha(p)
            plan=read(directory/"plan.json")
            data=event.recover.Data(dataset) if dataset=="CapERA" else VideoData(dataset)
            Engine=base.Engine if dataset=="CapERA" else extra.native_engine(base,len(data.caption_ids[0]))
            payload=torch.load(directory/"feature_cache/features.pt",map_location="cpu",weights_only=False)
            engine=Engine(data,plan,payload)
            reference=base.core.load_theta(directory/"reference200/final.pt",engine.device)
            ref_eval,ref_prob=engine.evaluate(reference,None)
            ref_mia,_=engine.mia_scores(reference)
            expected_ref=read(directory/"reference_metrics.json")["metrics"]
            for metric in ["loss","mR"]:assert math.isclose(ref_eval["retained"][metric],expected_ref["retained"][metric],abs_tol=2e-7)
            assert math.isclose(ref_mia["loss_score_auc"],read(directory/"reference_mia.json")["loss_score_auc"],abs_tol=1e-12)
            warm=base.core.load_theta(directory/"initial20/final.pt",engine.device)
            warm_eval,_=engine.evaluate(warm,ref_prob);warm_mia,_=engine.mia_scores(warm)
            expected=read(directory/"initial20/rounds.json")[-1]["evaluation"]
            for group in ["retained","forget"]:
                for metric in ["loss","mR"]:assert math.isclose(warm_eval[group][metric],expected[group][metric],abs_tol=2e-7)
            rows.append(dict(dataset=dataset,stage="pre_FU_round20",evaluation=warm_eval,mia=warm_mia,
                             MIA_AUC_gap_vs_retrain=abs(warm_mia["loss_score_auc"]-ref_mia["loss_score_auc"])))
            rows.append(dict(dataset=dataset,stage="retained_set_retrain_round200",evaluation=ref_eval,mia=ref_mia,
                             MIA_AUC_gap_vs_retrain=0.))
            del engine,data,payload,reference,warm,ref_prob
            import gc;gc.collect();torch.cuda.empty_cache()
    finally:base.base.parent.common.base.close()
    for p,digest in sources.items():assert sha(ROOT/p)==digest,p
    write(OUT/"pre_fu_reference.json",dict(passed=True,rows=rows,source_sha256=sources,
        scope="Read-only evaluation of existing warm20 and reference200 checkpoints. Warm20 is the pre-FU reference, not a 200-round no-unlearning baseline. No training, tuning or checkpoint modification."))
    print(dict(passed=True,evaluated_checkpoints=len(rows)),flush=True)


if __name__=="__main__":run()
