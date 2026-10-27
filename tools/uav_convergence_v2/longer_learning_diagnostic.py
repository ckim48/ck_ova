"""Predeclared continuation diagnostic; never substitute it for mission results."""
import fcntl
import math
import os
import time
from pathlib import Path

from pilot import ROOT, OUT, load_base, read, write, sha

DEST = ROOT / "results/uav_longer_learning_diagnostic_20261027"
CONTINUATION_LR = .001
FINAL_ROUND = 1000


def main():
    import torch
    from airscape import VideoData, extra
    from common import SOURCE

    assert torch.cuda.is_available()
    assert not (DEST / "started.json").exists(), "Preserve every diagnostic attempt"
    write(DEST / "protocol.json", dict(
        datasets=["CapERA", "AirScape"], seed=0, initial_rounds=200, final_round=FINAL_ROUND,
        schedule="Rounds 1–200 reproduce the original cosine schedule; rounds 201–1000 use fixed LR 0.001.",
        scope="Ideal aggregation, FL only, no FU, no radio or mission timing; not Proposed or a method comparison.",
        controls="Same features, captions, splits, gallery, partitions, initial vector and per-round batch generator. "
                 "The first 200 rounds must reproduce the existing independent FL-only pilot.",
        interpretation="Tests additional learning with a restarted nonzero learning rate, not the causal effect of round count alone. "
                       "All checkpoints and both dataset outcomes are retained. No best-checkpoint selection.",
        evaluation_every_rounds=20, source_sha256=sha(Path(__file__))))
    write(DEST / "started.json", dict(pid=os.getpid(), unix=time.time()))
    lock = (ROOT / "results/overnight_energy_v2/association_objective_gpu.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    event, base = load_base()
    try:
        from data_schedule import make_schedule
        from train import update_theta
        torch.set_num_threads(1)
        results = {}
        for dataset in ["CapERA", "AirScape"]:
            prior = OUT / (dataset.lower() + "_pilot")
            src = SOURCE / "capera_seed0" if dataset == "CapERA" else prior
            plan = read(src / "plan.json")
            data = event.recover.Data(dataset) if dataset == "CapERA" else VideoData(dataset)
            features = src / "feature_cache/features.pt"
            payload = torch.load(features, map_location="cpu", weights_only=False)
            engine = (base.Engine if dataset == "CapERA" else extra.native_engine(base, 1))(data, plan, payload)
            records = make_schedule(plan["partitions"], rounds=FINAL_ROUND, initial_rounds=20,
                                    requests=(), initial_fraction=1., batch_size=16, seed=0)
            original = make_schedule(plan["partitions"], rounds=200, initial_rounds=20,
                                     requests=(), initial_fraction=1., batch_size=16, seed=0)
            assert records[:200] == original
            expected = {r["round"]: r["evaluation"] for r in read(prior / "rounds.json")}
            theta = engine.theta0.clone()
            evaluation, _ = engine.evaluate(theta, None)
            rows = [dict(round=0, learning_rate=None, evaluation=evaluation)]
            folder = DEST / dataset.lower()
            write(folder / "inputs.json", dict(plan_sha256=sha(src / "plan.json"),
                features_sha256=sha(features), pilot_sha256=sha(prior / "rounds.json"),
                training_items=sum(map(len, plan["partitions"])), gallery_items=len(plan["gallery_images"]),
                validation_queries=len(plan["validation_images"])))
            for record in records:
                k = record["round"]
                lr = base.core.lr_at(k) if k <= 200 else CONTINUATION_LR
                engine.plan["learning_rate"] = lr
                learning, _, _ = engine.updates(theta, record, k, "retrain", [[] for _ in range(10)])
                theta = update_theta(theta, learning.sum(0))
                if k % 20 == 0:
                    evaluation, _ = engine.evaluate(theta, None)
                    if k <= 200:
                        for key, value in evaluation["retained"].items():
                            assert math.isclose(value, expected[k]["retained"][key], abs_tol=2e-7), (dataset, k, key)
                    rows.append(dict(round=k, learning_rate=lr, evaluation=evaluation))
                    write(folder / "rounds.json", rows)
                    write(DEST / "status.json", dict(dataset=dataset, round=k, pid=os.getpid(), unix=time.time()))
                    if k % 200 == 0:
                        print(dataset, k, evaluation["retained"]["mR"] * 100, evaluation["retained"]["loss"], flush=True)
            before = next(r["evaluation"]["retained"] for r in rows if r["round"] == 200)
            after = rows[-1]["evaluation"]["retained"]
            results[dataset] = dict(round200=before, round1000=after,
                mR_gain_percentage_points=100 * (after["mR"] - before["mR"]),
                last_five_mean_mR=sum(r["evaluation"]["retained"]["mR"] for r in rows[-5:]) / 5,
                original_200_rounds_reproduced=True)
            write(folder / "result.json", results[dataset])
            del engine, payload, theta
            torch.cuda.empty_cache()
        write(DEST / "finished.json", dict(status="completed", unix=time.time(), results=results,
            scope=read(DEST / "protocol.json")["scope"]))
        print(results, flush=True)
    finally:
        base.base.parent.common.base.close()


if __name__ == "__main__":
    main()
