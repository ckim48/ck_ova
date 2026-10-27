"""Matched ideal-link FL-only control using the existing untouched CapERA split."""
import fcntl
import os
import time
from pathlib import Path
from pilot import ROOT, OUT, load_base, read, write, sha


def main():
    import torch
    from common import SOURCE
    assert torch.cuda.is_available()
    lock = (ROOT / "results/overnight_energy_v2/association_objective_gpu.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    event, base = load_base()
    src = SOURCE / "capera_seed0"
    dest = OUT / "capera_pilot"
    assert not (dest / "started.json").exists()
    plan = read(src / "plan.json")
    data = event.recover.Data("CapERA")
    payload = torch.load(src / "feature_cache/features.pt", map_location="cpu", weights_only=False)
    engine = base.Engine(data, plan, payload)
    torch.set_num_threads(1)
    write(dest / "protocol.json", dict(dataset="CapERA", seed=0, rounds=200,
        scope="Matched ideal-link FL-only control, not a mission method comparison.",
        plan_sha256=sha(src / "plan.json"), features_sha256=sha(src / "feature_cache/features.pt"),
        source_sha256=sha(Path(__file__))))
    write(dest / "started.json", dict(pid=os.getpid(), unix=time.time()))
    try:
        from data_schedule import make_schedule
        from train import update_theta
        records = make_schedule(plan["partitions"], rounds=200, initial_rounds=20, requests=(),
            initial_fraction=1., batch_size=16, seed=0)
        theta = engine.theta0.clone()
        evaluation, _ = engine.evaluate(theta, None)
        rows = [dict(round=0, evaluation=evaluation)]
        for record in records:
            k = record["round"]
            engine.plan["learning_rate"] = base.core.lr_at(k)
            learning, _, _ = engine.updates(theta, record, k, "retrain", [[] for _ in range(10)])
            theta = update_theta(theta, learning.sum(0))
            if k % 5 == 0:
                evaluation, _ = engine.evaluate(theta, None)
                rows.append(dict(round=k, evaluation=evaluation))
                write(dest / "rounds.json", rows)
                print(k, evaluation["retained"]["mR"], evaluation["retained"]["loss"], flush=True)
        values = [r["evaluation"]["retained"]["mR"] for r in rows]
        result = dict(status="completed", initial_mR=values[0], final_mR=values[-1],
            gain_percentage_points=(values[-1]-values[0])*100,
            last_five_mean_mR=sum(values[-5:])/5,
            scope="Ideal-link FL-only control; not Proposed or a five-method mission comparison")
        write(dest / "finished.json", result)
        print(result, flush=True)
    finally:
        base.base.parent.common.base.close()


if __name__ == "__main__":
    main()
