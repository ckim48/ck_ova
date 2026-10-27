"""Read-only checkpoint/ledger audit and corrected bandwidth display/metric label."""
import collections
import fcntl
import json
import math
from pathlib import Path
import subprocess

from baseline_fairness import ROOT, OUT as SOURCE, MAIN, DATASETS, read, write, sha, load_base
import selected_graphs as plots

OUT = ROOT / "results/energy_loss_audit_v1_20261027"
PROPOSED = "proposed_fu_guard_cliperase"


def energy_audit(sources):
    rows = []
    ledgers = read(ROOT / "selected_graph/data/service_energy_delay.json")
    for dataset in DATASETS:
        directory = SOURCE / "cases/baseline" / (dataset.lower()+"_seed0")
        parameters = read(directory.parent / "settings.json")["energy_model"]
        plots.physical.PARAMETERS = parameters
        for method, label in MAIN.items():
            arm = directory / method
            groups = collections.defaultdict(lambda: dict(transactions=0, nonzero_uplinks=0, uplink_s=0.,
                downlink_s=0., compute_s=0., RF_energy_j=0., communication_energy_j=0., modes=set()))
            computing = 0.
            for path in sorted((arm / "work_attempts").glob("*.json")):
                sources[str(path.relative_to(ROOT))] = sha(path)
                work = read(path)
                computing += float(plots.physical.compute_cost(work["compute"]["device_macs"])[0].sum())
            for path in sorted((arm / "transactions").glob("*.json")):
                sources[str(path.relative_to(ROOT))] = sha(path)
                tx = read(path); radio = tx["radio"]
                comm, broadcast = plots.physical.communication(radio)
                assert math.isclose(broadcast, tx["timing"]["downlink_s"], abs_tol=1e-12)
                computing += parameters["communication"]["head_processing_w"]*radio["uplink_s"]
                group = groups[tx["kind"]]
                group["transactions"] += 1
                group["nonzero_uplinks"] += int(radio["uplink_s"] > 0)
                for key in ["uplink_s", "downlink_s", "compute_s"]:
                    group[key] += tx["timing"][key]
                group["RF_energy_j"] += radio["energy_bound_j"]
                group["communication_energy_j"] += comm
                group["modes"].add(radio.get("physical_mode", radio.get("access")))
            for group in groups.values():
                group["modes"] = sorted(group["modes"])
            ledger = next(r for r in ledgers if r["bandwidth_MHz"]==1 and r["dataset"]==dataset and r["method"]==method)
            comm = sum(g["communication_energy_j"] for g in groups.values())
            seconds = sum(g[k] for g in groups.values() for k in ["uplink_s","downlink_s","compute_s"])
            seconds += ledger["scheduler_wait_s"]
            hover_power = parameters["iot_collection"]["uavs"]*float(plots.physical.propulsion_power(0.))
            hover = seconds*hover_power
            assert math.isclose(comm,ledger["service_communication_j"],abs_tol=1e-8)
            assert math.isclose(computing,ledger["service_computing_j"],abs_tol=1e-8)
            assert math.isclose(seconds,ledger["delay_s"],abs_tol=1e-8)
            assert math.isclose(comm+computing+hover,ledger["total_energy_j"],abs_tol=1e-7)
            rows.append(dict(dataset=dataset, method=method, label=label, phases=dict(groups),
                service_delay_s=seconds, total_energy_kJ=ledger["total_energy_j"]/1000,
                communication_energy_j=comm, computing_energy_j=computing, hover_energy_j=hover,
                hover_fraction=hover/ledger["total_energy_j"], hover_power_W=hover_power,
                downlink_fraction=sum(g["downlink_s"] for g in groups.values())/seconds))
        hybrid = next(r for r in rows if r["dataset"]==dataset and r["label"]=="AC-OF")
        digital = next(r for r in rows if r["dataset"]==dataset and r["label"]=="S-OF")
        assert "integer_subcarrier_OFDMA" in hybrid["phases"]["fu"]["modes"]
        assert "joint" in hybrid["phases"]["fl"]["modes"]
        for key in ["transactions", "uplink_s", "RF_energy_j"]:
            assert math.isclose(hybrid["phases"]["fu"][key],digital["phases"]["fu"][key],abs_tol=1e-9)
    return rows


def loss_audit(sources):
    import torch
    import torch.nn.functional as F
    import numpy as np
    from airscape import VideoData, extra
    assert torch.cuda.is_available()
    torch.set_num_threads(1)
    lock = (ROOT / "results/overnight_energy_v2/association_objective_gpu.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    event, base = load_base()
    rows = []
    try:
        for dataset in DATASETS:
            directory = SOURCE / "cases/baseline" / (dataset.lower()+"_seed0")
            plan = read(directory/"plan.json")
            data = event.recover.Data(dataset) if dataset=="CapERA" else VideoData(dataset)
            count = len(data.caption_ids[0]); gallery = plan["gallery_images"]
            payload = torch.load(directory/"feature_cache/features.pt",map_location="cpu",weights_only=False)
            Engine = base.Engine if dataset=="CapERA" else extra.native_engine(base,count)
            engine = Engine(data,plan,payload)
            initial_saved = read(ROOT / "results/uav_convergence_v2_20261027" / (dataset.lower()+"_pilot/rounds.json"))[0]["evaluation"]["retained"]
            final_saved = read(directory/PROPOSED/"manifest.json")["evaluation"]["retained"]
            checkpoints = [("initial",engine.theta0.clone(),initial_saved),
                ("final200",base.core.load_theta(directory/PROPOSED/"final.pt",engine.device),final_saved)]
            snapshots = []
            with torch.no_grad():
                for stage,theta,expected in checkpoints:
                    actual,_ = engine.evaluate(theta,None)
                    for key in ["loss","mR"]:
                        assert math.isclose(actual["retained"][key],expected[key],abs_tol=2e-7)
                    score = engine.score(theta,gallery)
                    assert tuple(score.shape)==(len(gallery),len(gallery)*count)
                    idx = torch.tensor([gallery.index(i) for i in plan["validation_images"]],device=score.device)
                    caption_idx = (idx[:,None]*count+torch.arange(count,device=score.device)).flatten()
                    owner = caption_idx//count
                    image_nll = torch.logsumexp(score[idx],1)-torch.logsumexp(score[idx[:,None],idx[:,None]*count+torch.arange(count,device=score.device)],1)
                    text_nll = torch.logsumexp(score[:,caption_idx],0)-score[owner,caption_idx]
                    manual = float((image_nll.mean()+text_nll.mean())/2)
                    assert math.isclose(manual,expected["loss"],abs_tol=3e-6)
                    assert theta[-1].item()==engine.theta0[-1].item()
                    # A fixed training-only snapshot diagnostic, not a reconstructed learning curve.
                    losses=[];weights=[];forgotten=set(plan["forgotten_images"])
                    for user,part in enumerate(plan["partitions"]):
                        pool=sorted(set(part)-forgotten)
                        ids=np.random.default_rng(730000+21*10+user).choice(pool,min(16,len(pool)),replace=False).astype(int).tolist()
                        captions=data.training_caption_ids(ids,21)
                        im,tx,scale=engine.features(theta,ids,captions)
                        logits=scale.exp()*(im@tx.T); target=torch.arange(len(ids),device=logits.device)
                        losses.append(float((F.cross_entropy(logits,target)+F.cross_entropy(logits.T,target))/2))
                        weights.append(len(pool))
                    snapshots.append(dict(stage=stage,validation_loss=expected["loss"],manual_validation_loss=manual,
                        retained_mR_percent=100*expected["mR"], fixed_train_batch_loss=float(np.average(losses,weights=weights))))
            before,after=snapshots
            rows.append(dict(dataset=dataset,gallery_videos=len(gallery),captions_per_video=count,
                validation_queries=len(plan["validation_images"]),uniform_prediction_loss=math.log(len(gallery)),
                trainable_coordinates=engine.theta0.numel()-1, logit_scale_fixed=True,
                snapshots=snapshots, validation_loss_reduction=before["validation_loss"]-after["validation_loss"],
                geometric_correct_probability_ratio=math.exp(before["validation_loss"]-after["validation_loss"]),
                lr_at_40=base.core.lr_at(40),lr_at_200=base.core.lr_at(200)))
            for name in ["plan.json","feature_cache/features.pt",PROPOSED+"/final.pt",PROPOSED+"/manifest.json"]:
                path=directory/name;sources[str(path.relative_to(ROOT))]=sha(path)
            del engine,payload,score
            torch.cuda.empty_cache()
    finally:
        base.base.parent.common.base.close()
        fcntl.flock(lock,fcntl.LOCK_UN)
    return rows


def main():
    sources = {str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))}
    energy = energy_audit(sources)
    losses = loss_audit(sources)
    plots.generate("main");plots.generate("guard_ablation")
    git=["git","--git-dir=.git-push-ready/code-charts.git","--work-tree=."]
    unchanged=0
    for folder in ["selected_graph","ablation_graph"]:
        prior=json.loads(subprocess.check_output(git+["show",f"ed100c8dbd362607b004af0a6424d5fcb12fb392:{folder}/data/plot_series.json"],cwd=ROOT))
        current=read(ROOT/folder/"data/plot_series.json")
        assert len(prior)==len(current)
        for old,new in zip(prior,current):
            for key in ["figure","dataset","method","x","y","status"]:
                assert old.get(key)==new.get(key),(folder,key)
            if "bandwidth" in new["figure"]:
                assert new["display_x"]==[1,2,4]
                assert new["display_y"]==[y for x,y in zip(new["x"],new["y"]) if x>=1]
            unchanged+=1
        manifest=read(ROOT/folder/"data/manifest.json")
        for fig in manifest["figures"]:
            for ax in fig["axes"]:
                if "bandwidth" in fig["name"]:
                    assert ax["x"]["limits"]==[1,4] and ax["x"]["ticks"]==[1,2,3,4]
                elif "loss" in fig["name"]:
                    assert ax["ylabel"]=="Validation loss"
    diagnostic_path=ROOT/"results/uav_longer_learning_diagnostic_20261027/finished.json"
    diagnostic=read(diagnostic_path);sources[str(diagnostic_path.relative_to(ROOT))]=sha(diagnostic_path)
    write(OUT/"audit.json",dict(energy=energy,losses=losses,source_sha256=sources,unchanged_raw_series=unchanged,
        checkpoint_re_evaluation=True,no_training_or_parameter_updates=True,
        training_snapshot_scope="Fixed retained-training batches, same 16-or-fewer samples/captions before and after; not historical per-round training loss.",
        longer_diagnostic=diagnostic["results"],longer_diagnostic_scope=diagnostic["scope"]))
    lines=["# Bandwidth display, hybrid energy and validation loss audit","",
        "Bandwidth axes now start at 1 MHz, with regular ticks 1, 2, 3, 4. Measured displayed points are 1, 2 and 4 MHz. "
        "The 0.25/0.5 MHz outcomes remain in the raw data; no value was invented for the 3 MHz tick. "
        "The AirScape H-OVA 4 MHz infeasibility remains marked.","",
        "## AC-OF energy","",
        "AC-OF uses AirComp for FL and integer-subcarrier OFDMA only for FU. S-OF uses digital OFDMA for both. "
        "At 1 MHz, their FU transaction counts, uplink times and RF energy are equal on both datasets. "
        "No OFDMA cost was omitted from AC-OF. Its lower energy comes from its analog FL phase. "
        "Each transaction's radio energy, PA/circuit/broadcast cost, recorded MAC cost and hover duration was independently summed and reconciled with the plotted ledger.","",
        "| Dataset | Method | FL uplink (s) | FU uplink (s) | Joint uplink (s) | Communication (J) | Hover share (%) | Service energy (kJ) | Service delay (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in energy:
        phase=lambda k:row["phases"].get(k,{}).get("uplink_s",0.)
        lines.append(f"| {row['dataset']} | {row['label']} | {phase('fl'):.3f} | {phase('fu'):.3f} | {phase('joint'):.3f} | {row['communication_energy_j']:.3f} | {row['hover_fraction']*100:.4f} | {row['total_energy_kJ']:.3f} | {row['service_delay_s']:.3f} |")
    lines += ["","The modeled ten-UAV hover power is 1,684.9 W. Over 99.8% of displayed service energy is hovering, "
        "so energy follows service duration closely. This is not an RF-only energy comparison. A 32-bit aggregate "
        "broadcast at 1 Mbps is charged after every nonzero transaction. For baseline Proposed, broadcasts occupy "
        "117.401 s of about 121.09 s service; that downlink assumption, the reduced number of broadcasts under joint "
        "scheduling, and the fixed hover-power model strongly influence the reported advantage. "
        "These are modeled scenario costs, not hardware-calibrated measurements. Travel/collection/preflight are excluded.","",
        "## Loss definition and learning progress","",
        "The old FL loss label referred to held-out retained-query retrieval negative log likelihood, not the local "
        "minibatch training loss. Its label is now Validation loss; all values and filenames are preserved. "
        "Initial and final checkpoints were reevaluated and the loss independently recomputed from raw gallery scores. "
        "Image-to-text uses the probability mass of all positive captions, text-to-image uses the correct video probability, "
        "and the two directional mean negative log probabilities are averaged. With N videos and equal caption counts, "
        "uniform predictions give log(N): 6.22059 for CapERA and 6.41999 for AirScape. A value near 5 is not by itself a bug, "
        "nor is contrastive loss meaningless for a VLM. [CLIP (ICML 2021)](https://proceedings.mlr.press/v139/radford21a/radford21a.pdf) "
        "uses symmetric contrastive cross entropy; this project's multi-caption validation reduction is defined by its own evaluator.","",
        "| Dataset | Uniform loss | Initial validation loss | Final validation loss | Initial mR (%) | Final mR (%) | Fixed train-batch loss before / after |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for row in losses:
        before,after=row["snapshots"]
        lines.append(f"| {row['dataset']} | {row['uniform_prediction_loss']:.4f} | {before['validation_loss']:.4f} | {after['validation_loss']:.4f} | {before['retained_mR_percent']:.4f} | {after['retained_mR_percent']:.4f} | {before['fixed_train_batch_loss']:.4f} / {after['fixed_train_batch_loss']:.4f} |")
    lines += ["","A decrease in this log loss indicates improved correct-match probability, but the observed retained-mR "
        "gains are modest. This experiment freezes RemoteCLIP, pools eight frame features, and trains only 20,480 "
        "rank-10 adapter coordinates. It does not fine-tune the entire VLM. The fixed learning-rate schedule reaches "
        "zero at round 200, so a flat endpoint is not evidence that the task is fully optimized. "
        "Logit scale is frozen and unchanged, so loss improvement is not obtained by tuning temperature.","",
        "An existing separate FL-only, ideal-aggregation diagnostic restarted LR at 0.001 after round 200 and continued "
        "to round 1000: CapERA mR increased 16.4193→22.5130%, and AirScape 13.5417→15.4948%. "
        "This indicates remaining learning potential under that changed training setup. It changes both duration and "
        "LR, omits FU and radio, and cannot substitute for the main 200-round FL/FU method comparison. "
        "No new training or parameter updates were performed for this audit.","",
        f"All {unchanged} original plotted numerical series remain unchanged. Source hashes, phase-level accounting "
        "and checkpoint reevaluation are in `audit.json`. Rebuild with "
        "`.venv/bin/python tools/uav_convergence_v2/energy_loss_audit.py`.",""]
    (OUT/"README.md").write_text("\n".join(lines))
    print(dict(verified=True,unchanged_raw_series=unchanged,losses=losses),flush=True)


if __name__=="__main__":
    main()
