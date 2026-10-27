"""A fixed, learning-only pilot on published UAV region/caption observations.

The pilot measures whether the existing adapter learns; it is not a five-method
mission comparison and its ideal-link trajectory is never reported as Proposed.
"""
from pathlib import Path
import collections
import hashlib
import json
import os
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/iot_sensitivity_v1"))
from common import load_base, read, write, sha

OUT = ROOT / "results/uav_convergence_v2_20261027"
DATA = ROOT / "data/uav_construction_v1"
NAME = "UAV-Construction"


def prepare_data():
    import numpy as np
    from PIL import Image
    rows = read(DATA / "annotations.json")
    groups = sorted({(int(r["filename"].split(".")[0]) - 1) // 50 for r in rows.values()})
    validation = set(map(int, np.random.default_rng(0).permutation(groups)[:6]))
    selected, hashes, duplicates = [], {}, []
    with zipfile.ZipFile(DATA / "image_with_captions.zip") as archive:
        for row in sorted(rows.values(), key=lambda r: int(r["filename"].split(".")[0])):
            image_id = int(row["filename"].split(".")[0])
            group = (image_id - 1) // 50
            # Keep at most two largest valid annotated regions per source image.
            regions = [(key, r) for key, r in row["regions"].items()
                       if r["region_attributes"].get("phrase", "").strip()
                       and min(r["shape_attributes"].get("width", 0),
                              r["shape_attributes"].get("height", 0)) >= 32]
            regions.sort(key=lambda pair: (-pair[1]["shape_attributes"]["width"] *
                                          pair[1]["shape_attributes"]["height"], int(pair[0])))
            with archive.open("UAVdata/" + row["filename"]) as source:
                image = Image.open(source).convert("RGB")
            used = set()
            for region_id, region in regions:
                box = region["shape_attributes"]
                bounds = (max(0, box["x"]), max(0, box["y"]),
                          min(image.width, box["x"] + box["width"]),
                          min(image.height, box["y"] + box["height"]))
                if bounds in used or min(bounds[2]-bounds[0], bounds[3]-bounds[1]) < 32:
                    continue
                used.add(bounds)
                if len(used) > 2:
                    break
                crop = image.crop(bounds)
                pixel_hash = hashlib.sha256(str(crop.size).encode() + crop.tobytes()).hexdigest()
                if pixel_hash in hashes:
                    duplicates.append(dict(image=image_id, region=region_id, duplicate_of=hashes[pixel_hash]))
                    continue
                key = f"{image_id}_{region_id}"
                hashes[pixel_hash] = key
                path = DATA / "regions" / (key + ".png")
                path.parent.mkdir(exist_ok=True)
                if not path.exists():
                    crop.save(path)
                caption = region["region_attributes"]["phrase"].strip()
                labels = ["excavator", "truck", "worker", "crane", "foundation", "building",
                          "concrete", "soil", "fence", "mold", "tarpaulin"]
                label = next((w for w in labels if w in caption.lower()), "other")
                selected.append(dict(source_key=key, path=str(path.relative_to(ROOT)),
                    captions=[caption], label=label, original_image=row["filename"], region_id=region_id,
                    bbox_xyxy=bounds, source_block=group, pixel_sha256=pixel_hash,
                    split="validation" if group in validation else "train"))
    val = [r for r in selected if r["split"] == "validation"]
    # Bound evaluation size before any feature or learning outcome is examined.
    keep = set(np.random.default_rng(0).permutation(len(val))[:256])
    selected = [r for r in selected if r["split"] == "train"] + [r for i, r in enumerate(val) if i in keep]
    selected.sort(key=lambda r: (int(r["original_image"].split(".")[0]), int(r["region_id"])))
    write(DATA / "selection.json", dict(items=selected,
        policy="Two largest native annotated regions (minimum side32) per original frame; "
               "consecutive filename blocks of50 split by seed0; six whole blocks validation, "
               "at most256 fixed validation regions; exact pixel duplicates removed before training.",
        limitations="No author flight/site metadata or official split was supplied. Filename blocks reduce "
                    "adjacent-frame overlap but do not establish geographical independence. "
                    "This is a region-text adaptation of the dataset, not global image-caption retrieval.",
        validation_blocks=sorted(validation), removed_exact_duplicates=duplicates,
        source="https://data.mendeley.com/datasets/4h68fmktwh/1"))
    write(DATA / "ready.json", dict(status="ready", dataset=NAME, images=len(selected),
        selection_sha256=sha(DATA / "selection.json"),
        split_counts=dict(collections.Counter(r["split"] for r in selected)),
        native_captions_per_region=1, unique_caption_strings=len({r["captions"][0] for r in selected})))
    return read(DATA / "ready.json")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument("--rebuild-data", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.rebuild_data or not (DATA / "ready.json").exists():
        print(json.dumps(prepare_data()), flush=True)
    if args.data_only:
        return
    import fcntl
    import torch
    from prepare_extra import native_engine
    import prepare_extra as extra
    assert torch.cuda.is_available()
    lock = (ROOT / "results/overnight_energy_v2/association_objective_gpu.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    extra.DATA = ROOT / "data"
    extra.FOLDERS[NAME] = "uav_construction_v1"
    event, base = load_base()
    dest = OUT / "construction_pilot"
    dest.mkdir(exist_ok=True)
    assert not (dest / "started.json").exists(), "Preserve previous pilot attempts"
    data = extra.Data(NAME)
    plan = extra.register_data(base, NAME, data, dest)
    plan.update(scope="Learning-only feasibility pilot, region-text adaptation; no official test split available.",
                model="RemoteCLIP B/32 + unchanged rank10 residual adapters")
    write(dest / "plan.json", plan)
    write(dest / "protocol.json", dict(dataset=NAME, seed=0, rounds=200,
        change="Dataset only; original adapter, optimizer, batch and learning-rate schedule retained.",
        comparison="Ideal aggregation, FL only, no FU and no mission cost. Not a method ranking.",
        decision="Assess final and late-average held-out mR improvement; do not choose a seed or smooth the curve.",
        selection_sha256=sha(DATA / "selection.json"), source_sha256=sha(Path(__file__))))
    write(dest / "started.json", dict(pid=os.getpid(), unix=time.time()))
    def status(**kw):
        write(dest / "status.json", dict(pid=os.getpid(), unix=time.time(), **kw))
    try:
        payload, metadata = event.recover.original.prepare_features(data, plan,
            ROOT / "checkpoints/remoteclip_vitb32_uav_v1", dest / "feature_cache", write, status)
        metadata.update(video_representation="One original annotated region crop per sample",
                        native_captions_per_image=1)
        write(dest / "feature_cache/manifest.json", metadata)
        engine = native_engine(base, 1)(data, plan, payload)
        torch.set_num_threads(1)
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
                status(stage="learning_pilot", round=k, retained=evaluation["retained"])
                print(k, evaluation["retained"]["mR"], evaluation["retained"]["loss"], flush=True)
        values = [r["evaluation"]["retained"]["mR"] for r in rows]
        result = dict(status="completed", initial_mR=values[0], final_mR=values[-1],
            gain_percentage_points=(values[-1]-values[0])*100,
            last_five_mean_mR=sum(values[-5:])/5,
            scope="Ideal-link FL-only pilot; not Proposed or a five-method mission comparison")
        write(dest / "finished.json", result)
        print(json.dumps(result), flush=True)
    finally:
        base.base.parent.common.base.close()


if __name__ == "__main__":
    main()
