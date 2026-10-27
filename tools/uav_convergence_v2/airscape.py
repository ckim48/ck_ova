"""Fixed source-video-disjoint AirScape pilot using its public real UAV footage.

This adapts video/intention pairs to retrieval, not the paper's world-model task.
No generated video, official test sample, caption rewrite, or outcome filtering.
"""
from pathlib import Path
import concurrent.futures
import collections
import hashlib
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/iot_sensitivity_v1"))
from common import load_base, read, write, sha
import prepare_extra as extra

NAME = "AirScape"
DATA = ROOT / "data/airscape_v1"
OUT = ROOT / "results/uav_convergence_v2_20261027"
extra.DATA = ROOT / "data"
extra.FOLDERS[NAME] = DATA.name


def metadata():
    """Fetch only public metadata, including test hashes but no test videos."""
    import requests
    DATA.mkdir(parents=True, exist_ok=True)
    parquet = DATA / "prompt.parquet"
    if not parquet.exists():
        response = requests.get("https://huggingface.co/datasets/EmbodiedCity/AirScape-Dataset/resolve/main/prompt.parquet", timeout=60)
        response.raise_for_status()
        parquet.write_bytes(response.content)
    for split in ["train", "test"]:
        path = DATA / (split + "_tree.json")
        if path.exists():
            continue
        url = "https://huggingface.co/api/datasets/EmbodiedCity/AirScape-Dataset/tree/main/" + split + "?limit=1000"
        rows = []
        while url:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            rows.extend(response.json())
            url = response.links.get("next", {}).get("url")
        write(path, rows)


def select():
    import numpy as np
    import pandas as pd
    if (DATA / "selection.json").exists():
        return read(DATA / "selection.json")
    metadata()
    annotations = pd.read_parquet(DATA / "prompt.parquet")
    files = {Path(r["path"]).name: r for r in read(DATA / "train_tree.json")}
    source = annotations[annotations.video_id.str.contains("_WebUAV3M_")].copy()
    source["group"] = source.video_id.str.replace(
        r"^\d+_WebUAV3M_(train|test)_", "", regex=True).str.replace(r"_\d+\.mp4$", "", regex=True)
    test_groups = set(source[source["set"] == "test"].group)
    train = source[(source["set"] == "train") & ~source.group.isin(test_groups)
                   & source.video_id.isin(files)].copy()
    # Shared footage can have several target labels/sequence names. Conservatively
    # exclude every sequence containing an exact duplicate anywhere in this pool
    # or in the official test metadata, before choosing the split.
    test_hashes = {r["lfs"]["oid"] for r in read(DATA / "test_tree.json") if "lfs" in r}
    hash_counts = collections.Counter(files[name]["lfs"]["oid"] for name in train.video_id)
    duplicate_groups = set(train[train.video_id.map(
        lambda name: hash_counts[files[name]["lfs"]["oid"]] > 1
        or files[name]["lfs"]["oid"] in test_hashes)].group)
    train = train[~train.group.isin(duplicate_groups)]
    groups = sorted(train.group.unique())
    assert len(groups) >= 1792
    rng = np.random.default_rng(0)
    selected_groups = rng.permutation(groups)[:1792]
    items = []
    for i, group in enumerate(selected_groups):
        candidates = train[train.group == group].sort_values("video_id")
        row = candidates.iloc[int(rng.integers(len(candidates)))]
        media = files[row.video_id]
        items.append(dict(source_key=row.video_id, source_group=str(group),
            label=str(group).rsplit("_", 1)[0],
            split="validation" if i < 256 else "train",
            path=str((DATA / "videos" / row.video_id).relative_to(ROOT)),
            remote_path=media["path"], bytes=media["size"], sha256=media["lfs"]["oid"],
            captions=[row.motion_intention]))
    assert len({r["sha256"] for r in items}) == len(items), "Duplicate original media"
    payload = dict(dataset=NAME, items=items, seed=0,
        policy="Use original WebUAV3M-source videos from the official AirScape training split. "
               "Group by source sequence (remove leading dataset index and trailing clip index); "
               "exclude every official-test source sequence; shuffle groups with seed0, "
               "first256 validation and next1536 training; choose one random clip per group. "
               "Exclude entire source sequences containing a file duplicated in the candidate pool "
               "or official test metadata. Only publicly present files are eligible. Native text unchanged.",
        scope="Video-intention retrieval adaptation of AirScape, WebUAV3M source subset. "
              "Not reproduction of the original world-model benchmark. One native text per video; "
              "same eight-frame mean pooling as CapERA cannot represent temporal ordering.",
        eligible_source_groups=len(groups), official_test_groups_excluded=len(test_groups),
        exact_duplicate_source_groups_excluded=len(duplicate_groups),
        metadata_sha256=sha(DATA / "prompt.parquet"),
        source="https://huggingface.co/datasets/EmbodiedCity/AirScape-Dataset",
        paper="https://doi.org/10.1145/3746027.3758180")
    write(DATA / "selection.json", payload)
    write(OUT / "airscape_selection.json", dict(payload,
        items=[{k: v for k, v in row.items() if k != "captions"} for row in items],
        caption_source="Native text remains in the official parquet; no captions redistributed here.",
        full_selection_sha256=sha(DATA / "selection.json")))
    return payload


def acquire():
    import requests
    selected = select()
    def get(row):
        path = ROOT / row["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size == row["bytes"] and sha(path) == row["sha256"]:
            return
        url = "https://huggingface.co/datasets/EmbodiedCity/AirScape-Dataset/resolve/main/" + row["remote_path"]
        temp = path.with_suffix(".part")
        for attempt in range(5):
            try:
                with requests.get(url, stream=True, timeout=(30, 90)) as response:
                    response.raise_for_status()
                    with temp.open("wb") as f:
                        for chunk in response.iter_content(1 << 20):
                            f.write(chunk)
                assert temp.stat().st_size == row["bytes"] and sha(temp) == row["sha256"]
                temp.replace(path)
                return
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
    print(json.dumps(dict(stage="download", samples=len(selected["items"]),
        GB=sum(r["bytes"] for r in selected["items"]) / 1e9)), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
        jobs = {pool.submit(get, row): row for row in selected["items"]}
        for n, future in enumerate(concurrent.futures.as_completed(jobs), 1):
            future.result()
            if n % 40 == 0:
                write(DATA / "download_status.json", dict(completed=n, total=len(jobs), unix=time.time()))
                print("downloaded", n, "/", len(jobs), flush=True)
    result = dict(status="ready", dataset=NAME, images=len(selected["items"]),
        selection_sha256=sha(DATA / "selection.json"), native_captions_per_video=1,
        split_counts=dict(collections.Counter(r["split"] for r in selected["items"])))
    write(DATA / "ready.json", result)
    print(json.dumps(result), flush=True)


class VideoData(extra.Data):
    def frames(self, index):
        import cv2
        import numpy as np
        from PIL import Image
        path = self.media[int(index)]
        capture = cv2.VideoCapture(str(path))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        assert count >= 8 and capture.isOpened(), str(path)
        indices = np.linspace(0, count - 1, 8).round().astype(int).tolist()
        frames = []
        try:
            # Same eight frame indices; decode once instead of seeking eight times.
            selected = set(indices)
            for frame in range(indices[-1] + 1):
                assert capture.grab(), (str(path), frame)
                if frame in selected:
                    okay, pixels = capture.retrieve()
                    assert okay, (str(path), frame)
                    frames.append(Image.fromarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)))
        finally:
            capture.release()
        return frames, indices


def pilot():
    import fcntl
    import torch
    assert torch.cuda.is_available()
    lock = (ROOT / "results/overnight_energy_v2/association_objective_gpu.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    event, base = load_base()
    dest = OUT / "airscape_pilot"
    dest.mkdir(parents=True, exist_ok=True)
    assert not (dest / "started.json").exists(), "Preserve previous pilot attempts"
    data = VideoData(NAME)
    plan = extra.register_data(base, NAME, data, dest)
    plan.update(scope=select()["scope"], model="RemoteCLIP B/32 + unchanged rank10 residual adapters")
    write(dest / "plan.json", plan)
    write(dest / "protocol.json", dict(dataset=NAME, seed=0, rounds=200,
        change="Dataset only; original adapter, optimizer, batch and learning-rate schedule retained.",
        comparison="Ideal aggregation, FL only, no FU and no mission cost. Not a method ranking.",
        decision="Assess final and late-average held-out mR improvement; retain all pilot outcomes.",
        selection_sha256=sha(DATA / "selection.json"), source_sha256=sha(Path(__file__))))
    write(dest / "started.json", dict(pid=os.getpid(), unix=time.time()))
    def status(**kw):
        write(dest / "status.json", dict(pid=os.getpid(), unix=time.time(), **kw))
    try:
        payload, metadata = event.recover.original.prepare_features(data, plan,
            ROOT / "checkpoints/remoteclip_vitb32_uav_v1", dest / "feature_cache", write, status)
        metadata.update(native_captions_per_video=1, subset_manifest_sha256=sha(DATA / "selection.json"))
        write(dest / "feature_cache/manifest.json", metadata)
        engine = extra.native_engine(base, 1)(data, plan, payload)
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["select", "acquire", "pilot"])
    args = parser.parse_args()
    if args.action == "select":
        result = select()
        print(json.dumps(dict(samples=len(result["items"]),
            GB=sum(r["bytes"] for r in result["items"]) / 1e9, policy=result["policy"])))
    elif args.action == "acquire":
        acquire()
    else:
        pilot()
