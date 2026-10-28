"""Actual native text/video retrieval from unchanged final200 checkpoints.

No generated captions/images or outcome-based training. Full query rank tables
accompany two explicitly selected contrasting illustrations per dataset.
"""
import argparse
import collections
import fcntl
import math
from pathlib import Path
import time

import numpy as np
import proposed_pks_report as current

ROOT, read, write, sha = current.ROOT, current.read, current.write, current.sha
OUT = ROOT / 'results/tgcn_retrieval_examples_v1_20261028'
METHODS, DATASETS = current.METHODS, current.DATASETS


def register():
    path = OUT/'protocol.json'
    if path.exists():
        p = read(path)
        for name, digest in p['source_sha256'].items():
            assert sha(ROOT/name) == digest, name
        return p
    sources = {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))}
    for ds in DATASETS:
        for m in METHODS:
            arm = current.source('baseline', ds, m)
            for f in [arm/'final.pt', arm.parent/'plan.json', arm.parent/'feature_cache/features.pt']:
                sources[str(f.relative_to(ROOT))] = sha(f)
    p = dict(registered_unix=time.time(), methods=METHODS, datasets=DATASETS,
             checkpoint='Unchanged current final200 model, same45 requests and225 FU cycles, same full gallery and native captions as the published mR. Same workload, not equal energy/time.',
             inference='Use original frozen RemoteCLIP eight-frame mean features plus the trained rank10 image/text adapters; recompute full video-caption similarity matrices and verify all six published retrieval recalls and loss. This is text-to-video retrieval; an original representative frame visualizes each returned video, not generated image or caption output.',
             illustrations='For each dataset, select one query with Proposed better GT rank than every comparator and one with worse rank than at least one comparator. Prefer a top1 success/failure reversal. Restrict illustrative candidates to native caption strings that are unique in this gallery and different top1 predictions; if empty, relax only caption uniqueness, then the top1 reversal preference. Choose median absolute GT-rank advantage/disadvantage, tie-break by caption ID. Disclose empty categories; never fabricate a contrast. These are diagnostic selected examples, not representative performance estimates.',
             comprehensive='Export rank/top3 video IDs for every retained text query and paired better/equal/worse counts. No query filtering for the aggregate metrics. No training or model selection uses these results.',
             publication='Only derived scientific figures with a few original representative frames/native query excerpts, selected-case provenance and full numerical query ranks. No raw video files, full annotation files, feature caches or checkpoints.',
             source_sha256=sources)
    write(path, p)
    return p


def data_for(dataset):
    from airscape import VideoData, load_base
    event, base = load_base()
    return (event.recover.Data(dataset) if dataset == 'CapERA' else VideoData(dataset)), base


def select_examples(rows, captions, gallery_caption_counts):
    alternatives = [m for m in METHODS if m != 'gate_select16']
    selected = []
    for category, sign in [('advantage', 1), ('disadvantage', -1)]:
        candidates = []
        for row in rows:
            ours = row['gate_select16_rank']
            other = min(row[m+'_rank'] for m in alternatives)
            delta = other-ours
            disagreement = any(row[m+'_top1_video'] != row['gate_select16_top1_video'] for m in alternatives)
            if sign*delta > 0 and disagreement:
                candidates.append(dict(row, best_comparator_rank=other, advantage=delta))
        unique = [r for r in candidates if gallery_caption_counts[captions[r['caption_id']].strip().casefold()] == 1]
        used_unique = bool(unique)
        if unique:
            candidates = unique
        reversal = [r for r in candidates if (r['gate_select16_rank'] == 1 if sign == 1 else r['best_comparator_rank'] == 1)]
        used_reversal = bool(reversal)
        if reversal:
            candidates = reversal
        if not candidates:
            selected.append(dict(category=category, available=False))
            continue
        ordered = sorted(candidates, key=lambda r: (abs(r['advantage']), r['caption_id']))
        row = ordered[(len(ordered)-1)//2]
        selected.append(dict(category=category, available=True, query=captions[row['caption_id']],
                             unique_caption_filter=used_unique, top1_reversal=used_reversal,
                             eligible_examples=len(candidates), **row))
    return selected


def run():
    import torch
    from airscape import extra
    register()
    assert not (OUT/'started.json').exists(), 'Preserve previous inference attempts.'
    assert torch.cuda.is_available()
    torch.set_num_threads(1)
    with (ROOT/'results/overnight_energy_v2/association_objective_gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        write(OUT/'started.json', dict(unix=time.time()))
        all_summaries, all_examples = [], []
        for ds in DATASETS:
            data, base = data_for(ds)
            arm = current.source('baseline', ds, 'gate_select16')
            plan = read(arm.parent/'plan.json')
            gallery, validation = plan['gallery_images'], set(plan['validation_images'])
            count = len(data.caption_ids[0])
            assert not validation & set(plan['forgotten_images'])
            payload = torch.load(arm.parent/'feature_cache/features.pt', map_location='cpu', weights_only=False)
            Engine = base.Engine if ds == 'CapERA' else extra.native_engine(base, count)
            engine = Engine(data, plan, payload)
            caption_ids = [cid for vid in gallery for cid in data.caption_ids[vid]]
            gallery_counts = collections.Counter(data.captions[cid].strip().casefold() for cid in caption_ids)
            queries = [(ci, cid, gallery[ci//count]) for ci, cid in enumerate(caption_ids) if gallery[ci//count] in validation]
            rows = [dict(dataset=ds, caption_id=cid, gallery_caption_index=ci, ground_truth_video=vid) for ci, cid, vid in queries]
            arrays = {}
            for m, label in METHODS.items():
                other = current.source('baseline', ds, m)
                assert sha(other.parent/'plan.json') == sha(arm.parent/'plan.json')
                theta = base.core.load_theta(other/'final.pt', engine.device)
                with torch.no_grad():
                    actual, _ = engine.evaluate(theta, None)
                    expected = read(other/'service_completion.json')['evaluation']['retained']
                    for key in ['mR', 'loss', 'i2t_R1', 'i2t_R5', 'i2t_R10', 't2i_R1', 't2i_R5', 't2i_R10']:
                        assert math.isclose(actual['retained'][key], expected[key], abs_tol=2e-7), (ds, m, key)
                    score = engine.score(theta, gallery).cpu()
                    assert tuple(score.shape) == (len(gallery), len(gallery)*count)
                    order = torch.argsort(score.T, dim=1, descending=True)
                    ranks = torch.argsort(order, dim=1) + 1
                    for row, (ci, cid, vid) in zip(rows, queries):
                        gt = ci//count
                        row[m+'_rank'] = int(ranks[ci, gt])
                        row[m+'_top1_video'] = gallery[int(order[ci, 0])]
                        row[m+'_top3_video_ids'] = [gallery[int(i)] for i in order[ci, :3]]
                    for k in [1, 5, 10]:
                        measured = sum(r[m+'_rank'] <= k for r in rows)/len(rows)
                        assert math.isclose(measured, expected[f't2i_R{k}'], abs_tol=2e-7)
                    arrays[m] = score.numpy()
                    all_summaries.append(dict(dataset=ds, method=m, label=label,
                        validation_videos=len(validation), text_queries=len(rows), gallery_videos=len(gallery),
                        **{k: expected[k] for k in ['mR', 'loss', 'i2t_R1', 'i2t_R5', 'i2t_R10', 't2i_R1', 't2i_R5', 't2i_R10']}))
            examples = select_examples(rows, data.captions, gallery_counts)
            for case in examples:
                case['dataset'] = ds
                if case['available']:
                    ids = {case['ground_truth_video'], *[case[m+'_top1_video'] for m in METHODS]}
                    case['media'] = {str(i): dict(file=str(data.media[i].relative_to(ROOT)), sha256=sha(data.media[i])) for i in ids}
                    case['gallery_videos'] = len(gallery)
            pairs = []
            for m in METHODS:
                if m == 'gate_select16':
                    continue
                delta = np.array([r[m+'_rank']-r['gate_select16_rank'] for r in rows])
                pairs.append(dict(dataset=ds, comparator=m, queries=len(rows), better=int((delta>0).sum()),
                                  equal=int((delta==0).sum()), worse=int((delta<0).sum()),
                                  mean_rank_difference=float(delta.mean()), median_rank_difference=float(np.median(delta))))
            write(OUT/(ds.lower()+'_query_ranks.json'), rows)
            write(OUT/(ds.lower()+'_paired_counts.json'), pairs)
            np.savez_compressed(OUT/(ds.lower()+'_scores.npz'), **arrays)
            all_examples += examples
            del data, engine, payload, theta, score, order, ranks, arrays
            torch.cuda.empty_cache()
            print(dict(dataset=ds, queries=len(rows), examples=[dict(category=r['category'], available=r['available']) for r in examples]), flush=True)
        write(OUT/'examples.json', all_examples)
        write(OUT/'summary.json', all_summaries)
        write(OUT/'finished.json', dict(passed=True, models=8, full_original_gallery=True,
                                      original_metrics_reproduced=True, no_training_or_generated_media=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['register', 'run'])
    args = parser.parse_args()
    register() if args.action == 'register' else run()
