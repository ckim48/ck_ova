"""Fresh RSITMD/CapERA experiment; frozen four-method transport and objectives.

RemoteCLIP is an IEEE TGRS backbone. Uniform eight-frame mean pooling and the
residual adapters are our adaptation, not a reproduction of a video model.
No previous learned checkpoint or metric is imported into this experiment.
"""
from pathlib import Path
import argparse
import copy
import fcntl
import hashlib
import importlib.util
import json
import os
import sys
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'results/uav_remoteclip_capera_v1'
DATA = ROOT / 'data/capera_official_v1'
CHECKPOINT = ROOT / 'checkpoints/remoteclip_vitb32_uav_v1'
sys.path.insert(0, str(ROOT / '.cache/uav_experiment_deps'))
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '4'

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def state(dest=OUT, **kwargs):
    write(Path(dest) / 'status.json', dict(unix=time.time(), pid=os.getpid(), **kwargs))


def baseline_module():
    path = ROOT / 'tools/overnight_energy_v2/uav_single_model_two_dataset_v1/campaign.py'
    spec = importlib.util.spec_from_file_location('uav_frozen_adapter_campaign', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def prepare_videos():
    destination = DATA / 'videos'
    marker = DATA / 'videos_verified.json'
    if marker.exists():
        report = read(marker)
        assert all((destination / name).is_file() for name in report['video_files'])
        return report
    archive = DATA / 'ERA_Dataset.zip'
    if not archive.exists():
        raise FileNotFoundError('ERA video archive download is not complete')
    state(stage='verifying_and_extracting_ERA_videos')
    videos = []
    with zipfile.ZipFile(archive) as zipped:
        for info in zipped.infolist():
            name = Path(info.filename)
            assert not name.is_absolute() and '..' not in name.parts
            if name.suffix.lower() not in {'.mp4', '.avi', '.mov'}:
                continue
            # Reading to EOF verifies each compressed member's CRC.
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with zipped.open(info) as source, target.open('wb') as output:
                for chunk in iter(lambda: source.read(1024 * 1024), b''):
                    output.write(chunk)
            videos.append(info.filename)
    assert len(videos) == 2864, len(videos)
    report = dict(video_count=len(videos), video_files=videos,
                  archive_sha256=sha(archive), archive_bytes=archive.stat().st_size,
                  source='https://lcmou.github.io/ERA_Dataset/',
                  captions='https://github.com/yakoubbazi/CapEra', crc_verified=True)
    write(marker, report)
    return report


class Data:
    def __init__(self, dataset):
        self.dataset = dataset
        self.items, self.media = [], {}
        if dataset == 'RSITMD':
            root = ROOT / 'data/rsitmd_original_v1'
            self.items = sorted(read(root / 'dataset_RSITMD.json')['images'], key=lambda item: item['imgid'])
            paths = {p.name: p for p in root.rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}}
            self.media = {item['imgid']: paths[item['filename']] for item in self.items}
        elif dataset == 'CapERA':
            paths = list((DATA / 'videos').rglob('*.mp4'))
            for split in ['train', 'test']:
                for annotation in read(DATA / (split + '.json'))['ERA_caption']:
                    filename = annotation['video_id']
                    aliases = {'tra', 'train', 'training', 'trainset', 'train_set'} if split == 'train' else {'test', 'testing', 'testset', 'test_set'}
                    matches = [p for p in paths if p.stem.strip() + p.suffix == filename and aliases.intersection(part.lower() for part in p.parts)]
                    assert len(matches) == 1, (split, filename, [str(p) for p in matches])
                    index = len(self.items)
                    self.items.append(dict(imgid=index, filename=filename, split=split,
                                           sentences=[dict(raw=text) for text in annotation['annotation']['English_caption']],
                                           media_key=split + '/' + filename))
                    self.media[index] = matches[0]
        else:
            raise ValueError(dataset)
        self.captions, self.owners, self.caption_ids = [], [], []
        assert [item['imgid'] for item in self.items] == list(range(len(self.items)))
        for item in self.items:
            assert len(item['sentences']) == 5
            ids = []
            for sentence in item['sentences']:
                assert sentence['raw'].strip()
                ids.append(len(self.captions))
                self.captions.append(sentence['raw'])
                self.owners.append(item['imgid'])
            self.caption_ids.append(ids)
        self.owners = np.asarray(self.owners)

    def training_caption_ids(self, ids, round_number):
        return [self.caption_ids[int(i)][(round_number + int(i)) % 5] for i in ids]

    def frames(self, index):
        from PIL import Image
        path = self.media[int(index)]
        if self.dataset == 'RSITMD':
            with Image.open(path) as image:
                return [image.convert('RGB')], [0]
        import cv2
        capture = cv2.VideoCapture(str(path))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        assert count >= 8 and capture.isOpened(), str(path)
        indices = np.linspace(0, count - 1, 8).round().astype(int).tolist()
        frames = []
        try:
            for frame in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
                okay, pixels = capture.read()
                assert okay, (str(path), frame)
                frames.append(Image.fromarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)))
        finally:
            capture.release()
        return frames, indices


def prepare_features(data, plan, checkpoint, dest, writer, status):
    import gc
    import open_clip
    import torch
    import torch.nn.functional as F
    dest.mkdir(parents=True, exist_ok=True)
    assert not (dest / 'features.pt').exists(), 'No replacement of existing feature archive'
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=None)
    model.load_state_dict(torch.load(Path(checkpoint) / 'RemoteCLIP-ViT-B-32.pt', map_location='cpu', weights_only=True), strict=True)
    model = model.cuda().float().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    ids = sorted(set(sum(plan['partitions'], []) + plan['gallery_images'] + plan['mia_member_images'] + plan['mia_nonmember_images']))
    assert set(ids).isdisjoint(plan['official_test_images'])
    cids = [cid for index in ids for cid in data.caption_ids[index]]
    images, texts, lineage = [], [], []
    training = set(sum(plan['partitions'], []))
    started = time.time()
    with torch.no_grad():
        for batch_start in range(0, len(ids), 8):
            batch = ids[batch_start:batch_start + 8]
            frame_tensors, sizes = [], []
            for index in batch:
                frames, frame_ids = data.frames(index)
                sizes.append(len(frames))
                frame_tensors.extend(preprocess(frame) for frame in frames)
                lineage.append(dict(sample_id=index, file=str(data.media[index].relative_to(ROOT)),
                                    sha256=sha(data.media[index]), frames=frame_ids,
                                    training_sample=index in training))
            encoded = F.normalize(model.encode_image(torch.stack(frame_tensors).cuda()), dim=-1)
            offset = 0
            for size in sizes:
                images.append(F.normalize(encoded[offset:offset + size].mean(0), dim=0).cpu())
                offset += size
            status(stage='RemoteCLIP_visual_features',
                   done=min(batch_start + 8, len(ids)), total=len(ids))
        for first in range(0, len(cids), 128):
            batch = cids[first:first + 128]
            tokens = tokenizer([data.captions[i] for i in batch]).cuda()
            texts.append(F.normalize(model.encode_text(tokens), dim=-1).cpu())
            status(stage='RemoteCLIP_text_features',
                   done=min(first + 128, len(cids)), total=len(cids))
    payload = dict(image_ids=torch.tensor(ids), caption_ids=torch.tensor(cids),
                   images=torch.stack(images), texts=torch.cat(texts),
                   logit_scale=model.logit_scale.detach().cpu())
    assert payload['images'].shape == (len(ids), 512)
    assert payload['texts'].shape == (len(cids), 512)
    assert all(torch.isfinite(payload[key]).all() for key in ['images', 'texts', 'logit_scale'])
    torch.save(payload, dest / 'features.pt')
    metadata = dict(status='completed', backbone='RemoteCLIP ViT-B/32', dataset=data.dataset,
                    image_ids=ids, caption_ids=cids, lineage=lineage,
                    cache_sha256=sha(dest / 'features.pt'), checkpoint_sha256=sha(Path(checkpoint) / 'RemoteCLIP-ViT-B-32.pt'),
                    official_test_features_not_extracted=True, server_seconds=time.time() - started,
                    video_representation='L2-normalized mean of 8 uniformly sampled normalized frame embeddings',
                    paper='https://doi.org/10.1109/TGRS.2024.3390838',
                    adaptation='Temporal mean pooling and residual rank10 adapters are our adaptation, not the paper model architecture.',
                    costing='Training-media feature extraction charged once per independent method; evaluation-only media excluded from deployment accounting.')
    writer(dest / 'manifest.json', metadata)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return payload, metadata


def register(base):
    if (OUT / 'protocol.json').exists():
        return read(OUT / 'protocol.json')
    from data_schedule import dirichlet_partition, make_schedule
    from energy import PARAMETERS
    original = ROOT / 'results/overnight_energy_v2/uav_single_model_two_dataset_v1'
    old_protocol = read(original / 'protocol.json')
    geometry = read(base.OLD / 'mission_spec.json')
    input_hashes = {}
    for dataset in ['RSITMD', 'CapERA']:
        data = Data(dataset)
        dest = OUT / (dataset.lower() + '_seed0')
        if dataset == 'RSITMD':
            for name in ['plan', 'requests', 'initial_records', 'proxy_records', 'reference_records']:
                value = read(original / 'rsitmd_seed0' / (name + '.json'))
                write(dest / (name + '.json'), value)
            plan = read(dest / 'plan.json')
        else:
            rng = np.random.default_rng(0)
            train = np.array([item['imgid'] for item in data.items if item['split'] == 'train'])
            order = rng.permutation(train)
            validation, train = sorted(map(int, order[:256])), order[256:]
            labels = np.array([Path(item['filename']).stem.rsplit('_', 1)[0] for item in data.items])
            partitions = [train[part] for part in dirichlet_partition(labels[train], 10, .1, 0, minimum_size=16)]
            records = make_schedule(partitions, rounds=200, initial_rounds=20, requests=(),
                                    initial_fraction=1., batch_size=16, seed=0)
            seen = [set() for _ in range(10)]
            for record in records[:20]:
                for user, batch in enumerate(record['batches']):
                    seen[user].update(batch)
            eligible = [u for u in range(10) if len(seen[u]) >= 36]
            assert len(eligible) >= 5
            requesting = sorted(map(int, np.random.default_rng(231023).choice(eligible, 5, replace=False)))
            cohort = [[] for _ in range(10)]
            for user in requesting:
                pool = sorted(seen[user])
                cohort[user] = sorted(map(int, rng.choice(pool, len(pool) // 2, replace=False)))
            request_data = base.clip.stream.make_requests(cohort)
            requests = base.pilot.prepare_requests(request_data, geometry)
            forgotten = sorted({i for request in requests for i in request['images']})
            forgotten_set = set(forgotten)
            references = []
            for record in records:
                pools = [sorted(set(pool) - forgotten_set) for pool in record['retained']]
                batches = [np.random.default_rng(seed).choice(pool, min(16, len(pool)), replace=False).astype(int).tolist()
                           for pool, seed in zip(pools, record['batch_seeds'])]
                references.append(dict(round=record['round'], retained=pools, batches=batches))
            plan = dict(read(original / 'rsitmd_seed0/plan.json'), dataset='CapERA',
                        partitions=[list(map(int, part)) for part in partitions], training_images=len(train),
                        validation_images=validation, official_test_images=[item['imgid'] for item in data.items if item['split'] == 'test'],
                        unused_official_validation_images=[], forgotten_images=forgotten, forget_probe=forgotten,
                        mia_member_images=forgotten[:min(128, len(forgotten))],
                        mia_nonmember_images=validation[:min(128, len(forgotten))],
                        gallery_images=sorted(validation + forgotten),
                        annotation_sha256={split: sha(DATA / (split + '.json')) for split in ['train', 'test']})
            for name, value in [('requests', requests), ('initial_records', records[:20]),
                                ('proxy_records', base.core.proxy_schedule(records, forgotten_set)), ('reference_records', references)]:
                write(dest / (name + '.json'), value)
        plan.update(version='uav_remoteclip_capera_v1', model='RemoteCLIP ViT-B/32 + residual rank10 adapters',
                    scope='Fresh model-specific training; real UAV video-caption pairs for CapERA; paired single seed across four methods.')
        write(dest / 'plan.json', plan)
        requests = read(dest / 'requests.json')
        assert len(requests) == 45 and all(request['required_steps'] == 5 for request in requests)
        assert set(plan['gallery_images']).isdisjoint(plan['official_test_images'])
        assert set(plan['forgotten_images']).issubset(plan['gallery_images'])
        for path in dest.glob('*.json'):
            input_hashes[str(path.relative_to(ROOT))] = sha(path)
    model_info = read(CHECKPOINT / 'provenance.json')
    assert sha(CHECKPOINT / 'RemoteCLIP-ViT-B-32.pt') == model_info['sha256']
    protocol = dict(version='uav_remoteclip_capera_v1', registered_unix=time.time(), seed=0,
                    datasets=['RSITMD', 'CapERA'], model=dict(checkpoint=str(CHECKPOINT.relative_to(ROOT)), **model_info),
                    methods=old_protocol['methods'], config=old_protocol['config'], rounds=200,
                    common='Fresh initial20, proxy20 and reference200 for each dataset; same initial state, data/request schedule and radio channels across its four arms.',
                    preserved='Frozen existing four-method objectives, radio solvers, Q6 codec, 45 requests with five FU steps, shared cosine FL/FU rate.',
                    data='Official CapERA split; 256 validation clips drawn only from official training; videos are kept intact, never split by frame. Test remains untouched during adaptation.',
                    feature_model='Official RemoteCLIP ViT-B/32, frozen. 8 uniform frames per video, mean of normalized features; residual rank10 adapters. Our adaptation, not an exact paper reproduction.',
                    comparison='RSITMD is the historical remote-sensing comparator; CapERA supplies real UAV video-language data. New metrics use only this new model campaign.',
                    plotting=dict(layout='one row, two datasets', energy='bars: modeled communication + computing + flight',
                                  delay='bars', captions=False, labels={'proposed_fu_guard_cliperase':'Proposed','separate_aircomp_periodic':'S-AC','aircomp_ofdma_periodic':'AC-OF','sequential_digital_periodic':'S-OF'}),
                    energy_model=PARAMETERS, input_sha256=input_hashes,
                    source_sha256={str(path.relative_to(ROOT)): sha(path) for path in HERE.glob('*.py')},
                    frozen_source_sha256=old_protocol['source_hashes'])
    write(OUT / 'protocol.json', protocol)
    return protocol


def verify():
    protocol = read(OUT / 'protocol.json')
    for group in ['input_sha256', 'source_sha256', 'frozen_source_sha256']:
        for path, expected in protocol[group].items():
            assert sha(ROOT / path) == expected, path
    return protocol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    own = (OUT / 'campaign.lock').open('a')
    fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state(stage='preparing_assets')
    prepare_videos()
    base = baseline_module()
    try:
        # Check all actual media mappings and video decoding before training.
        for dataset in ['RSITMD', 'CapERA']:
            data = Data(dataset)
            frames, indices = data.frames(0)
            assert len(frames) == (8 if dataset == 'CapERA' else 1)
        write(OUT / 'data_preflight.json', dict(passed=True, datasets=['RSITMD', 'CapERA'],
                                               capera_train=1473, capera_test=1391, eight_frame_decode=True))
        if args.preflight_only:
            state(stage='data_preflight_passed')
            return
        from energy import self_check
        from accounting import instrumented_engine
        from uav_report import generate
        write(OUT / 'energy_preflight.json', self_check())
        protocol = register(base)
        verify()
        assert not (OUT / 'started.json').exists(), 'Refuse duplicate campaign'
        import torch
        assert torch.cuda.is_available()
        gpu = (ROOT / 'results/overnight_energy_v2/association_objective_gpu.lock').open('a')
        state(stage='waiting_for_shared_GPU_lock')
        fcntl.flock(gpu, fcntl.LOCK_EX)
        adapted_arm = base.repeat.clone_function(base.adapted_arm,
            [('frozen_CLIP_B32_residual_rank10', 'RemoteCLIP_B32_mean8_residual_rank10', 1)])
        pure = base.repeat.clone_function(base.pure_trajectory, [], {'state': state})
        run_dataset = base.repeat.clone_function(base.run_dataset,
            [("data=(RetrievalData if dataset=='RSITMD' else base.RetrievalData)()", 'data=Data(dataset)', 1)],
            {'OUT': OUT, 'Data': Data, 'prepare_features': prepare_features, 'verify': verify,
             'state': state, 'adapted_arm': adapted_arm, 'pure_trajectory': pure,
             'Engine': instrumented_engine(base.Engine)})
        write(OUT / 'started.json', dict(unix=time.time(), pid=os.getpid(), gpu=torch.cuda.get_device_name(),
                                        protocol_sha256=sha(OUT / 'protocol.json')))
        results = []
        for dataset in protocol['datasets']:
            state(stage='running_dataset', dataset=dataset)
            try:
                result = run_dataset(dataset, protocol)
                results.append(dict(dataset=dataset, arms=result))
            except Exception as error:
                write(OUT / (dataset.lower() + '_seed0') / 'failure.json',
                      dict(error=repr(error), traceback=traceback.format_exc(), unix=time.time()))
                results.append(dict(dataset=dataset, status='failed', error=repr(error)))
        write(OUT / 'finished.json', dict(results=results, unix=time.time()))
        report = generate(OUT)
        state(stage='terminal', results=results, report=report)
    finally:
        base.base.parent.common.base.close()


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        state(stage='failed', error=repr(error))
        write(OUT / 'failure.json', dict(error=repr(error), traceback=traceback.format_exc(), unix=time.time()))
        raise
