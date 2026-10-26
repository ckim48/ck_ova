"""Evaluation coverage repair only; training partitions and requests stay intact."""
import copy
import hashlib
import json


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def prepare_plan(original, requests):
    assert len(requests) == 45
    assert len({r['id'] for r in requests}) == 45
    assert all(r['required_steps'] == 5 for r in requests)
    requested = {i for r in requests for i in r['images']}
    assert requested == set(original['forgotten_images']), 'Request cohort changed'
    plan = copy.deepcopy(original)
    before = set(plan['gallery_images'])
    plan['gallery_images'] = sorted(before | set(plan['validation_images']) | requested)
    index = {im: k for k, im in enumerate(plan['gallery_images'])}
    for req in requests:
        assert all(i in index for i in req['images'])
    assert all(plan[k] == v for k, v in original.items() if k != 'gallery_images')
    report = dict(
        passed=True, original_gallery_count=len(before), gallery_count=len(index),
        request_image_count=len(requested), previously_missing_count=len(requested - before),
        original_gallery_sha256=digest(original['gallery_images']),
        repaired_gallery_sha256=digest(plan['gallery_images']),
        request_sample_sha256=digest([{k: r[k] for k in ('id', 'burst', 'user', 'images', 'required_steps')} for r in requests]),
        training_partitions_sha256=digest(plan['partitions']),
        changes='Evaluation gallery coverage only. Training partitions, samples and initial checkpoint unchanged.',
        comparison='Re-evaluate frozen reference and all reported candidates on this same gallery. Never reuse probabilities from a different gallery.',
    )
    return plan, report
