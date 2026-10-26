"""Queued independent cache/vector audit of prespecified FU61 task pilot."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    for attempt in range(10):
        try: return json.loads(path.read_text())
        except json.JSONDecodeError:
            if attempt == 9: raise
            time.sleep(.02)


def identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except FileNotFoundError: return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True); parser.add_argument('--wait-pid', type=int, required=True)
    parser.add_argument('--plan-only', action='store_true'); args = parser.parse_args(); root = args.project.resolve()
    family = root/'results/overnight_energy_v2'; source = family/'fu_quality_v22'; out = family/'fu_quality_audit_v25'; out.mkdir(exist_ok=True)
    def write(name, value):
        target = out/name; tmp = target.with_suffix('.tmp'); tmp.write_text(json.dumps(value, indent=2)+'\n'); tmp.replace(target)
    plan = dict(version='fu_quality_audit_v25', wait_pid=args.wait_pid, code_sha256=sha(__file__),
        pilot_plan_sha256=sha(source/'plan.json'), pilot_selected_plan_sha256=sha(source/'selected_plan.json'),
        checks=['All19 complete update vectors: SHA256 and independent physical reconstruction, including paid half coefficients and original8bitU.',
                'All57 cohort predictions: normalization, finiteness, independently recomputed loss/accuracy/ideal-stepKL/final-retrainKL and prediction disagreement.',
                'Verify6 energy-only selected designs really minimize the full fair58-case eligible pool and match independent physical certificate paths/hashes.',
                'Aggregate prespecified3common-noise pairs by mode/rank, keep both reference directions. No model inference, test arrays, settings choice, GPU or automatic promotion.'])
    if (out/'plan.json').exists(): assert read(out/'plan.json') == plan
    else: write('plan.json', plan)
    if args.plan_only: print(json.dumps(plan)); return
    initial = identity(args.wait_pid); write('status.json', dict(status='waiting_for_v22_complete', wait_pid=args.wait_pid))
    while initial is not None and identity(args.wait_pid) == initial: time.sleep(15)
    assert read(source/'status.json')['status'] == 'completed_cpu_request61_quality_no_gpu_promotion'
    pilot = read(source/'plan.json'); selected_plan = read(source/'selected_plan.json'); rows = read(source/'results.json'); assert len(rows) == 19
    assert sha(source/'plan.json') == plan['pilot_plan_sha256'] and sha(source/'selected_plan.json') == plan['pilot_selected_plan_sha256']
    for n, h in pilot['core_sha256'].items(): assert sha(root/n) == h
    for n, h in pilot['code_sha256'].items(): assert sha(root/'tools/overnight_energy_v2/fu_quality_v22'/n) == h
    source_vectors = family/'public_history_fu_v19/screen/vectors.npz'; assert sha(source_vectors) == pilot['input_vector_sha256']
    cfg = read(family/'public_history_fu_v19/screen/plan.json')['config']; groups = pilot['groups']
    radio_rows = read(family/'public_history_fu_v19/radio/results.json')+read(family/'public_history_fu_v19b/results.json'); assert len(radio_rows) == 58
    cert = read(family/'transport_certificate_v20/fu_quality_v22.json'); assert cert['checked_saved_candidates'] == 48
    with np.load(source_vectors) as data:
        learning, unlearning, channels = [data[k].copy() for k in ['learning', 'unlearning', 'channels']]
        basis, scale = data['rank3_basis'].copy(), float(data['public_scale'])
    dimension = learning.shape[1]; width = cfg['block_dimension']; sigma = np.sqrt(10**((cfg['noise_dbm']-30)/10))
    block = lambda x: np.pad(x, ((0, 0), (0, (-dimension) % width))).reshape(len(x), -1, width)
    ideal_update = (learning+unlearning).sum(axis=0)
    verified_vectors = []
    with np.load(source/'update_vectors.npz') as vectors:
        np.testing.assert_allclose(vectors['ideal_one_step'], ideal_update, rtol=1e-12, atol=1e-15)
        for name in vectors.files:
            assert hashlib.sha256(np.ascontiguousarray(vectors[name]).tobytes()).hexdigest() == selected_plan['update_sha256'][name]
        for selected in selected_plan['selected']:
            mode, rank = selected['mode'], selected['rank']; path = root/selected['design_path']; assert sha(path) == selected['design_sha256']
            eligible = [x for x in radio_rows if x['relative_tolerance'] == .2 and x['mode'] == mode and x['rank'] == rank and x['status'] == 'feasible_radio_only']
            incumbent = min(eligible, key=lambda x: x['energy_bound_j']); assert incumbent == selected['row']
            assert any(x['design_sha256'] == sha(path) for x in cert['certificates'])
            with np.load(path) as data: d = {k: data[k].copy() for k in data.files}
            raw = learning+unlearning if mode == 'mixed_aircomp' else learning
            if rank:
                coeff = ((raw@basis)/scale).astype(np.float16).astype(float)*scale; raw = raw-coeff@basis.T
            analog = block(raw); digital = block(np.zeros_like(unlearning) if mode == 'mixed_aircomp' else unlearning)
            levels = 127; qscale = (np.max(np.abs(digital), axis=-1)/levels).astype(np.float32).astype(float)
            decoded = np.clip(np.rint(digital/np.where(qscale > 0, qscale, 1.)[..., None]), -levels, levels)*qscale[..., None]
            users = d['users']; rho = np.linalg.norm(analog[users], axis=-1); na = d['symbols']
            b = np.divide(d['analog_rms']*np.sqrt(na), rho, out=np.zeros_like(rho), where=rho > 0)
            c = d['receiver']/(sigma*np.sqrt(na)); h = channels[users, int(d['head']), None]*sigma
            gain = c*h*b
            bias = ((gain-1)[..., None]*analog[users]+decoded[users]-digital[users]).sum(axis=0)
            for seed in pilot['noise_seeds']:
                name = f'{mode}_rank{rank}_noise{seed}'; normal = np.random.default_rng(seed).normal(size=bias.shape)
                error = (bias+c[:, None]*sigma/np.sqrt(2)*normal).reshape(-1)[:dimension]
                np.testing.assert_allclose(vectors[name], ideal_update+error, rtol=1e-10, atol=1e-14)
                verified_vectors.append(name)
    assert len(verified_vectors) == 18
    canonical = root/'results/energy_quality_v1'; full = read(canonical/'cohorts.json')['groups']; reference = {}; labels = {}
    assert sha(canonical/'cohorts.json') == pilot['canonical_cohorts_sha256']
    with np.load(canonical/'retrain_reference_ideal.npz') as data:
        for group, spec in groups.items():
            lookup = {z: i for i, z in enumerate(full[group]['ids'])}; indices = [lookup[z] for z in spec['ids']]
            reference[group] = data[group][indices].copy(); labels[group] = data[group+'_labels'][indices].copy()
    with np.load(source/'ideal_one_step.npz') as data: ideal = {group: data[group].copy() for group in groups}
    verified = []
    for row in rows:
        name = row['method']; path = source/(name+'.npz')
        with np.load(path) as data:
            for group, spec in groups.items():
                p = data[group]; q = reference[group]; target = ideal[group]; n = len(spec['ids'])
                assert p.shape == q.shape == (n, 100) and np.all(np.isfinite(p)) and np.all(p > 0)
                np.testing.assert_allclose(p.sum(axis=1), 1., rtol=1e-12, atol=1e-14)
                metrics = dict(loss=float(-np.log(p[np.arange(n), labels[group]]).mean()), accuracy=float(np.mean(p.argmax(axis=1) == labels[group])))
                for key, value in metrics.items(): np.testing.assert_allclose(value, row[group][key], rtol=1e-12, atol=1e-14)
                for label, ref in [('ideal_step', target), ('final_retrain_descriptive_only', q)]:
                    kl = max(0., float(np.mean(np.sum(ref*(np.log(ref)-np.log(p)), axis=1))))
                    disagreement = float(np.mean(ref.argmax(axis=1) != p.argmax(axis=1)))
                    np.testing.assert_allclose([kl, disagreement], [row[group]['gap_to_'+label]['kl_reference_to_candidate'], row[group]['gap_to_'+label]['prediction_disagreement']], rtol=1e-12, atol=1e-14)
                    metrics[label+'_kl'] = kl; metrics[label+'_prediction_disagreement'] = disagreement
                verified.append(dict(method=name, cohort=group, probability_file_sha256=sha(path), **metrics))
    summary = []; paired = []
    for mode in pilot['modes']:
        for group in groups:
            summaries = {}
            for rank in pilot['ranks']:
                entries = [v for v in verified if v['method'].startswith(f'{mode}_rank{rank}_') and v['cohort'] == group]; assert len(entries) == 3
                keys = ['loss', 'accuracy', 'ideal_step_kl', 'final_retrain_descriptive_only_kl', 'ideal_step_prediction_disagreement', 'final_retrain_descriptive_only_prediction_disagreement']
                means = {k: float(np.mean([v[k] for v in entries])) for k in keys}; summaries[rank] = means
                summary.append(dict(mode=mode, rank=rank, cohort=group, noise_draws=3, **means))
            paired.append(dict(mode=mode, cohort=group, rank3_minus_rank0_mean={k: summaries[3][k]-summaries[0][k] for k in keys}))
    result = dict(status='passed19_vectors_and57_cohort_predictions_and6_energy_only_selections', at_kst=datetime.now().isoformat(timespec='seconds'),
        source_sha256=sha(__file__), certificates=verified, summary=summary, paired=paired,
        limits='Fixed actualFU61 vectors,3commonnoise draws,200percohort. Finalretrain100 differs in stage: its KL is descriptive, not deletion/quality equivalence. Allreferences and failures kept. No MIA/newinference/test/GPU/trainingpromotion.')
    write('results.json', result); write('status.json', dict(status=result['status'], at_kst=result['at_kst']))
    print(json.dumps({k: v for k, v in result.items() if k != 'certificates'}))


if __name__ == '__main__': main()
