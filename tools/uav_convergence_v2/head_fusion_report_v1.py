"""All registered head-fusion arms, full native cohorts, explicit CPU accounting."""
import argparse
import html
import math
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import head_fusion_study_v1 as study
import combined_iq_experiment_a_report as previous
import forgetting_quality_v1 as audit
import joint_quality_audit_v1 as references
import manuscript_iq_report_v1 as curves
import paper_uav_charts as ledger

ROOT, read, write, sha = study.ROOT, study.read, study.write, study.sha
DEST = ROOT/'final_final_charts/head_fusion_v1'
METHODS = study.METHODS
DATASETS = ['CapERA', 'AirScape']
COLORS = ['#EFAA54', '#66B366', '#9370A8', '#4C72B0', '#C65365']
MARKERS = ['s', 'v', 'D', 'o', '^']
STYLES = ['--', '-.', ':', '-', (0, (3, 1, 1, 1))]
FIGURES = []
ACTIVE_FAMILY = None


def source(family, ds, mode):
    if family == 'current' and mode in ['combined_sum', 'iq_sum']:
        original = 'combined' if mode == 'combined_sum' else 'iq'
        return previous.source('baseline', ds, original)
    return study.OUT/family/mode/'cases/baseline'/(ds.lower()+'_seed0')/study.parent.learning.quality.PROPOSED


def evaluate():
    import torch
    torch.set_num_threads(1)
    for ds in DATASETS:
        base, data, parent, plan, payload, anchor = audit.runtime(ds)
        engine = study.parent.learning.quality.engine_wrapper(parent)(data, plan, payload)
        for family in ['current', 'manuscript']:
            paths = dict(common=anchor.parent/'reference200/final.pt', matched=(
                references.OUT/ds.lower()/'uniform_after100_reference/final.pt' if family == 'current'
                else audit.OUT/ds.lower()/'references/equal/final.pt'))
            if family == 'manuscript':
                assert all(len(set(p)-set(plan['forgotten_images'])) >= 16 for p in plan['partitions'])
            refs = {k: audit.per_video(engine, base.core.load_theta(p, engine.device), plan)[1] for k, p in paths.items()}
            for mode in METHODS:
                arm = source(family, ds, mode)
                dest = DEST/'evaluation'/family/ds.lower()/mode
                if not (arm/'final.pt').exists() or not (arm/'verification.json').exists():
                    write(dest/'unavailable.json', dict(source=str(arm.relative_to(ROOT)), reason='No completed verified run'))
                    continue
                if (dest/'source.json').exists():
                    assert read(dest/'source.json')['checkpoint_sha256'] == sha(arm/'final.pt')
                    continue
                theta = base.core.load_theta(arm/'final.pt', engine.device)
                metrics, _ = audit.store_eval(engine, theta, plan, dest, refs)
                if (dest/'unavailable.json').exists():
                    (dest/'unavailable.json').unlink()
                write(dest/'source.json', dict(checkpoint=str((arm/'final.pt').relative_to(ROOT)),
                    checkpoint_sha256=sha(arm/'final.pt'), references={k: dict(path=str(p.relative_to(ROOT)),
                        sha256=sha(p)) for k, p in paths.items()}))
                values = np.load(dest/'per_video.npz')
                requests = read(arm.parent/'requests.json')
                records = []
                for req in requests:
                    idx = [plan['gallery_images'].index(i) for i in req['images']]
                    records.append(dict(request=req['id'], common_KL=float(values['common_KL'][idx].mean()),
                        matched_KL=float(values['matched_KL'][idx].mean())))
                write(dest/'request_kl.json', records)
                print(family, ds, mode, 'mR', 100*metrics['retained']['mR'],
                    'matchedKL', metrics['forget']['matched_KL'], flush=True)


def validate(arm, family, mode):
    records = [read(p) for p in sorted((arm/'transactions').glob('*.json'))]
    works = {w['transaction']: w for w in [read(p) for p in sorted((arm/'work_attempts').glob('*.json'))]}
    settings = read(arm.parents[1]/'settings.json')
    ledger.physical.PARAMETERS = settings['energy_model']
    conf = settings['config']
    rows = []
    for tx in records:
        r = tx['radio']
        var = np.array(r['block_variance'])
        eps = np.array(r['component_epsilon'])
        weights = np.array(r['normalized_curvature'])
        assert np.allclose(var.sum(1), r['component_mse'], rtol=1e-10, atol=1e-30)
        assert np.all(var.sum(1) <= eps**2*(1+2e-6)+1e-30)
        assert np.all(weights@var.sum(0) <= np.array(r['normalized_distortion_limits'])*(1+2e-6)+1e-30)
        assert r['energy_bound_j'] <= .1*(1+1e-6)
        assert r['max_power_w'] <= .32*(1+1e-6)
        assert r['max_user_energy_j'] <= conf['available_energy_j']*(1+1e-6)
        assert r['uplink_s'] <= conf['round_uplink_deadline_s']+1e-12
        if mode.startswith('combined'):
            assert np.all(var[1] == 0)
        work = works[tx['sequence']]
        counts = np.zeros(10)
        counts[r['head']] = r.get('head_MAC_equivalents', 0)
        energies, seconds = ledger.physical.compute_cost(counts)
        head_s, head_j = float(seconds.max()), float(energies.sum())
        assert math.isclose(head_s, tx['timing'].get('head_fusion_s', 0.), abs_tol=1e-12)
        assert math.isclose(head_j, r.get('head_fusion_energy_j', 0.), abs_tol=1e-12)
        local_s = ledger.physical.compute_cost(work['compute']['device_macs'])[1].max()
        timing = tx['timing']
        assert math.isclose(timing['duration_s'], local_s+r['uplink_s']+head_s+timing['downlink_s'], abs_tol=1e-10)
        assert math.isclose(work['FL_receiver_apply_s'], tx['start_s']+local_s+r['uplink_s']+head_s, abs_tol=1e-9)
        assert work['compute']['manuscript_curvature']['hvp_calls'] in [2, 4]
        for local in tx['local']:
            if family == 'current':
                assert local['RM_weight'] == .95 and local['threshold'] == .01
            else:
                assert local['RM_weight'] == 1. and local['FU_variant'] == 'manuscript_cosine'
        f = r.get('head_fusion', {})
        clean = r.get('clean_global_diagnostic', {})
        rows.append(dict(family=family, method=mode, sequence=tx['sequence'], kind=tx['kind'],
            round=tx['completed_FL_rounds'], head=r['head'], n=r['frame_count'],
            RF_energy_j=r['energy_bound_j'], uplink_s=r['uplink_s'], head_s=head_s, head_j=head_j,
            global_conflict=clean.get('conflict'), global_cosine=clean.get('cosine'),
            local_conflicts=sum(x['conflict'] for x in r.get('local_fusion_diagnostics', [])),
            head_corrected=f.get('corrected', False), estimated_cosine=f.get('cosine'),
            L_multiplier=f.get('L_multiplier', 1.), U_multiplier=f.get('U_multiplier', 1.),
            pre_fusion_error=r.get('pre_fusion_realized_error_norm', r['realized_error_to_raw']),
            post_fusion_error=r.get('post_fusion_error_norm', r['realized_error_to_raw']),
            pre_fusion_MSE=float(var.sum())))
    assert len(rows) == 180 and sum(len(t['FU_request_ids']) for t in records) == 225
    return rows


def save(fig, axes, name, caption):
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc='upper center', bbox_to_anchor=(.54, 1.025),
        frameon=False, fontsize=12, handlelength=1.6, columnspacing=1.2, labelspacing=.25)
    fig.canvas.draw()
    positions = []
    for ax in axes:
        ticks = ax.get_yticks()
        assert np.allclose(np.diff(ticks), ticks[1]-ticks[0])
        assert np.allclose(ax.get_ylim(), ticks[[0, -1]])
        b = ax.get_position()
        positions.append([b.width*7.4*72, b.height*3.735*72])
    assert np.allclose(positions[0], positions[1])
    files = []
    for ext in ['pdf', 'png', 'svg']:
        path = DEST/(name+'.'+ext)
        fig.savefig(path, dpi=250, bbox_inches=None, facecolor='white')
        if ext == 'svg':
            path.write_text('\n'.join(x.rstrip() for x in path.read_text().splitlines())+'\n')
        files.append(dict(file=path.name, sha256=sha(path)))
        shutil.copyfile(path, ROOT/'final_final_charts/figures'/path.name)
    FIGURES.append(dict(name=name, caption=caption, axis_dimensions_pt=positions, files=files))
    plt.close(fig)


def notes(rows, changes):
    text = '''Head-side FL/FU fusion: complete development comparison

The current manuscript recovers FL and FU separately but adds them linearly.
Its c_L and c_U are channel/noise design coefficients. Under exact alignment,
they do not become task-combination weights. The new experiments explicitly
change the receiver update, preserving the previous results separately.

Methods:
  C-Sum: one real combined AirComp stream, linear sum.
  IQ-Sum: separate I/Q aggregates, linear sum.
  C-Local: two-task PCGrad at each UAV, followed by combined transmission.
  IQ-Head: two-task PCGrad after noisy global I/Q recovery.
  IQ-NR: fixed noise-margin and variance-regularized head PCGrad.
All five use a parallel transmission interval; this is not a serial-vs-parallel
comparison. PCGrad is an existing algorithm, credited in references.bib.

Families are reported separately:
  current: latest proxy-KL-gated FU, retention weight0.95, priority16 FL through
    round100 then uniform16, retained-pool weights.
  manuscript: supplied cosine-forgetting/retained-CE/embedding-consistency
    equations, uniform16 FL and minibatch weights. The numeric3/1/3 FU weights
    were predeclared from the earlier pilot; the pasted text did not give them.
Do not attribute a difference BETWEEN these families to I/Q or head fusion.

Each arm has200 total FL rounds (a shared20-round warm start and180 active
updates),45 requests and225 FU steps,10 UAVs and5 regions. Four current-family
sum controls are reused from the frozen experiment A;16 complete fresh models
were scheduled. Every attempted condition is retained. This is one previously
inspected development seed, not independent statistical confirmation.

Evaluation uses all247 deleted CapERA videos and358 deleted AirScape videos,
all native captions, and256 retained-validation videos per dataset. The full
galleries contain503 and614 videos. mR averages bidirectional R@1/5/10.
KL is D_KL(P_retrained || P_candidate) over both retrieval directions, averaged
by video/native captions. Both fixed common and policy-matched references are
provided. The manuscript matched reference uses uniform16/equal-client updates;
every retained client pool has at least16 samples, making minibatch weighting
equal to1/10. References never enter head decisions. These are adapter-only
retained-data retraining references; the pretrained encoders remain frozen.

The current-family inherited training proxy assumes the future deletion cohort
is known. That assumption is unchanged and is not a new head capability.

Local correction changes the actual waveform and its RF energy, including the
true cross term for a combined stream. Its error budgets still derive from the
uncorrected local updates. Head correction uses only received estimates and
designed branch-noise variances; clean aggregates are diagnostic-only.

Accounting includes active computation, uplink, broadcast, circuits and hovering.
Common warm-start and fixed travel energy remain excluded, as in experiment A.
Head fusion runs AFTER uplink and BEFORE broadcast, adding CPU and hovering time.
The metrics.csv total adds its explicit CPU energy to the inherited ledger;
legacy_ledger_total_j is retained for audit. Raw inherited energy_and_delay.json
does not include this new explicit CPU term; use the corrected exported totals.
Local/head arithmetic uses conservative scalar MAC equivalents under the existing
processor model, not measured onboard runtimes. The older generic physical_mode
metadata is overwritten by the parent runner; the actual transport is specified
by fusion_method and the implementation, and independently checked in
physical_checks.json. This metadata quirk does not change transmitted signals.

Pre-fusion component MSE and curvature constraints are checked on every round.
Nonlinear fusion can bias or amplify noise, so those constraints do not certify
post-fusion objective distortion. Transactions export both pre/post realized
errors. Their sums are realized diagnostics, not estimated conditional MSEs.
The addendum gives the changed ideal/global updates and explains why the old
zero-mean linear-sum theorem cannot be reused. The old SOCP remains a transport
design; no convexity claim is made for a new nonlinear output-quality constraint.

Files47--52 show the current family;53--58 show the manuscript family. Each
family includes matched/common KL, energy, delay, mR and loss. Exactly overlapping
curves are preserved without artificial offsets. Bars start at zero. The graphs
have identical panel geometry, regular endpoint ticks and no outcome labels.
The baseline and candidate names are deliberately not renamed Proposed.
Figure59 shows the matched-KL change relative to IQ-Sum; positive means worse.
Zero manuscript bars represent exactly unchanged KL, not missing runs.
The fitted-model equality audit is in model_equivalence_checks.json.

Reproduce:
  .venv/bin/python tools/uav_convergence_v2/head_fusion_study_v1.py register
  .venv/bin/python tools/uav_convergence_v2/head_fusion_study_v1.py queue
  .venv/bin/python tools/uav_convergence_v2/head_fusion_check_v1.py
  .venv/bin/python tools/uav_convergence_v2/head_fusion_report_v1.py evaluate
  .venv/bin/python tools/uav_convergence_v2/head_fusion_report_v1.py report
Completed or interrupted attempts are not silently overwritten. Existing native
features, plans, warm-start models and references are required; evaluation source
JSONs and the protocol preserve hashes. Original scripts were not modified.

Endpoints (mR in percent, energy in kJ, active delay in seconds):
'''
    for family in ['current', 'manuscript']:
        text += '\n'+family+'\n'
        text += 'Dataset  Method  mR  matched_KL  common_KL  energy_kJ  delay_s  head_corrections\n'
        for r in rows:
            if r['family'] == family:
                text += (f"{r['dataset']}  {METHODS[r['method']]}  {r['audited_retained_mR']:.6f}  "
                    f"{r['matched_KL']:.10g}  {r['common_KL']:.10g}  {r['total_energy_j']/1000:.6f}  "
                    f"{r['delay_s']:.6f}  {r['head_corrections']}\n")
    text += '\nHead changes relative to the same-family IQ-Sum (positive KL reduction is better):\n'
    for c in changes:
        if c['baseline'] == 'iq_sum' and c['method'] in ['iq_head_pc', 'iq_head_nr']:
            text += (f"{c['family']} {c['dataset']} {METHODS[c['method']]}: "
                f"mR change {c['mR_change_pp']:+.6f} percentage points; "
                f"matched KL reduction {c['matched_KL_reduction_percent']:+.6f}%; "
                f"energy reduction {c['energy_reduction_percent']:+.6f}%.\n")
    text += '\nInterpretation:\n'
    text += ('The new rule really uses separately recovered task vectors. In the current family, '
        'it does not improve final retained mR and slightly increases matched KL in this seed. '
        'In the manuscript family, neither head rule triggers; the final parameter tensors '
        'are exactly equal to IQ-Sum on both datasets. The task directions are mostly aligned '
        'in the current policy and always aligned at the global level in the manuscript policy. '
        'Conflict-only fusion therefore has little or no opportunity to act. These results '
        'do not support promoting this fusion rule as a demonstrated I/Q quality advantage. '
        'They also do not rule out other head policies or other operating conditions.\n')
    (DEST/'README.txt').write_text(text)


def model_checks():
    import torch
    checks = []
    for family in ['current', 'manuscript']:
        for ds in DATASETS:
            baseline = source(family, ds, 'iq_sum')/'final.pt'
            a = torch.load(baseline, map_location='cpu', weights_only=False)['theta']
            for mode in ['iq_head_pc', 'iq_head_nr']:
                arm = source(family, ds, mode)
                candidate = arm/'final.pt'
                b = torch.load(candidate, map_location='cpu', weights_only=False)['theta']
                records = [read(p) for p in sorted((arm/'transactions').glob('*.json'))]
                count = sum(t['radio'].get('head_fusion', {}).get('corrected', False) for t in records)
                equal = torch.equal(a, b)
                if count == 0:
                    assert equal, 'No-action head rule must preserve the original final tensor.'
                checks.append(dict(family=family, dataset=ds, method=mode,
                    head_corrections=count, identical_to_iq_sum=equal,
                    maximum_parameter_difference=float((a-b).abs().max()),
                    baseline_sha256=sha(baseline), candidate_sha256=sha(candidate)))
    write(DEST/'model_equivalence_checks.json', dict(passed=True, checks=checks))
    return checks


def report():
    global ACTIVE_FAMILY
    DEST.mkdir(parents=True, exist_ok=True)
    protocol = study.register()
    assert read(study.OUT/'queue_finished.json')
    assert read(study.OUT/'physical_checks.json')['passed']
    shutil.copyfile(study.OUT/'physical_checks.json', DEST/'physical_checks.json')
    ledger.source = lambda case, ds, mode: source(ACTIVE_FAMILY, ds, mode)
    ledger.METHODS = METHODS
    rows, txrows, series, missing = [], [], [], []
    for family in ['current', 'manuscript']:
        ACTIVE_FAMILY = family
        for ds in DATASETS:
            anchor = source('current', ds, 'combined_sum')
            for mode in METHODS:
                arm = source(family, ds, mode)
                evaluation = DEST/'evaluation'/family/ds.lower()/mode
                if not (evaluation/'metrics.json').exists():
                    missing.append(dict(family=family, dataset=ds, method=mode, source=str(arm.relative_to(ROOT))))
                    continue
                for name in ['plan.json', 'requests.json', 'initial20/final.pt', 'proxy20/final.pt', 'reference200/final.pt', 'feature_cache/features.pt']:
                    assert sha(arm.parent/name) == sha(anchor.parent/name), (family, ds, mode, name)
                assert read(arm.parents[1]/'settings.json') == read(anchor.parents[1]/'settings.json')
                flat = validate(arm, family, mode)
                txrows.extend(dict(dataset=ds, **t) for t in flat)
                row = ledger.ledger('baseline', ds, mode)
                head_j = sum(t['head_j'] for t in flat)
                row['legacy_ledger_total_j'] = row['total_energy_j']
                row['total_energy_j'] += head_j
                row['compute_j'] += head_j
                row.update(family=family, head_fusion_energy_j=head_j,
                    head_fusion_s=sum(t['head_s'] for t in flat),
                    head_corrections=sum(t['head_corrected'] for t in flat),
                    global_conflicts=(sum(bool(t['global_conflict']) for t in flat)
                        if any(t['global_conflict'] is not None for t in flat) else None),
                    local_conflicts=sum(t['local_conflicts'] for t in flat),
                    uplink_s=sum(t['uplink_s'] for t in flat),
                    pre_fusion_error_sq=sum(t['pre_fusion_error']**2 for t in flat),
                    post_fusion_error_sq=sum(t['post_fusion_error']**2 for t in flat))
                metrics = read(evaluation/'metrics.json')
                requests = read(evaluation/'request_kl.json')
                row.update(matched_KL=metrics['forget']['matched_KL'], common_KL=metrics['forget']['common_KL'],
                    worst_matched_KL=max(r['matched_KL'] for r in requests),
                    audited_retained_mR=100*metrics['retained']['mR'],
                    audited_retained_loss=metrics['retained']['loss'], forget_mR=100*metrics['forget']['mR'])
                assert abs(row['retained_mR_percent']-row['audited_retained_mR']) < 2e-6
                rows.append(row)
                for p in curves.curves(arm):
                    v = p['evaluation']
                    series.append(dict(family=family, dataset=ds, method=mode, round=p['round'],
                        mR=100*v['retained']['mR'], loss=v['retained']['loss']))
    changes = []
    for a in rows:
        for baseline in ['combined_sum', 'iq_sum', 'combined_local_pc']:
            b = next((r for r in rows if r['family'] == a['family'] and r['dataset'] == a['dataset'] and r['method'] == baseline), None)
            if b is None or a['method'] == baseline:
                continue
            changes.append(dict(family=a['family'], dataset=a['dataset'], method=a['method'], baseline=baseline,
                mR_change_pp=a['audited_retained_mR']-b['audited_retained_mR'],
                matched_KL_reduction_percent=100*(1-a['matched_KL']/b['matched_KL']),
                common_KL_reduction_percent=100*(1-a['common_KL']/b['common_KL']),
                energy_reduction_percent=100*(1-a['total_energy_j']/b['total_energy_j']),
                delay_reduction_percent=100*(1-a['delay_s']/b['delay_s'])))
    for name, data in [('metrics', rows), ('transactions', txrows), ('curves', series), ('changes', changes)]:
        ledger.write_csv(DEST/(name+'.csv'), data)
        if name in ['metrics', 'changes']:
            write(DEST/(name+'.json'), data)
    write(DEST/'protocol.json', protocol)
    write(DEST/'verification.json', dict(passed=not missing, completed_arms=len(rows), expected_arms=20,
        missing=missing, math_checks=read(study.OUT/'math_checks.json'),
        physical_checks='physical_checks.json',
        scope='Native cohorts, fixed matched references, executed schedule, RF caps, pre-fusion MSE/distortion and explicit post-uplink head CPU costs checked. No certified post-fusion distortion guarantee.'))
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    for fi, family in enumerate(['current', 'manuscript']):
        for mi, (key, scale, label, decimals, suffix) in enumerate([
            ('matched_KL', 1000, r'Forget-set KL ($\times10^{-3}$)', 2, 'matched_kl'),
            ('common_KL', 1000, r'Forget-set KL ($\times10^{-3}$)', 2, 'common_kl'),
            ('total_energy_j', .001, 'Total energy (kJ)', 0, 'energy'),
            ('delay_s', 1, 'Active delay (s)', 0, 'delay')]):
            fig, axes = previous.canvas(label)
            for ax, ds in zip(axes, DATASETS):
                vals = []
                for j, mode in enumerate(METHODS):
                    row = next((r for r in rows if r['family'] == family and r['dataset'] == ds and r['method'] == mode), None)
                    if row is None:
                        continue
                    value = row[key]*scale
                    vals.append(value)
                    ax.bar(j, value, width=.66, color=COLORS[j], edgecolor='#282828', lw=.7,
                        hatch=['\\\\', '..', '++', '//', 'xx'][j], label=METHODS[mode])
                ax.set_xlim(-.6, 4.6)
                ax.set_xticks([])
                if vals:
                    previous.yscale(ax, vals, zero=True, decimals=decimals)
            name = f'{47+6*fi+mi:02d}_head_fusion_{family}_{suffix}_1x2'
            save(fig, axes, name, f'Comparison of {suffix.replace("_", " ")} under the {family} learning policy on CapERA and AirScape.')
        for mi, (key, label, decimals) in enumerate([('mR', 'Retained mR (%)', 1), ('loss', 'Validation loss', 2)]):
            fig, axes = previous.canvas(label)
            for ax, ds in zip(axes, DATASETS):
                vals = []
                for j, mode in enumerate(METHODS):
                    points = {p['round']: p[key] for p in series if p['family'] == family and p['dataset'] == ds and p['method'] == mode}
                    x = sorted(points)
                    if not x:
                        continue
                    y = [points[k] for k in x]
                    vals += y
                    ax.plot(x, y, color=COLORS[j], ls=STYLES[j], lw=1.4, marker=MARKERS[j],
                        markevery=[i for i, k in enumerate(x) if k % 25 == 0], ms=4.2, mfc='white', mew=1., label=METHODS[mode])
                ax.set_xlim(0, 200)
                ax.set_xticks([0, 50, 100, 150, 200])
                ax.set_xlabel('FL round', fontsize=14)
                if vals:
                    previous.yscale(ax, vals, decimals=decimals)
            name = f'{51+6*fi+mi:02d}_head_fusion_{family}_{key.lower()}_1x2'
            save(fig, axes, name, f'Comparison of {label.lower()} over FL rounds under the {family} learning policy on CapERA and AirScape.')
    fig, axes = previous.canvas('KL change from IQ-Sum (%)')
    for ax, ds in zip(axes, DATASETS):
        values = []
        for j, mode in enumerate(['iq_head_pc', 'iq_head_nr']):
            vals = [-next(c['matched_KL_reduction_percent'] for c in changes
                if c['family'] == family and c['dataset'] == ds and c['method'] == mode
                and c['baseline'] == 'iq_sum') for family in ['current', 'manuscript']]
            values += vals
            ax.bar(np.array([0., 1.])+(j-.5)*.3, vals, width=.3,
                color=COLORS[3+j], edgecolor='#282828', lw=.7,
                hatch=['//', 'xx'][j], label=METHODS[mode])
        ax.set_xlim(-.5, 1.5)
        ax.set_xticks([0, 1], ['Current', 'Manuscript'])
        previous.yscale(ax, values, zero=True, decimals=2)
    save(fig, axes, '59_head_fusion_kl_change_1x2',
        'Change in matched-retraining forget-set KL relative to IQ-Sum on CapERA and AirScape; a positive value denotes higher divergence.')
    write(DEST/'figures.json', FIGURES)
    notes(rows, changes)
    model_checks()
    (DEST/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Head fusion audit</title><style>body{font:16px Georgia;max-width:1100px;margin:30px auto}img{width:100%}section{margin:30px 0}</style><h1>Head-side FL/FU fusion</h1><p>All 20 predeclared development arms. No outcome filtering. See README.txt and metrics.csv.</p>'+''.join(
        '<section><a href="'+r['name']+'.pdf"><img src="'+r['name']+'.png"></a><p>'+html.escape(r['caption'])+'</p></section>' for r in FIGURES))
    print('Report:', DEST, 'completed', len(rows), 'missing', len(missing), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['evaluate', 'report'])
    a = p.parse_args()
    evaluate() if a.action == 'evaluate' else report()
