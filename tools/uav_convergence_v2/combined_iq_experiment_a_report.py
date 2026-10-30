"""Export all experiment-A outcomes; no smoothing, selection or forced ranks."""
import argparse
import html
import math
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import combined_iq_experiment_a as study
import forgetting_quality_v1 as audit
import joint_quality_audit_v1 as references
import manuscript_iq_report_v1 as old_report
import paper_uav_charts as ledger

ROOT, read, write, sha = study.ROOT, study.read, study.write, study.sha
DEST = ROOT/'final_final_charts/experiment_a_combined_iq'
DATASETS = ['CapERA', 'AirScape']
METHODS = {'iq': 'I/Q AirComp', 'combined': 'Combined AirComp'}
COLORS = ['#4C72B0', '#EFAA54']
FIGURES = []


def source(case, ds, mode):
    return study.OUT/mode/'cases/baseline'/(ds.lower()+'_seed0')/study.learning.quality.PROPOSED


def evaluate():
    import torch
    torch.set_num_threads(1)
    for ds in DATASETS:
        base, data, parent, plan, payload, anchor = audit.runtime(ds)
        engine = study.learning.quality.engine_wrapper(parent)(data, plan, payload)
        refpaths = dict(common=anchor.parent/'reference200/final.pt',
            matched=references.OUT/ds.lower()/'uniform_after100_reference/final.pt')
        refs = {}
        for label, p in refpaths.items():
            _, refs[label] = audit.per_video(engine, base.core.load_theta(p, engine.device), plan)
        for mode in METHODS:
            arm = source('baseline', ds, mode)
            theta = base.core.load_theta(arm/'final.pt', engine.device)
            dest = DEST/'evaluation'/ds.lower()/mode
            metrics, _ = audit.store_eval(engine, theta, plan, dest, refs)
            write(dest/'source.json', dict(checkpoint=str((arm/'final.pt').relative_to(ROOT)),
                checkpoint_sha256=sha(arm/'final.pt'),
                references={k: dict(path=str(p.relative_to(ROOT)), sha256=sha(p)) for k, p in refpaths.items()}))
            print(ds, mode, {g: {k: v for k, v in vals.items() if k in ['mR', 'matched_KL', 'common_KL']} for g, vals in metrics.items()}, flush=True)


def validate(arm, mode):
    settings = read(arm.parents[1]/'settings.json')['config']
    records = [read(p) for p in sorted((arm/'transactions').glob('*.json'))]
    works = {r['transaction']: r for r in [read(p) for p in sorted((arm/'work_attempts').glob('*.json'))]}
    rows = []
    for t in records:
        r = t['radio']
        v = np.array(r['block_variance'])
        eps = np.array(r['component_epsilon'])
        w = np.array(r['normalized_curvature'])
        assert r['experiment_A_mode'] == mode
        assert np.allclose(v.sum(1), r['component_mse'], rtol=1e-10, atol=1e-30)
        assert np.all(v.sum(1) <= eps**2*(1+2e-6)+1e-30)
        assert np.allclose(w@v.sum(0), r['distortion'], rtol=1e-10, atol=1e-30)
        assert np.all(w@v.sum(0) <= np.array(r['normalized_distortion_limits'])*(1+2e-6)+1e-30)
        assert r['energy_bound_j'] <= .1*(1+1e-6)
        assert r['max_power_w'] <= .32*(1+1e-6)
        assert r['max_user_energy_j'] <= settings['available_energy_j']*(1+1e-6)
        assert r['uplink_s'] <= settings['round_uplink_deadline_s']+1e-12
        assert r['frame_count']*settings['block_dimension']/settings['symbol_rate_hz'] <= settings['coherence_time_s']+1e-12
        assert math.isclose(sum(r['user_RF_energy_j']), r['energy_bound_j'], rel_tol=1e-10, abs_tol=1e-20)
        assert all(b <= a*(1+1e-8) for a, b in zip(r['BCD_objectives'], r['BCD_objectives'][1:]))
        if mode == 'combined':
            assert np.all(v[1] == 0)
        work = works[t['sequence']]['compute']['manuscript_curvature']
        assert work['hvp_calls'] in [2, 4] and not work['certified']
        for local in t['local']:
            assert local['RM_weight'] == .95 and local['threshold'] == .01
        rows.append(dict(sequence=t['sequence'], round=t['completed_FL_rounds'], kind=t['kind'],
            head=r['head'], n=r['frame_count'], uplink_s=r['uplink_s'], RF_energy_j=r['energy_bound_j'],
            aggregate_MSE=float(v.sum()), allowed_total_MSE=float(np.sum(eps**2)),
            max_power_w=r['max_power_w'], HVP_calls=work['hvp_calls']))
    assert len(rows) == 180
    assert sum(len(t['FU_request_ids']) for t in records) == 225
    return rows


def canvas(label):
    fig = plt.figure(figsize=(7.4, 3.735))
    axes = [fig.add_axes(p) for p in [(.095, .22, .385, .65), (.595, .22, .385, .65)]]
    for i, ax in enumerate(axes):
        ax.set_ylabel(label, fontsize=14)
        ax.tick_params(labelsize=12, pad=3)
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='#c7cbd1', ls=':', lw=.5)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(.8)
        p = ax.get_position()
        fig.text((p.x0+p.x1)/2, .025, f'({chr(97+i)}) {DATASETS[i]}', ha='center', fontsize=14)
    return fig, axes


def yscale(ax, values, zero=False, decimals=1):
    lo, hi = min(values), max(values)
    span = max(hi-lo, .01*max(abs(lo), abs(hi)), 10**(-decimals))
    lo = 0 if zero else lo-.08*span
    hi += (.08*max(abs(hi), span) if zero else .12*span)
    ticks = MaxNLocator(nbins=4, min_n_ticks=4).tick_values(lo, hi)
    if zero:
        ticks = ticks[ticks >= 0]
    ax.set_ylim(ticks[0], ticks[-1])
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.'+str(decimals)+'f'))


def save(fig, axes, name, caption):
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc='upper center', bbox_to_anchor=(.54, 1.015),
        frameon=False, fontsize=13, handlelength=1.8, columnspacing=1.8)
    fig.canvas.draw()
    geometry = []
    for ax in axes:
        for dim in ['x', 'y']:
            ticks = getattr(ax, 'get_'+dim+'ticks')()
            limits = getattr(ax, 'get_'+dim+'lim')()
            if dim == 'y':
                assert np.allclose(np.diff(ticks), ticks[1]-ticks[0])
                assert np.allclose(limits, ticks[[0, -1]])
        box = ax.get_position()
        geometry.append(dict(width_pt=box.width*7.4*72, height_pt=box.height*3.735*72))
    assert np.allclose(list(geometry[0].values()), list(geometry[1].values()))
    files = []
    for ext in ['pdf', 'png', 'svg']:
        p = DEST/(name+'.'+ext)
        fig.savefig(p, dpi=250, bbox_inches=None, facecolor='white')
        if ext == 'svg':
            p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
        files.append(dict(file=p.name, sha256=sha(p)))
        shutil.copyfile(p, ROOT/'final_final_charts/figures'/p.name)
    plt.close(fig)
    FIGURES.append(dict(name=name, caption=caption, axes=geometry, files=files))


def report():
    DEST.mkdir(parents=True, exist_ok=True)
    protocol = study.register()
    assert read(study.OUT/'physical_checks.json')['passed']
    ledger.source = source
    ledger.METHODS = METHODS
    rows, series, txrows = [], [], []
    for ds in DATASETS:
        arms = [source('baseline', ds, mode) for mode in METHODS]
        for name in ['plan.json', 'requests.json', 'initial20/final.pt', 'proxy20/final.pt', 'reference200/final.pt', 'feature_cache/features.pt']:
            assert sha(arms[0].parent/name) == sha(arms[1].parent/name), name
        assert read(arms[0].parents[1]/'settings.json') == read(arms[1].parents[1]/'settings.json')
        for mode in METHODS:
            arm = source('baseline', ds, mode)
            assert read(arm/'verification.json')['passed']
            flat = validate(arm, mode)
            txrows += [dict(dataset=ds, method=mode, **r) for r in flat]
            row = ledger.ledger('baseline', ds, mode)
            metrics = read(DEST/'evaluation'/ds.lower()/mode/'metrics.json')
            row.update(matched_KL=metrics['forget']['matched_KL'], common_KL=metrics['forget']['common_KL'],
                audited_retained_mR=100*metrics['retained']['mR'], audited_retained_loss=metrics['retained']['loss'],
                uplink_s=sum(r['uplink_s'] for r in flat), RF_energy_j=sum(r['RF_energy_j'] for r in flat))
            assert abs(row['retained_mR_percent']-row['audited_retained_mR']) < 2e-6
            assert abs(row['forget_KL']-row['common_KL']) < 1e-6
            rows.append(row)
            points = {p['round']: p['evaluation'] for p in old_report.curves(arm)}
            for k, v in sorted(points.items()):
                series.append(dict(dataset=ds, method=mode, round=k, mR=100*v['retained']['mR'],
                    loss=v['retained']['loss'], common_KL=v['forget'].get('KL_retrain_to_candidate')))
        # Before the first FU request the single active stream must match exactly.
        for seq in range(1, 9):
            a, b = [read(arm/'transactions'/f'{seq:04d}.json') for arm in arms]
            if a['kind'] == b['kind'] == 'fl':
                assert np.allclose(a['radio']['block_variance'], b['radio']['block_variance'], rtol=1e-8, atol=1e-20)
    changes = []
    for ds in DATASETS:
        a = next(r for r in rows if r['dataset'] == ds and r['method'] == 'iq')
        b = next(r for r in rows if r['dataset'] == ds and r['method'] == 'combined')
        changes.append(dict(dataset=ds, mR_change_pp=a['audited_retained_mR']-b['audited_retained_mR'],
            matched_KL_reduction_percent=100*(1-a['matched_KL']/b['matched_KL']),
            common_KL_reduction_percent=100*(1-a['common_KL']/b['common_KL']),
            energy_reduction_percent=100*(1-a['total_energy_j']/b['total_energy_j']),
            delay_reduction_percent=100*(1-a['delay_s']/b['delay_s'])))
    ledger.write_csv(DEST/'metrics.csv', rows)
    ledger.write_csv(DEST/'curves.csv', series)
    ledger.write_csv(DEST/'transactions.csv', txrows)
    write(DEST/'metrics.json', rows)
    write(DEST/'changes.json', changes)
    write(DEST/'protocol.json', protocol)
    shutil.copyfile(study.OUT/'physical_checks.json', DEST/'physical_checks.json')
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    for name, key, scale, label, decimals in [
        ('35_combined_iq_forget_kl_1x2', 'matched_KL', 1000, r'Forget-set KL ($\times 10^{-3}$)', 2),
        ('37_combined_iq_energy_1x2', 'total_energy_j', .001, 'Total energy (kJ)', 0),
        ('38_combined_iq_delay_1x2', 'delay_s', 1, 'Active delay (s)', 0),
        ('39_combined_iq_common_kl_1x2', 'common_KL', 1000, r'Forget-set KL ($\times 10^{-3}$)', 2)]:
        fig, axes = canvas(label)
        for ax, ds in zip(axes, DATASETS):
            values = []
            for i, mode in enumerate(METHODS):
                val = next(r[key]*scale for r in rows if r['dataset'] == ds and r['method'] == mode)
                values.append(val)
                ax.bar(i, val, width=.55, color=COLORS[i], edgecolor='#282828', lw=.7,
                    hatch=['//', '\\\\'][i], label=METHODS[mode])
            ax.set_xlim(-.65, 1.65)
            ax.set_xticks([])
            scale_values = [r[key]*scale for r in rows] if key in ['total_energy_j', 'delay_s'] else values
            yscale(ax, scale_values, zero=True, decimals=decimals)
        caption = ('Comparison of forget-set KL divergence from '+('matched' if key == 'matched_KL' else 'common')+' retraining between combined and I/Q AirComp on CapERA and AirScape.' if 'KL' in key else
                   'Comparison of '+('total energy consumption' if key == 'total_energy_j' else 'active FL/FU delay')+' between combined and I/Q AirComp on CapERA and AirScape.')
        save(fig, axes, name, caption)
    for name, key, label, decimals in [('36_combined_iq_round_mr_1x2', 'mR', 'Retained mR (%)', 1),
        ('40_combined_iq_round_loss_1x2', 'loss', 'Validation loss', 2)]:
        fig, axes = canvas(label)
        for ax, ds in zip(axes, DATASETS):
            values = []
            for i, mode in enumerate(METHODS):
                ps = [r for r in series if r['dataset'] == ds and r['method'] == mode]
                x, y = [p['round'] for p in ps], [p[key] for p in ps]
                values += y
                ax.plot(x, y, c=COLORS[i], ls=['-', '--'][i], lw=1.5, label=METHODS[mode],
                    marker=['o', 's'][i], markevery=[j for j, k in enumerate(x) if k % 25 == 0],
                    ms=4.8, mfc='white', mew=1.1)
            ax.set_xlim(0, 200)
            ax.set_xticks([0, 50, 100, 150, 200])
            ax.set_xlabel('FL round', fontsize=14)
            yscale(ax, values, decimals=decimals)
        save(fig, axes, name, 'Comparison of '+('retained mR' if key == 'mR' else 'validation loss')+' over FL rounds between combined and I/Q AirComp on CapERA and AirScape.')
    write(DEST/'verification.json', dict(passed=True, source_sha256=sha(Path(__file__)),
        pairs=2, completed_arms=4, all_transactions_checked=len(txrows), changes=changes, figures=FIGURES,
        identical_initial_inputs=True, shared_retraining_reference=True, same_learning_policy=True,
        single_development_seed=True, all_outcomes_reported=True))
    (DEST/'captions.tex').write_text('\n\n'.join('% '+f['name']+'\n\\caption{'+f['caption']+'}' for f in FIGURES)+'\n')
    body = ['Experiment A: Combined AirComp versus I/Q AirComp', '',
        'Same retain0.95/late_uniform100 learning policy in both arms; 200 FL rounds, 45 requests and 225 FU steps. Same frozen features, initialization, sampling seeds, channels, power caps, head candidates, common task-distortion budget and charged HVP work. Both methods process FL and FU simultaneously.',
        'Combined transmits the true sum on one real stream. I/Q transmits separate updates on two quadratures of one complex stream and enforces additional component-MSE bounds. Both finally apply the summed update. An I/Q gain is therefore not assumed.',
        'The shared warm start contains FL rounds1–20. Energy and delay cover the ensuing180 FL rounds and225 FU cycles, including computation/curvature, uplink/downlink, circuits and hovering. Common warm-start cost and fixed travel energy are excluded. No hardware wall-clock claim is made.',
        'The main KL plot uses the SAME retained-only late_uniform100 retraining model for both methods. The common-reference KL plot is also included. Neither reference is used as a training teacher. Lower KL is closer to this reference, not a certified deletion guarantee.',
        'Curves use measured values, with markers at multiples of25; no smoothing, manual shifts or artificial error bars. This is one paired development seed; small differences do not establish statistical superiority.', '',
        'I/Q minus Combined / percentage reductions (negative reduction means I/Q uses more or has higher KL):']
    body += [str(r) for r in changes]
    (DEST/'README.txt').write_text('\n\n'.join(body)+'\n')
    cards = ''.join('<section><h2>'+html.escape(f['caption'])+'</h2><a href="'+f['name']+'.pdf"><img src="'+f['name']+'.png"></a></section>' for f in FIGURES)
    (DEST/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Experiment A</title><style>body{max-width:1100px;margin:30px auto;font:16px system-ui;color:#222}img{width:100%}h2{font-size:18px}section{margin:35px 0}pre{white-space:pre-wrap}</style><h1>Experiment A: Combined AirComp vs I/Q AirComp</h1><pre>'+html.escape('\n\n'.join(body))+'</pre>'+cards)
    print(changes, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['evaluate', 'report', 'all'], default='all', nargs='?')
    a = p.parse_args()
    if a.action in ['evaluate', 'all']:
        evaluate()
    if a.action in ['report', 'all']:
        report()
