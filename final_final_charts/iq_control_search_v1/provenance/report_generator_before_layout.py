"""Publish the complete bounded I/Q controller search, including failures."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import time
import numpy as np
import progress_iq_pilot_v1 as progress
import adaptive_iq_pilot_v2 as rms

ROOT, read, write, sha = progress.ROOT, progress.read, progress.write, progress.sha
DEST = ROOT/'final_final_charts/iq_control_search_v1'
DATASETS = ['CapERA', 'AirScape']


def controls():
    import torch
    progress.audit.runtime('CapERA')
    from train import update_theta
    rms.original.configure()
    torch.set_num_threads(1)
    for ds in DATASETS:
        dest = DEST/'controls'/ds.lower()
        if (dest/'finished.json').exists():
            continue
        assert not (dest/'started.json').exists(), 'Preserve interrupted control.'
        write(dest/'started.json', dict(unix=time.time(), source_sha256=sha(Path(__file__))))
        base, data, parent, plan, payload, anchor = progress.audit.runtime(ds)
        engine = progress.learning.quality.engine_wrapper(parent)(data, plan, payload)
        engine.joint_base = base
        theta = base.core.load_theta(anchor.parent/'initial20/final.pt', engine.device)
        engine.set_teacher(theta)
        txs = [read(p) for p in sorted((anchor/'transactions').glob('*.json'))]
        queue = base.RoundQueue(read(anchor.parent/'requests.json'))
        for fl, tx in enumerate(txs):
            queue.sync(fl, tx['start_s'], 0.)
            engine.start_work()
            l, u, local = progress.learning.vectors(engine, theta, plan, queue,
                tx['start_s'], fl, tx['kind'], [], None, {}, False)
            assert not local and not np.any(u)
            theta = update_theta(theta, l.sum(0))
            engine.finish_work()
            # Keep the same request release/exclusion schedule, but apply no FU.
            queue.commit(tx['FU_request_ids'], tx['start_s'], tx['end_s'])
        assert len(txs) == 180
        refs = {k: progress.audit.per_video(engine, base.core.load_theta(p, engine.device), plan)[1]
            for k, p in dict(common=anchor.parent/'reference200/final.pt',
                matched=progress.audit.OUT/ds.lower()/'references/equal/final.pt').items()}
        metrics, _ = progress.audit.store_eval(engine, theta, plan, dest, refs)
        base.core.save_theta(dest/'final.pt', theta, 200)
        result = dict(dataset=ds, retained_mR_percent=100*metrics['retained']['mR'],
            matched_KL=metrics['forget']['matched_KL'], common_KL=metrics['forget']['common_KL'],
            scope='Paired exact FL-only control: identical warm20, uniform16 minibatches, '
                  'request release/exclusion schedule, and 180 further FL rounds; zero FU gradients.')
        write(dest/'finished.json', result)
        print(result, flush=True)


def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter
    DEST.mkdir(parents=True, exist_ok=True)
    selections = dict(rms=read(rms.OUT/'selection.json'), progress=read(progress.OUT/'selection.json'))
    allrows = []
    for family, selection in selections.items():
        assert len(selection['results']) == 24
        assert not any(r['joint_winner'] for r in selection['rankings'])
        src = rms.OUT if family == 'rms' else progress.OUT
        write(DEST/'protocols'/(family+'.json'), read(src/'protocol.json'))
        write(DEST/(family+'_selection.json'), selection)
        for item in selection['results']:
            row = dict(family=family, **item)
            arm = src/'runs'/item['dataset'].lower()/item['setting']/item['mode']
            row['checkpoint_sha256'] = sha(arm/'final.pt')
            row['source'] = str(arm.relative_to(ROOT))
            allrows.append(row)
    fields = sorted(set().union(*(r.keys() for r in allrows)))
    with (DEST/'all_48_pilot_results.csv').open('w') as f:
        w = csv.DictWriter(f, fields); w.writeheader(); w.writerows(allrows)
    plt.rcParams.update({'font.family':'serif', 'font.serif':['Times New Roman','DejaVu Serif'],
        'font.size':12, 'axes.labelsize':14, 'pdf.fonttype':42, 'ps.fonttype':42,
        'axes.linewidth':1.0, 'hatch.linewidth':.7})
    methods = ['combined', 'combined_local', 'iq_progress', 'fl_only']
    labels = ['C-Sum', 'C-Local', 'I/Q-Control', 'FL-only']
    colors = ['#EFAA54', '#66B366', '#4C72B0', '#9370A8']
    figures, comparisons = [], []
    for number, metric, ylabel, scale in [(60,'mR','Retained mR (%)',1.),
            (61,'matched_KL','Forget-set KL divergence',1000.)]:
        fig = plt.figure(figsize=(7.4,3.735))
        for di, ds in enumerate(DATASETS):
            ax = fig.add_axes((.095+.5*di,.22,.385,.65))
            armrows = {m: next(r for r in selections['progress']['results']
                if r['dataset']==ds and r['setting']=='r1_p1' and r['mode']==m) for m in methods[:-1]}
            control = read(DEST/'controls'/ds.lower()/'finished.json')
            vals = [armrows[m][metric]*scale for m in methods[:-1]]
            vals += [(control['retained_mR_percent'] if metric=='mR' else control[metric])*scale]
            for j, value in enumerate(vals):
                ax.bar(j, value, width=.62, facecolor=colors[j], edgecolor='#333333',
                    linewidth=.85, hatch=['//','..','xx','++'][j], label=labels[j])
            lo, hi = (min(vals)-.12, max(vals)+.12) if metric=='mR' else (0.,max(vals)*1.06)
            ticks = MaxNLocator(nbins=5).tick_values(lo, hi)
            if metric != 'mR': ticks = ticks[ticks >= 0]
            ax.set_yticks(ticks); ax.set_ylim(ticks[0],ticks[-1])
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f' if metric=='mR' else '%.1f'))
            ax.set_xticks(range(4),labels,fontsize=10)
            ax.set_ylabel(ylabel+(r' ($\times 10^{-3}$)' if scale==1000 else ''))
            ax.grid(axis='y',ls=':',color='#b9b9b9',lw=.7); ax.set_axisbelow(True)
            ax.text(.5,-.27,f'({chr(97+di)}) {ds}',transform=ax.transAxes,ha='center',va='top',fontsize=14)
        name = f'{number:02d}_progress_{metric.lower()}_exact_1x2'
        for ext in ['pdf','svg','png']: fig.savefig(DEST/(name+'.'+ext),dpi=230)
        plt.close(fig); figures.append(name)
    for ds in DATASETS:
        rows = {m: next(r for r in selections['progress']['results']
            if r['dataset']==ds and r['setting']=='r1_p1' and r['mode']==m) for m in methods[:-1]}
        a = rows['iq_progress']
        for m in methods[:-2]:
            b = rows[m]
            comparisons.append(dict(dataset=ds, reference=m,
                mR_change_percentage_points=a['mR']-b['mR'],
                matched_KL_reduction_percent=100*(1-a['matched_KL']/b['matched_KL']),
                common_KL_reduction_percent=100*(1-a['common_KL']/b['common_KL'])))
    write(DEST/'comparisons.json', comparisons)
    write(DEST/'status.json', dict(completed=True, development_runs=48, matched_RMS_retrains=8,
        paired_FL_only_controls=2, joint_winners=0, adopted_method=None,
        physical_training_runs=0, figures=figures,
        decision='No candidate passed the shared two-dataset mR/common-KL/matched-KL criterion. '
                 'Stop this bounded search and retain the existing Proposed. Physical optimizer '
                 'checks are not end-to-end training or performance confirmation.',
        source_sha256=sha(Path(__file__))))
    lines = ['I/Q controller search: complete development results (2026-10-30/31)', '',
        '48 exact-aggregation development runs; eight fresh RMS-policy retrains; two paired FL-only controls.',
        'All eight candidate configurations failed the joint retained-mR/common-KL/matched-KL criterion.',
        'Existing Proposed and manuscript performance figures are unchanged. No candidate is promoted.', '',
        '60/61 show progress controller r1_p1, selected by the declared worst-margin rule.',
        'This rule changes the learning update: the head minimally changes the FL step subject to',
        'estimated FL preservation and an FU progress target, with a norm trust region.',
        'C-Sum has the same preknown FU scaling; C-Local uses the same rule locally before aggregation.',
        'The experiment uses the manuscript cosine FU loss (3 forget +1 retained CE +3 consistency).',
        'These are exact-aggregation pilots. No wireless/energy advantage or statistical significance is claimed.',
        'The existing linear-sum distortion guarantee does not apply to this nonlinear head rule.', '',
        'Quality tradeoffs for the selected progress rule:']
    for r in comparisons:
        lines.append(f"{r['dataset']} vs {r['reference']}: matched-KL reduction {r['matched_KL_reduction_percent']:.2f}%; "
            f"retained-mR change {r['mR_change_percentage_points']:+.5f} percentage points.")
    lines += ['', 'FL-only control is included because closer retraining KL alone is not proof of successful forgetting.',
        'Its requests still exclude the same samples after arrival; FU gradients alone are disabled.',
        'All 48 outcomes, including less favorable configurations, are in all_48_pilot_results.csv.',
        'The same previously inspected cohorts are used for candidate development. There is no held-out test claim.',
        'Do not compare the absolute values to the old Proposed as if only the radio changed:',
        'this pilot uses the supplied manuscript objective, whereas earlier charts use a different FU policy.', '',
        'RMS failed startup attempts and reference zero-LR fix are preserved under results/adaptive_iq_pilot_v1_20261030.',
        'The subsequent progress protocol explicitly deferred RMS physical confirmation after failed selection.',
        'The eight frozen-update radio/accounting checks are under results/adaptive_iq_physical_v1_20261030.',
        'They validate implementation mechanics only; no physical training campaign was run.', '',
        'Reproduce controls/report: tools/uav_convergence_v2/iq_control_search_report_v1.py controls / report',
        'RMS motivation: Adaptive Federated Learning Over the Air, IEEE TSP 2025,',
        'https://doi.org/10.1109/TSP.2025.3585002 ; https://arxiv.org/pdf/2403.06528',
        'Our AWGN, FL/FU, reset and norm-preserving rule differs from that paper; its theorem is not imported.']
    (DEST/'report.txt').write_text('\n'.join(lines)+'\n')
    write(DEST/'physical_checks_not_performance.json',read(ROOT/'results/adaptive_iq_physical_v1_20261030/physical_checks.json'))
    blocks = ''.join(f'<p><a href="{s}.pdf"><img width="800" src="{s}.png"></a></p>' for s in figures)
    (DEST/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>I/Q control search</title>'
        '<p>Exact-aggregation development pilot: no joint winner; existing Proposed unchanged.</p>'
        '<p><a href="report.txt">Scope and all findings</a> · <a href="all_48_pilot_results.csv">All 48 runs</a></p>'+blocks)
    print(json.dumps(read(DEST/'status.json'),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['controls','report']);a=p.parse_args()
    {'controls':controls,'report':report}[a.action]()
