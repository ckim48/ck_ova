"""Publish the actual adaptive FU outcomes; preserve the previous final bundle."""
import argparse
import collections
import hashlib
import html
import json
import math
from pathlib import Path
import shutil
import tarfile
import time

import numpy as np
import final_charts as f
import adaptive_fu as study

ROOT, OUT, NEW, OLD = study.ROOT, study.OUT, study.NEW, study.PROPOSED
DATASETS = study.DATASETS
VIEW = OUT/'chart_view'
STAGE = OUT/'final_chart_staging'
ARCHIVE = ROOT/'final_chart_archive/20261028_before_adaptive'
PEERS = ROOT/'results/learning_ablation_v1_20261027'
REQUESTS = ROOT/'results/final_request_sweep_v1_20261027'
read, write, sha = study.read, study.write, study.sha
chart, plots = f.chart, f.plots
METHODS = dict(f.METHODS)
METHODS = {NEW if m==OLD else m:label for m,label in METHODS.items()}


def load(path):
    if path == ROOT/'selected_graph/data/service_energy_delay.json':
        path=STAGE/'data/service_energy_delay.json'
    path=path.resolve()
    f.SOURCES[str(path.relative_to(ROOT))]=sha(path)
    return read(path)


def build_view():
    mapping=[]
    for case in study.CASES:
        cid=case['id'];dest=VIEW/'cases'/cid
        actual=OUT/'cases'/cid
        for p in actual.glob('*.json'):study.link(p,dest/p.name)
        peerroot=(PEERS if cid=='baseline' or cid.startswith('bandwidth_') else REQUESTS)/'cases'/cid
        for ds in DATASETS:
            dd=actual/(ds.lower()+'_seed0');pd=peerroot/dd.name;vd=dest/dd.name
            for p in dd.iterdir():
                if p.name!=NEW:study.link(p,vd/p.name)
            assert read(dd/'plan.json')==read(pd/'plan.json')
            assert read(dd/'requests.json')==read(pd/'requests.json')
            for method in METHODS:
                source=dd/NEW if method==NEW else pd/method
                if cid=='request_bursts_45' and method in f.broadcast_fix.METHODS:
                    source=f.broadcast_fix.OUT/'cases'/cid/dd.name/method
                assert (source/'outcome.json').exists(),source
                study.link(source,vd/method)
                mapping.append(dict(case=cid,dataset=ds,method=method,label=METHODS[method],
                                    source=str(source.resolve().relative_to(ROOT)),status=load(source/'outcome.json')['status']))
    study.link(REQUESTS/'pre_fu_reference.json',VIEW/'pre_fu_reference.json')
    write(OUT/'chart_source_mapping.json',mapping)
    return mapping


def configure():
    f.OUT=f.BASELINE=VIEW;f.DEST=STAGE;f.METHODS=METHODS
    f.SOURCES={};f.SERIES=[];f.FIGURES=[];f.VALIDATED=set();f.load=load
    f.configure();chart.SOURCE=VIEW
    f.broadcast_fix.ENERGY=plots.physical
    f.broadcast_fix.ORIGINAL_COMMUNICATION=plots.physical.communication
    plots.physical.communication=f.broadcast_fix.communication
    for p in [Path(__file__),Path(study.__file__),Path(f.__file__),Path(plots.__file__),
              Path(chart.__file__),Path(f.broadcast_fix.__file__),OUT/'protocol.json',OUT/'verification.json']:
        load(p) if p.suffix=='.json' else f.SOURCES.update({str(p.relative_to(ROOT)):sha(p)})


def save(fig,axes,name,title,supplementary=False,status=None):
    record=plots.save(fig,axes,name,status)
    # plots.save historically stored basename only, so repair nested export paths.
    for item in record['files']:item['file']=str(Path(name).with_suffix(Path(item['file']).suffix))
    record.update(title=title,supplementary=supplementary)
    f.FIGURES.append(record)


def trajectories():
    result={}
    initial={}
    for ds in DATASETS:
        path=ROOT/'results/uav_convergence_v2_20261027'/(ds.lower()+'_pilot/rounds.json')
        initial[ds]=load(path)[0]['evaluation']['retained']
        for m in METHODS:
            events,prefix,costs=chart.convergence(ds,m,f.SOURCES)
            arm=VIEW/'cases/baseline'/(ds.lower()+'_seed0')/m
            timeline=load(arm/'mission_timeline.json')
            evals=[load(p) for p in sorted((arm/'evaluations').glob('*.json'))]
            final=load(arm/'service_completion.json')
            if final['sequence']!=evals[-1]['sequence']:evals.append(final)
            for r in evals:
                tx=load(arm/'transactions'/f"{r['sequence']:04d}.json")
                assert r['evaluation']==tx['evaluation']
                assert math.isclose(r['mission_clock_s'],tx['end_s'],abs_tol=1e-9)
            result[(ds,m)]=dict(events=events,prefix=prefix,evals=evals,timeline=timeline)
    return result,initial


def convergence():
    paths,initial=trajectories()
    horizon=math.ceil(max(p['events'][-1]['active_s'] for p in paths.values())/100)*100
    specs=[('01_round_retained_mr_1x2','round','mR','Retained mR (%)',100.,False),
           ('02_time_retained_mr_1x2','time','mR','Retained mR (%)',100.,False),
           ('03_time_forget_kl_1x2','time','KL','Forget-set KL',1.,False),
           ('supplementary/round_validation_loss_1x2','round','loss','Validation loss',1.,True),
           ('supplementary/time_validation_loss_1x2','time','loss','Validation loss',1.,True),
           ('supplementary/round_forget_kl_1x2','round','KL','Forget-set KL',1.,True)]
    for name,domain,metric,ylabel,scale,supplementary in specs:
        fig,axes=chart.style.canvas(ylabel);panel=[]
        for ax,ds in zip(axes,DATASETS):
            vals=[]
            for n,(m,label) in enumerate(METHODS.items()):
                p=paths[(ds,m)]
                if metric=='KL':
                    x=[r['completed_FL_rounds'] if domain=='round' else chart.active_time(p['timeline'],r['mission_clock_s']) for r in p['evals']]
                    y=[r['evaluation']['forget']['KL_retrain_to_candidate'] for r in p['evals']]
                elif domain=='round':
                    points=[(0,initial[ds])]+[(r['round'],r['evaluation']) for r in p['prefix']['initial_points']]
                    points += [(e['round'],e['evaluation']) for e in p['events'][1:]]
                    x=[r[0] for r in points];y=[r[1][metric]*scale for r in points]
                else:
                    x=[r['active_s'] for r in p['events']];y=[r['evaluation'][metric]*scale for r in p['events']]
                assert all(a<=b for a,b in zip(x,x[1:]))
                if domain=='time':dx,dy=chart.draw_time(ax,x,y,n,label,horizon)
                else:chart.style.curve(ax,x,y,n,label);dx,dy=x,y
                f.SERIES.append(dict(figure=name,dataset=ds,method=m,x=x,y=y,display_x=dx,display_y=dy,
                    completion_x=x[-1],final_value_held=domain=='time',metric=metric))
                vals+=y
            chart.regular_axis(ax,'x',[0,50,100,150,200] if domain=='round' else list(range(0,horizon+1,100)),0)
            ax.set_xlabel('FL round' if domain=='round' else 'Time (s)');panel.append(vals)
        if metric=='KL':
            ticks=[chart.nice_ticks([0.,*v],bins=4,simple=True) for v in panel]
            precision=max(chart.tick_decimals(t) for t in ticks)
            for ax,t in zip(axes,ticks):chart.regular_axis(ax,'y',t,precision)
        else:chart.finish_axes(axes,panel,decimals=2 if metric=='loss' else 1)
        save(fig,axes,name,f'{ylabel} versus '+('FL rounds' if domain=='round' else 'active service time'),supplementary)


def bandwidth():
    rows=[]
    for value,cid in [(1,'baseline'),(2,'bandwidth_2'),(4,'bandwidth_4')]:
        case=VIEW/'cases'/cid;parameters=load(case/'settings.json')['energy_model']
        for ds in DATASETS:
            dd=case/(ds.lower()+'_seed0');prefix=plots.prefix_energy(dd,parameters,f.SOURCES)
            for m in METHODS:
                rows.append(dict(bandwidth_MHz=value,dataset=ds,method=m,
                    **plots.service_ledger(dd,m,parameters,prefix,f.SOURCES)))
    write(STAGE/'data/service_energy_delay.json',rows);f.csv_file(STAGE/'data/service_energy_delay.csv',rows)
    for name,metric,ylabel,scale in [('04_bandwidth_total_energy_1x2','total_energy_j','Total energy (kJ)',.001),
                                    ('05_bandwidth_delay_1x2','delay_s','Delay (s)',1.)]:
        fig,axes=chart.style.canvas(ylabel);panel=[];statuses=[]
        for ax,ds in zip(axes,DATASETS):
            vals=[]
            for n,(m,label) in enumerate(METHODS.items()):
                records=[r for r in rows if r['dataset']==ds and r['method']==m]
                x=[r['bandwidth_MHz'] for r in records];y=[r.get(metric)*scale if r['status']=='completed' else None for r in records]
                chart.style.curve(ax,x,[v if v is not None else math.nan for v in y],n,label)
                f.SERIES.append(dict(figure=name,dataset=ds,method=m,x=x,y=y,display_x=x,display_y=y,status=[r['status'] for r in records]))
                for r in records:
                    if r['status']!='completed':
                        ax.plot([r['bandwidth_MHz']],[1.04],transform=ax.get_xaxis_transform(),marker='x',color=chart.COLORS[n],clip_on=False)
                        statuses.append(dict(dataset=ds,method=m,x=r['bandwidth_MHz'],status=r['status'],axes_y_fraction=1.04,metric_y=None))
                vals+=y
            chart.regular_axis(ax,'x',[1,2,3,4],0);ax.set_xlabel('Bandwidth (MHz)');panel.append(vals)
        chart.finish_axes(axes,panel,decimals=0)
        save(fig,axes,name,ylabel+' versus bandwidth',status=statuses)
    return rows


def supplemental():
    old_methods=plots.METHODS
    plots.METHODS={NEW:'Proposed',OLD:'Previous'}
    for metric,ylabel,name in [('mR','Retained mR (%)','adaptive_vs_previous_mr_1x2'),
                               ('KL','Forget-set KL','adaptive_vs_previous_kl_1x2')]:
        name='supplementary/'+name;fig,axes=chart.style.canvas(ylabel);panel=[]
        for ax,ds in zip(axes,DATASETS):
            vals=[]
            for n,(m,label) in enumerate(plots.METHODS.items()):
                arm=(OUT if m==NEW else PEERS)/'cases/baseline'/(ds.lower()+'_seed0')/m
                rows=[load(p) for p in sorted((arm/'evaluations').glob('*.json'))]
                x=[r['completed_FL_rounds'] for r in rows]
                y=[r['evaluation']['retained']['mR']*100 if metric=='mR' else r['evaluation']['forget']['KL_retrain_to_candidate'] for r in rows]
                chart.style.curve(ax,x,y,n,label);vals+=y
                f.SERIES.append(dict(figure=name,dataset=ds,method=m,x=x,y=y,display_x=x,display_y=y))
            chart.regular_axis(ax,'x',[0,50,100,150,200],0);ax.set_xlabel('FL round');panel.append(vals)
        if metric=='KL':
            ticks=[chart.nice_ticks([0.,*v],bins=4,simple=True) for v in panel]
            precision=max(chart.tick_decimals(t) for t in ticks)
            for ax,t in zip(axes,ticks):chart.regular_axis(ax,'y',t,precision)
        else:chart.finish_axes(axes,panel,decimals=1)
        save(fig,axes,name,ylabel+': adaptive versus previous Proposed',True)
    plots.METHODS={NEW:'Mean', 'weight_range':'Range'}
    name='supplementary/adaptive_retention_weight_1x2';fig,axes=chart.style.canvas(r'Retention weight $\lambda_R$')
    audits=[]
    for ax,ds in zip(axes,DATASETS):
        arm=OUT/'cases/baseline'/(ds.lower()+'_seed0')/NEW
        work=[load(p) for p in sorted((arm/'work_attempts').glob('*.json'))];group=collections.defaultdict(list)
        for w in work:
            for r in w['local']:group[w['FL_round']].append(r['RM_weight'])
        x=sorted(group);means=[float(np.mean(group[t])) for t in x];lo=[min(group[t]) for t in x];hi=[max(group[t]) for t in x]
        chart.style.curve(ax,x,means,0,'Mean');ax.fill_between(x,lo,hi,color=chart.COLORS[0],alpha=.18,label='Range')
        chart.regular_axis(ax,'x',[0,50,100,150,200],0);chart.regular_axis(ax,'y',[.25,.5,.75,1.],2);ax.set_xlabel('FL round')
        f.SERIES.append(dict(figure=name,dataset=ds,method=NEW,x=x,y=means,display_x=x,display_y=means,weight_min=lo,weight_max=hi))
        local=[r for w in work for r in w['local']]
        audits.append(dict(dataset=ds,FU_steps=len(local),min_weight=min(lo),max_weight=max(hi),mean_weight=float(np.mean([r['RM_weight'] for r in local])),
            repaired_steps=sum(r['radio_weight_repaired'] for r in local),actual_feedback_active=len(set(r['RM_weight'] for r in local))>1))
    save(fig,axes,name,'Automatic retention weight: mean and range across active requests',True)
    plots.METHODS=old_methods;write(STAGE/'data/controller_summary.json',audits)
    return audits


def paired_summary():
    rows=[]
    for ds in DATASETS:
        for seed in [0,1,2]:
            for m in [NEW,OLD]:
                root=(OUT if m==NEW else PEERS) if seed==0 else OUT/f'sampling_seed_{seed}'
                dd=root/'cases/baseline'/(ds.lower()+'_seed0');arm=dd/m
                outcome=load(arm/'outcome.json');assert outcome['status']=='completed'
                e=load(arm/'manifest.json')['evaluation'];params=load(dd.parent/'settings.json')['energy_model']
                prefix=plots.prefix_energy(dd,params,f.SOURCES);ledger=plots.service_ledger(dd,m,params,prefix,f.SOURCES)
                rows.append(dict(dataset=ds,sampling_seed=seed,method=m,retained_mR_percent=100*e['retained']['mR'],
                    forget_KL=e['forget']['KL_retrain_to_candidate'],worst_request_KL=e['worst_request_KL'],
                    MIA_AUC_gap=load(arm/'mia.json')['auc_gap_vs_retrain'],energy_kJ=ledger['total_energy_j']/1000,delay_s=ledger['delay_s']))
    contrasts=[]
    for ds in DATASETS:
        for seed in [0,1,2]:
            a=next(r for r in rows if r['dataset']==ds and r['sampling_seed']==seed and r['method']==NEW)
            b=next(r for r in rows if r['dataset']==ds and r['sampling_seed']==seed and r['method']==OLD)
            contrasts.append(dict(dataset=ds,sampling_seed=seed,mR_change_pp=a['retained_mR_percent']-b['retained_mR_percent'],
                KL_reduction_percent=100*(1-a['forget_KL']/b['forget_KL']),energy_change_percent=100*(a['energy_kJ']/b['energy_kJ']-1),
                delay_change_percent=100*(a['delay_s']/b['delay_s']-1)))
    summary=[]
    for ds in DATASETS:
        group=[r for r in contrasts if r['dataset']==ds]
        summary.append(dict(dataset=ds,**{k:dict(mean=float(np.mean([r[k] for r in group])),min=min(r[k] for r in group),max=max(r[k] for r in group))
            for k in ['mR_change_pp','KL_reduction_percent','energy_change_percent','delay_change_percent']}))
    for name,data in [('paired_seed_metrics',rows),('paired_seed_contrasts',contrasts)]:
        write(STAGE/'data'/f'{name}.json',data);f.csv_file(STAGE/'data'/f'{name}.csv',data)
    write(STAGE/'data/paired_seed_summary.json',dict(rows=summary,
        scope='Three post-warm minibatch/radio seeds, paired within each dataset. Same warm20, data split, features, proxy and retrain references. Conditional robustness check, not three independent dataset/training replications or test-set inference.'))
    return rows,contrasts,summary


def documentation(ledgers,contrasts,summary,controller):
    figures=sorted(f.FIGURES,key=lambda x:x['name'])
    lines=['# Final charts: adaptive retrieval FU','',
        'Updated 2026-10-28. Left: CapERA. Right: AirScape. [Browse figures](index.html). '+
        'Proposed now uses the automatic retention-weight controller in every current primary figure. '+
        'Comparators keep their recorded objectives, data and physical constraints.','',
        '| Dataset | mR change (pp) | KL reduction (%) | Energy change (%) | Delay change (%) |',
        '|---|---:|---:|---:|---:|']
    for r in contrasts:
        if r['sampling_seed']==0:lines.append(f"| {r['dataset']} | {r['mR_change_pp']:+.3f} | {r['KL_reduction_percent']:.2f} | {r['energy_change_percent']:+.3f} | {r['delay_change_percent']:+.3f} |")
    lines += ['', 'Changes above are against previous Proposed at 1 MHz, seed 0. Lower KL is better; a negative mR change is a loss of utility. '+
        'The automatic controller substantially reduces retrain-distance in this development experiment but does not improve both evaluation metrics simultaneously. '+
        'It is a new tradeoff, not evidence of universal superiority. All paired seed outcomes and all sweep conditions are retained.', '',
        '## Figures','', '| Figure | Content | Files |','|---|---|---|']
    for item in figures:
        n=item['name'];lines.append(f"| {n} | {item['title']} | [PNG]({n}.png) · [PDF]({n}.pdf) · [SVG]({n}.svg) |")
    lines += ['', '## Automatic rule and evaluation','',
        'The FU objective is 3 KL_forget(proxy20 || current) + lambda_R CE_retain + 3 KL_retain(request-start teacher || current). '+
        'At each request step, the rule compares relative retained CE change against the request teacher with relative forget proxy-KL change against the first request step. '+
        'Their clipped difference is smoothed with EMA 0.5. The previous weight is multiplied by exp(0.5 EMA), bounded to [0.25, 1], with each request initialized at 0.5. '+
        'If the resulting radio design is infeasible, the rule tries bounded weight floors 0.5 and 1 and stops at the first feasible design; all attempts are logged. '+
        'No physical constraint is relaxed. Both datasets use exactly the same constants and code.', '',
        'Training retained CE and proxy KL are surrogates for held-out mR and final retrain KL. Validation mR/KL, the reference200 model and dataset names are never controller inputs. '+
        'No guarantee that lowering either training surrogate improves its evaluation metric is implied. The clean proxy20 inherits the existing offline assumption that the final forget cohort is known before the mission. '+
        'This is not a solution for an unknown future deletion cohort. The design follows earlier development results; there is no independent test-set selection claim.', '',
        'Retained mR averages R@1, R@5 and R@10 in both retrieval directions. Loss is common held-out retained retrieval negative log likelihood after both FL and FU updates. '+
        'Forget-set KL is KL(reference200 || candidate), averaged over both directions. It starts at the first recorded observation, round 25; no round-0 KL is invented. '+
        'Early KL also reflects FL progress and future requests. These metrics and the MIA AUC gap are empirical diagnostics, not a deletion or privacy certificate. '+
        'The Pre-FU table row is shared round20, not a 200-round no-FU control. Retrain is the retained-set round200 reference.', '',
        '## Costs and comparisons','',
        'Total energy sums active FL/FU communication, computing and hovering across ten UAVs. Travel, collection and preflight preparation are excluded. '+
        'Time is modeled active service time including scheduling waits, not GPU wall time. Extra controller computations, retrieval matrices, teacher forwards and the extra retained backward are charged. '+
        'Vector repair is conservatively precharged. As for existing methods, resource-optimizer wall time has no calibrated UAV energy model; radio trial counts are disclosed.', '',
        'Primary charts use post-warm seed0. The supplement reports paired seeds 0, 1, 2 for adaptive and previous Proposed. '+
        'Only minibatch and radio randomness change; warm20, split, feature cache, proxy20 and reference200 are shared. These are conditional robustness checks, not independent training/split replicates. '+
        'All methods retain 200 FL rounds and 225 FU steps across 45 requests. The comparison changes both the FU objective and its controller; '+
        'it does not isolate the causal effect of adaptive weighting from the objective change.', '',
        'Bandwidth measurements are at 1, 2 and 4 MHz; the tick at 3 is not another measurement. Request-load points are 1, 3, 5, 9 and 15 requests per release, '+
        'with the same 45 requests and original FL-triggered release schedule. Mean and 95th-percentile request delay run from arrival to final broadcast. '+
        'S-OF retains its previously authorized soft TX cap with overruns recorded; analog methods keep hard caps. Corrected nonzero head-local broadcasts are used in the affected serial single-request conditions.', '',
        'The common target remains all 45 requests complete, retained mR >=95% of reference200, mean and worst-request KL <=0.01, '+
        'confirmed at three distinct consecutive evaluated rounds. See data/common_quality_costs.csv for observed confirmations. '+
        'This broad acceptance criterion is not a claim that methods have equal quality. Time curves hold their last measured value to the common display horizon without extra training.', '',
        '## Previous figures and reproducibility','',
        'The complete previous final_chart is preserved byte-for-byte in '+
        '[the archive](../final_chart_archive/20261028_before_adaptive/README.md), including all original radio ablations and the AirScape H-OVA 4 MHz infeasibility. '+
        'Those radio ablations belong to the previous objective and are not relabeled as the new Proposed. '+
        '[The previous Git revision](https://github.com/ckim48/ck_ova/tree/3f6316351d7247e1424053329be4e2f8bdc83b31/final_chart) preserves original relative paths.', '',
        'The frozen protocol, outcome summaries, feedback verification and source mapping are in '+
        '[results/adaptive_fu_v1_20261028](../results/adaptive_fu_v1_20261028). Numerical plot arrays and hashes are in data/. '+
        'Raw records, checkpoints, runner log and the verified raw archive remain local in that results directory and are not included in the GitHub publication. '+
        'The local raw archive omits shared feature caches/checkpoints already referenced by the original experiment; it is not a standalone dataset release.', '',
        'Run `.venv/bin/python tools/uav_convergence_v2/adaptive_fu.py run`, then '+
        '`.venv/bin/python tools/uav_convergence_v2/adaptive_final_charts.py build` and `... adaptive_final_charts.py publish`.','']
    (STAGE/'README.md').write_text('\n'.join(lines))
    cards=[]
    for item in figures:
        n=item['name'];title=html.escape(item['title'])
        cards.append(f'<article><h2>{title}</h2><a href="{n}.png"><img loading="lazy" src="{n}.png" alt="{title}"></a><p><a href="{n}.pdf">PDF</a> · <a href="{n}.svg">SVG</a></p></article>')
    (STAGE/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Adaptive FU final charts</title>'
        '<style>body{margin:0;background:#f3f5f8;color:#182338;font:16px system-ui}main{max-width:1450px;margin:auto;padding:30px}h2{font-size:17px}a{color:#315c96}'
        '.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,580px),1fr));gap:24px}article{background:white;padding:20px;border-radius:10px;border:1px solid #dfe4eb}img{width:100%}p{line-height:1.6}</style>'
        '<main><h1>Final charts — adaptive FU</h1><p>CapERA · AirScape | <a href="README.md">Definitions, results and limitations</a> · <a href="final_quality.csv">Quality CSV</a></p>'
        '<p>Lower retrain KL comes with a retained mR tradeoff. Every primary plot uses the new controller; the previous version is preserved in the archive.</p><div class="grid">'+''.join(cards)+'</div></main></html>\n')


def raw_archive():
    paths=set()
    for case,seed,method in study.tasks():
        root=OUT if seed==0 else OUT/f'sampling_seed_{seed}'
        directory=root/'cases'/case
        paths.update(p for p in directory.glob('*.json') if not p.is_symlink())
        for ds in DATASETS:
            dd=directory/(ds.lower()+'_seed0');paths.update([dd/'plan.json',dd/'requests.json'])
            paths.update(p for p in (dd/method).rglob('*') if p.is_file() and not p.is_symlink())
    entries={str(p.relative_to(ROOT)):dict(sha256=sha(p),size_bytes=p.stat().st_size) for p in sorted(paths)}
    archive=OUT/'raw_new_arms.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=6) as h:
        for p in entries:h.add(ROOT/p,arcname=p,recursive=False)
    checked=0
    with tarfile.open(archive,'r:gz') as h:
        for member in h:
            assert member.isfile() and member.name in entries
            assert not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
            data=h.extractfile(member).read();expected=entries[member.name]
            assert len(data)==expected['size_bytes'] and hashlib.sha256(data).hexdigest()==expected['sha256'];checked+=1
    assert checked==len(entries)
    write(OUT/'raw_archive_manifest.json',dict(file=archive.name,sha256=sha(archive),archive_bytes=archive.stat().st_size,verified_entries=checked,entries=entries))


def audit_feedback():
    audits=[]
    for case,seed,method in study.tasks():
        if method!=NEW:continue
        root=OUT if seed==0 else OUT/f'sampling_seed_{seed}'
        for ds in DATASETS:
            arm=root/'cases'/case/(ds.lower()+'_seed0')/method
            outcome=load(arm/'outcome.json')
            if outcome['status']!='completed':
                audits.append(dict(case=case,seed=seed,dataset=ds,status=outcome['status']));continue
            state={};steps=0;repairs=0;trials=0;extra_total=0
            for path in sorted((arm/'work_attempts').glob('*.json')):
                work=load(path);charged=sum(work['compute']['FU_control_scalar_ops'])
                actual=sum(r.get('extra_control_MAC_equivalents',0) for r in work['local'])
                assert actual==charged,(path,actual,charged)
                extra_total+=actual
                radio=work['radio']
                attempts=radio.get('adaptive_weight_trials',[])
                if attempts:
                    assert attempts[-1]['feasible'] and not any(r['feasible'] for r in attempts[:-1])
                    trials+=len(attempts)
                for r in work['local']:
                    rid=r['request'];previous=state.get(rid,dict(step=0,weight=.5,ema=0.,initial=r['request_initial_proxy_KL']))
                    assert r['step']==previous['step']+1
                    assert math.isclose(r['previous_RM_weight'],previous['weight'],abs_tol=1e-12)
                    assert r['request_initial_proxy_KL']==previous['initial']
                    dr=(r['RM']-r['retained_teacher_CE'])/max(r['retained_teacher_CE'],1e-8)
                    df=(max(r['forget_loss'],0.)-previous['initial'])/max(previous['initial'],1e-8)
                    desired,ema,signal=study.feedback(previous['weight'],previous['ema'],dr,df)
                    for a,b in [(dr,r['retain_relative_change']),(df,r['forget_relative_change']),
                                (desired,r['desired_RM_weight']),(ema,r['feedback_ema']),(signal,r['feedback_signal'])]:
                        assert math.isclose(a,b,abs_tol=1e-12)
                    assert attempts and math.isclose(attempts[-1]['weights'][rid],r['RM_weight'],abs_tol=1e-12)
                    state[rid]=dict(step=r['step'],weight=r['RM_weight'],ema=ema,initial=previous['initial'])
                    steps+=1;repairs+=r['radio_weight_repaired']
            assert steps==225 and len(state)==45 and all(r['step']==5 for r in state.values())
            audits.append(dict(case=case,seed=seed,dataset=ds,status='completed',verified_FU_steps=steps,
                feedback_recomputed_from_prior_request_state=True,extra_MAC_equivalents=extra_total,
                repaired_steps=repairs,radio_trials=trials))
    result=dict(passed=True,rows=audits)
    write(OUT/'feedback_audit.json',result);return result


def gradient_check():
    """Check the actual callback against the direct combined-loss derivative."""
    import torch
    import torch.nn.functional as F
    torch.manual_seed(61028)
    image=F.normalize(torch.randn(12,4,dtype=torch.double),dim=-1)
    text=F.normalize(torch.randn(24,4,dtype=torch.double),dim=-1)
    theta=torch.cat([.03*torch.randn(16,dtype=torch.double),torch.tensor([1.],dtype=torch.double)])
    proxy_theta=theta.clone();proxy_theta[:16]+=.01*torch.randn(16,dtype=torch.double)
    class Data:
        caption_ids={i:[2*i,2*i+1] for i in range(12)}
        def training_caption_ids(self,ids,step):return [2*i for i in ids]
    class Engine:
        data=Data();sample_owner={i:0 for i in range(12)};work_started=None
        adaptive_request_id='test';adaptive_pending={}
        def features(self,v,ids,cids):
            matrix=torch.eye(4,dtype=torch.double)+v[:16].reshape(4,4)
            return F.normalize(image[ids]@matrix,dim=-1),F.normalize(text[cids]@matrix,dim=-1),v[-1]
    engine=Engine()
    class Teacher:
        def __init__(self,v):self.v=v
        def features(self,ids,cids):
            with torch.no_grad():return engine.features(self.v,ids,cids)[:2]
    teacher=Teacher(theta);engine.adaptive_proxy=Teacher(proxy_theta)
    fids=[0,1,2];rids=[3,4,5,6]
    actual,meta=study.request_gradient(engine,theta,fids,rids,0,teacher)
    variable=theta.detach().requires_grad_();fcids=[c for i in fids for c in engine.data.caption_ids[i]]
    rcids=engine.data.training_caption_ids(rids,21)
    im,tx,scale=engine.features(variable,fids,fcids);ri,rt,_=engine.features(variable,rids,rcids)
    fk=study.retrieval_kl(im,tx,*engine.adaptive_proxy.features(fids,fcids),scale)
    ck=study.retrieval_kl(ri,rt,*teacher.features(rids,rcids),scale)
    logits=scale.exp()*(ri@rt.T);target=torch.arange(len(rids))
    rm=.5*(F.cross_entropy(logits,target)+F.cross_entropy(logits.T,target))
    direct=torch.autograd.grad(3*fk+.5*rm+3*ck,variable)[0];direct[-1]=0
    assert meta['RM_weight']==.5 and torch.allclose(actual,direct,rtol=1e-10,atol=1e-12)
    assert torch.isfinite(actual).all() and actual[-1]==0
    write(OUT/'gradient_check.json',dict(passed=True,direct_loss_gradient_matches_callback=True,
        first_step_weight=.5,frozen_temperature=True,training_only_synthetic_inputs=True))


def build():
    verification=read(OUT/'finished.json');assert verification['passed']
    study.register();assert len(verification['rows'])==22
    STAGE.mkdir(exist_ok=True);(STAGE/'data').mkdir(exist_ok=True);(STAGE/'supplementary').mkdir(exist_ok=True)
    configure();gradient_check();audit_feedback();mapping=build_view();convergence();ledgers=bandwidth()
    requests=f.request_curves();costs=f.quality_costs();quality=f.quality_table();controller=supplemental()
    metrics,contrasts,summary=paired_summary()
    documentation(ledgers,contrasts,summary,controller)
    write(STAGE/'data/plot_series.json',f.SERIES)
    # Canonicalize paths produced by inherited readers through the symlink view.
    sources={str((ROOT/p).resolve().relative_to(ROOT)):digest for p,digest in f.SOURCES.items()}
    # Our staged ledger becomes final_chart/data/... after publication.
    sources={('final_chart/'+str(Path(p).relative_to(STAGE.relative_to(ROOT)))) if (ROOT/p).is_relative_to(STAGE) else p:d for p,d in sources.items()}
    write(STAGE/'data/manifest.json',dict(figures=f.FIGURES,methods=METHODS,datasets=DATASETS,source_sha256=sources,
        actual_source_mapping=mapping,experiment_outcomes=verification['outcomes'],primary_sampling_seed=0,
        paired_sampling_seeds=[0,1,2],controller=study.CONTROLLER,
        previous_snapshot='final_chart_archive/20261028_before_adaptive',
        quality_definition='Existing 95%-retrain mR /0.01 mean and worst KL /all45 requests /3 distinct evaluated rounds',
        energy_scope='Active FL/FU communication + computing + hovering; modeled scenario',
        regenerated_all_primary_figures=True,limitations='Empirical KL-versus-utility tradeoff; shared known-cohort proxy; conditional seeds; no universal dominance or deletion guarantee.'))
    for name,digest in sources.items():
        p=STAGE/Path(name).relative_to('final_chart') if name.startswith('final_chart/') else ROOT/name
        assert sha(p)==digest,name
    for figure in f.FIGURES:
        for item in figure['files']:assert sha(STAGE/item['file'])==item['sha256']
    raw_archive()
    write(OUT/'publication_ready.json',dict(passed=True,figures=len(f.FIGURES),plot_series=len(f.SERIES),request_conditions=len(requests),
        quality_confirmations=sum(r['status']=='met' for r in costs),controller=controller,paired_summary=summary,
        files_sha256={str(p.relative_to(STAGE)):sha(p) for p in sorted(STAGE.rglob('*')) if p.is_file()}))
    print(dict(passed=True,figures=len(f.FIGURES),contrasts=contrasts),flush=True)


def publish():
    ready=read(OUT/'publication_ready.json');assert ready['passed']
    protocol=read(OUT/'protocol.json');target=ROOT/'final_chart'
    if not ARCHIVE.exists():
        current={str(p.relative_to(target)):sha(p) for p in target.rglob('*') if p.is_file()}
        assert current==protocol['previous_final_chart_sha256'],'Previous bundle changed since registration'
        for p,d in ready['files_sha256'].items():assert sha(STAGE/p)==d
        ARCHIVE.parent.mkdir(exist_ok=True);target.rename(ARCHIVE);STAGE.rename(target)
    assert {str(p.relative_to(ARCHIVE)):sha(p) for p in ARCHIVE.rglob('*') if p.is_file()}==protocol['previous_final_chart_sha256']
    for p,d in ready['files_sha256'].items():assert sha(target/p)==d,p
    manifest=read(target/'data/manifest.json')
    for p,d in manifest['source_sha256'].items():assert sha(ROOT/p)==d,p
    write(OUT/'publication.json',dict(passed=True,unix=time.time(),folder='final_chart',archive=str(ARCHIVE.relative_to(ROOT)),
        figures=ready['figures'],new_runs=22,all_previous_files_preserved=True,all_sources_verified=True))
    print(read(OUT/'publication.json'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['build','publish']);args=parser.parse_args()
    build() if args.action=='build' else publish()
