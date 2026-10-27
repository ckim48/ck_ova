"""1x2 figures from the paper-core transfer benchmark; no smoothing or rank filtering."""
import argparse
import copy
import csv
import html
import importlib.util
import math
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
import paper_uav_benchmarks as study
import proposed_resource_sweep as proposed
import proposed_resource_timing as timing
import paper_uav_drain_fix as correction
import time_energy_report as chart
from adaptive_fu import ROOT, read, write, sha

DEST=ROOT/'final_final_charts'
METHODS={study.lib.PROPOSED:'Proposed',**study.METHODS}
DATASETS=study.DATASETS
SOURCES={};SERIES=[];FIGURES=[]
spec=importlib.util.spec_from_file_location('paper_chart_physics',ROOT/'tools/uav_real_vlm_v1/energy.py')
physical=importlib.util.module_from_spec(spec);spec.loader.exec_module(physical)


def load(p):
    SOURCES[str(p.relative_to(ROOT))]=sha(p);return read(p)


def source(case,ds,method):
    origin=study.OUT
    if method==study.lib.PROPOSED:origin=timing.OUT if case.startswith(('downlink_','frequency_')) else proposed.OUT
    if method==correction.METHOD and (correction.OUT/'protocol.json').exists():
        affected=read(correction.OUT/'protocol.json')['affected']
        if any(r['case']==case and r['dataset']==ds for r in affected):origin=correction.OUT
    return origin/'cases'/case/(ds.lower()+'_seed0')/method


def comm(radio,p):
    energy=radio['energy_bound_j']/p['pa_efficiency']+radio['uplink_s']*(9*p['tx_circuit_w']+p['rx_circuit_w'])
    down=0.
    if radio['uplink_s']>0 or radio.get('broadcast_required',False):
        down=(p['update_coordinates']*p['broadcast_bits_per_coordinate']+p['control_bits_per_update'])/p['downlink_rate_bps']
        energy+=down*(p['downlink_power_w']/p['pa_efficiency']+p['tx_circuit_w']+9*p['rx_circuit_w'])
    return energy,down


def ledger(case,ds,method):
    arm=source(case,ds,method);o=load(arm/'outcome.json');assert o['status']=='completed',o
    settings=load(arm.parents[1]/'settings.json');physical.PARAMETERS=settings['energy_model'];p=physical.PARAMETERS
    timeline=load(arm/'mission_timeline.json');checks=load(arm/'verification.json');assert checks['passed']
    tx=[load(f) for f in sorted((arm/'transactions').glob('*.json'))]
    works=[load(f) for f in sorted((arm/'work_attempts').glob('*.json'))]
    active=sum(s['service_duration_s'] for s in timeline['sites']);compute=communication=0.
    for w in works:
        assert w['status']=='transmitted';compute+=float(physical.compute_cost(w['compute']['device_macs'])[0].sum())
    for t in tx:
        c,d=comm(t['radio'],p['communication']);communication+=c
        assert math.isclose(d,t['timing']['downlink_s'],abs_tol=1e-9)
        compute+=p['communication']['head_processing_w']*t['radio']['uplink_s']
    assert math.isclose(active,sum(t['timing']['duration_s'] for t in tx)+sum(w['duration_s'] for w in timeline['waits']),abs_tol=1e-8)
    hover=10*float(physical.propulsion_power(0))*active;final=load(arm/'service_completion.json')['evaluation']
    mia=load(arm/'mia.json')
    req=load(arm/'request_records.json');lat=[r['completed_s']-r['arrival_s'] for r in req]
    record=dict(case=case,dataset=ds,method=method,label=METHODS[method],status=o['status'],source=str(arm.relative_to(ROOT)),
        total_energy_j=compute+communication+hover,compute_j=compute,communication_j=communication,hover_j=hover,delay_s=active,
        retained_mR_percent=100*final['retained']['mR'],retained_loss=final['retained']['loss'],forget_KL=final['forget']['KL_retrain_to_candidate'],
        worst_request_KL=final['worst_request_KL'],mean_FU_latency_s=float(np.mean(lat)),p95_FU_latency_s=float(np.percentile(lat,95)),
        network_cap_overruns=sum(t['radio']['energy_bound_j']>.1*(1+2e-6) for t in tx),
        deadline_overruns=sum(t['radio']['uplink_s']>settings['config']['round_uplink_deadline_s']*(1+2e-6) for t in tx),
        energy_budget_overruns=sum(t['radio']['max_user_energy_j']>settings['config']['available_energy_j']*(1+2e-6) for t in tx),
        max_staleness=checks.get('max_staleness'),MIA_auc_gap_vs_retrain=mia['auc_gap_vs_retrain'],transactions=len(tx),FU_cycles=sum(r['steps'] for r in req))
    return record


def write_csv(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,keys,lineterminator='\n');w.writeheader();w.writerows(rows)


def configure():
    chart.configure()
    chart.COLORS=chart.style.COLORS=['#4C72B0','#EFAA54','#66B366']
    chart.MARKERS=chart.style.MARKERS=['o','s','v']
    chart.STYLES=chart.style.STYLES=['-','--','-.']


def save(fig,axes,name,title):
    fig.legend(*axes[0].get_legend_handles_labels(),ncol=3,loc='upper center',bbox_to_anchor=(.5,1.),fontsize=12,handlelength=1.6,columnspacing=1.)
    settings=[]
    for i,(ax,ds) in enumerate(zip(axes,DATASETS)):
        pos=ax.get_position();fig.text((pos.x0+pos.x1)/2,.015,f'({chr(97+i)}) {ds}',ha='center',va='bottom',fontsize=14)
        assert not ax.texts
        d=dict(dataset=ds)
        for dim in ['x','y']:
            ticks=getattr(ax,'get_'+dim+'ticks')();limits=getattr(ax,'get_'+dim+'lim')()
            assert math.isclose(ticks[0],limits[0],abs_tol=1e-9) and math.isclose(ticks[-1],limits[-1],abs_tol=1e-9)
            assert np.allclose(np.diff(ticks),ticks[1]-ticks[0])
            d[dim]=dict(ticks=ticks.tolist(),limits=list(limits),labels=[getattr(ax,dim+'axis').get_major_formatter()(v) for v in ticks])
        settings.append(d)
    files=[]
    for ext in ['png','pdf','svg']:
        p=DEST/(name+'.'+ext);p.parent.mkdir(parents=True,exist_ok=True);fig.savefig(p,dpi=350,bbox_inches='tight');files.append(dict(file=str(p.relative_to(DEST)),sha256=sha(p)))
    chart.plt.close(fig);FIGURES.append(dict(name=name,title=title,axes=settings,files=files))


def sweep(rows):
    by={(r['case'],r['dataset'],r['method']):r for r in rows}
    xticks=dict(bandwidth=[1,2,3,4,5],downlink=[.25,1.,1.75,2.5,3.25,4.],frequency=[.75,1.5,2.25,3.],fu_steps=[1,3,5,7,9],batch=[8,32,56,80,104,128])
    for ai,(axis,a) in enumerate(study.AXES.items()):
        for mi,(metric,ylabel,scale) in enumerate([('total_energy_j','Total energy (kJ)',.001),('delay_s','Delay (s)',1.)]):
            name=f'{1+ai*2+mi:02d}_{axis}_{"energy" if mi==0 else "delay"}_1x2'
            fig,axes=chart.style.canvas(ylabel);panels=[]
            for ax,ds in zip(axes,DATASETS):
                values=[]
                for n,(method,label) in enumerate(METHODS.items()):
                    x=a['values'];y=[by[(study.lib.case_id(axis,v),ds,method)][metric]*scale for v in x]
                    chart.style.curve(ax,x,y,n,label);values+=y
                    SERIES.append(dict(figure=name,dataset=ds,method=method,x=x,y=y,metric=metric))
                chart.regular_axis(ax,'x',xticks[axis],chart.tick_decimals(xticks[axis]));ax.set_xlabel(a['label']);panels.append(values)
            chart.finish_axes(axes,panels);save(fig,axes,name,ylabel+' versus '+a['label'])


def trajectories(ds,method):
    arm=source('baseline',ds,method);timeline=load(arm/'mission_timeline.json')
    prefix=load(arm.parent/'initial20/rounds.json')
    start=load(ROOT/'results/uav_convergence_v2_20261027'/(ds.lower()+'_pilot/rounds.json'))[0]['evaluation']['retained']
    warm=[(0,start)]+[(r['round'],r['evaluation']['retained']) for r in prefix if r.get('evaluation')]
    points=[load(f) for f in sorted((arm/'evaluations').glob('*.json'))];last=load(arm/'service_completion.json')
    if points[-1]['sequence']!=last['sequence'] or points[-1]['mission_clock_s']!=last['mission_clock_s']:points.append(last)
    return warm,points,timeline


def convergence():
    paths={(ds,m):trajectories(ds,m) for ds in DATASETS for m in METHODS}
    horizon=math.ceil(max(chart.active_time(tl,p[-1]['mission_clock_s']) for w,p,tl in paths.values())/100)*100
    for i,(metric,ylabel,scale) in enumerate([('mR','Retained mR (%)',100.),('loss','Validation loss',1.),('KL','Forget-set KL',1.)]):
        for j,domain in enumerate(['round','time']):
            name=f'{11+2*i+j:02d}_{domain}_{metric.lower()}_1x2';fig,axes=chart.style.canvas(ylabel);panels=[]
            for ax,ds in zip(axes,DATASETS):
                ys=[]
                for n,(m,label) in enumerate(METHODS.items()):
                    warm,p,tl=paths[(ds,m)]
                    y=[r['evaluation']['forget']['KL_retrain_to_candidate'] if metric=='KL' else scale*r['evaluation']['retained'][metric] for r in p]
                    x=[r['completed_FL_rounds'] if domain=='round' else chart.active_time(tl,r['mission_clock_s']) for r in p]
                    if metric!='KL':
                        before=warm if domain=='round' else [(0,warm[-1][1])]
                        x=[v[0] for v in before]+x;y=[scale*v[1][metric] for v in before]+y
                    actual_x,actual_y=list(x),list(y)
                    if domain=='time' and x[-1]<horizon:x.append(horizon);y.append(y[-1])
                    chart.style.curve(ax,x,y,n,label);ys+=y
                    SERIES.append(dict(figure=name,dataset=ds,method=m,x=actual_x,y=actual_y,display_x=x,display_y=y,metric=metric,
                        terminal_hold_only=domain=='time',no_smoothing=True))
                ticks=[0,50,100,150,200] if domain=='round' else list(np.linspace(0,horizon,5))
                chart.regular_axis(ax,'x',ticks,0);ax.set_xlabel('FL round' if domain=='round' else 'Time (s)');panels.append(ys)
            chart.finish_axes(axes,panels,decimals=2 if metric=='loss' else 1 if metric=='mR' else None)
            save(fig,axes,name,ylabel+' versus '+domain)


def docs(rows,complete):
    base=[r for r in rows if r['case']=='baseline'];improvements=[]
    for ds in DATASETS:
        our=next(r for r in base if r['dataset']==ds and r['method']==study.lib.PROPOSED)
        for r in base:
            if r['dataset']==ds and r['method']!=study.lib.PROPOSED:
                improvements.append(dict(dataset=ds,versus=r['label'],energy_reduction_percent=100*(1-our['total_energy_j']/r['total_energy_j']),
                    delay_reduction_percent=100*(1-our['delay_s']/r['delay_s']),mR_difference_pp=our['retained_mR_percent']-r['retained_mR_percent'],
                    KL_reduction_percent=100*(1-our['forget_KL']/r['forget_KL'])))
    write(DEST/'data/baseline_comparison.json',improvements)
    text=['# Paper-core UAV/VLM comparison','',
      'CapERA (left), AirScape (right). Main curves: Proposed, ISCC-P2, AFL-OTA. These are explicitly scoped transfers into the common10-UAV scenario, not complete reproductions of the cited papers.', '',
      '| Method | Published source | Preserved core | Explicit adaptation / omitted part |', '|---|---|---|---|',
      '| ISCC-P2 | [Tang et al., IEEE TWC24(4),2025](https://doi.org/10.1109/TWC.2024.3523381); [official code](https://github.com/TheaSherlock/ISCC-UAV) | Equal-client FedSGD, FP32 FDMA and exact fixed-position/batch P2 bandwidth allocation | Cached VLM data; sensing success1; fixed nominal UAV head; no BBPO batch/position/round optimization |',
      '| AFL-OTA | [Zhai et al., IEEE TWC24(3),2025](https://doi.org/10.1109/TWC.2024.3520501); [author manuscript](https://arxiv.org/abs/2403.06653) | Alg.1 cached gradients, ready-device updates, normalized OTA aggregation and overlapping local computation | UAV clients plus fixed nominal head; all-ready selection, channel inversion/repetition; no joint Alg.2 optimizer |',
      '| Proposed | Current repository formulation | Joint FL/FU OVA, optimized head/resources and adaptive retrieval-FU controller | Planned-cohort clean proxy20 available; the external baselines do not use this teacher |', '',
      'The two cited papers are FL-only. Both receive the same **GA→FT FU extension**: cosine forgetting then retained contrastive SGD, based on the [SLUG ICML2025 official comparison code](https://github.com/CSIPlab/SLUG/blob/main/src/clip/unlearn/GAFT.py). This is a federated adapter adaptation, not an original UAV-FU method from either paper. Learning objectives and radio protocols both differ; the main comparison measures complete adapted systems. It does not isolate a communication effect.', '',
      'The previous FM-CM/S-AC/AC-OF/S-OF curves remain available in [the ablation bundle](ablation/index.html), with their original names and data. They are not relabeled as literature methods.', '',
      '**Scope and limitations.** Shared frozen RemoteCLIP B32 mean8 features, rank10 adapters, warm20, same partitions and45 requests. The final round is200 (20 common warmup +180 mission FL updates). FU-cycle sweeps apply the declared cycle count to every method. Single development seed and subset diagnostics, not official dataset scores or statistical superiority. Cached features mean active compute cost is adapter work, not re-encoding/full-VLM training. Methods use different FU training signals, and Proposed has an additional offline known-cohort proxy assumption. Equal-device versus sample-weighted FL aggregation follows the respective method.', '',
      'Energy is active FL/FU communication + executed computation + hovering. Delay is active service time; travel, collection and preflight are excluded from these figures. All-peer circuit costs are retained even when only selected receivers consume a broadcast. OTA mean/variance metadata is modeled as error-free with zero overhead, following the source convention. Resource-optimizer wall time is not calibrated to UAV energy.', '',
      'The homogeneous cached-adapter scenario may make all devices ready together: an asynchronous algorithm can then operate synchronously. We do not manufacture staleness or drop clients to separate curves. Actual maximum staleness is in the data table. CPU/batch effects may be small because communication/hover dominates; plotted slopes are not amplified or invented.', '',
      'Every point is a fresh simulation or the identical pre-existing Proposed arm; no time/energy rescaling of another learning trajectory. Power bounds stay hard. Paper transfers complete their payloads and record network-energy/update-energy/deadline QoS exceedances. `data/metrics.csv` reports these; a completed QoS-violating point must not be described as feasible under all original Proposed constraints. The full digital upload is32bit, not the old Q6 ablation.', '',
      'Both axes include endpoint ticks at uniform numerical spacing. Panel precision matches. No point values, extra grey captions, smoothing, or fabricated confidence intervals. Time plots hold the terminal model after completion; the appended segment is not additional training. Repeated FL rounds show intervening FU evaluations.', '',
      '| Dataset | Method | mR (%) | Forget KL | Energy (kJ) | Active delay (s) |', '|---|---|---:|---:|---:|---:|']
    for r in base:text.append(f"| {r['dataset']} | {r['label']} | {r['retained_mR_percent']:.3f} | {r['forget_KL']:.6g} | {r['total_energy_j']/1000:.3f} | {r['delay_s']:.3f} |")
    text+=['','Status: '+('complete' if complete else 'baseline comparison complete; remaining sensitivity runs pending')+'.','',
      'Five asynchronous large-batch arms exposed a terminal timing assertion: already-started local jobs outlasted the final zero-LR update by0.128–0.24576ms. Their failed records remain local and unchanged. The source-frozen `paper_uav_drain_fix.py` reruns include this final compute wait/hover cost. Every transaction, job and work ledger is byte-identical to the original attempt; final learned metrics are identical. See `data/terminal_drain_correction.json`.','',
      'Reproduce the complete workflow: `python tools/uav_convergence_v2/paper_uav_release.py run`. It completes the preserved original attempts, accepts only the documented terminal-drain rejection, runs the correction, audits and exports. Rebuild/audit existing results with `paper_uav_release.py finish`. Requires the existing local datasets/caches. Raw samples, checkpoints, transaction logs and local archives are not published. Source hashes and plot arrays are included.','']
    (DEST/'README.md').write_text('\n'.join(text))
    cards=[]
    for f in FIGURES:cards.append('<section><h2>'+html.escape(f['title'])+'</h2><a href="'+f['name']+'.pdf"><img loading="lazy" src="'+f['name']+'.png"></a></section>')
    (DEST/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>UAV VLM comparisons</title><style>body{font:16px system-ui;max-width:1150px;margin:35px auto;padding:0 20px;color:#202530}img{width:100%;height:auto}h2{font-size:20px}section{margin:45px 0}</style><h1>UAV/VLM paper-core comparisons</h1><p>CapERA · AirScape. <a href="README.md">Method transfers and limitations</a> · <a href="data/metrics.csv">Metrics CSV</a> · <a href="ablation/index.html">Ablations</a></p>'+''.join(cards))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--baseline-only',action='store_true');a=parser.parse_args()
    study.register();proposed.register();timing.register()
    DEST.mkdir(exist_ok=True);(DEST/'data').mkdir(exist_ok=True)
    for origin in [study.OUT,proposed.OUT,timing.OUT]:load(origin/'protocol.json')
    if not a.baseline_only:
        correction.register()
        load(correction.OUT/'protocol.json')
        fixed=load(correction.OUT/'verification.json');assert fixed['passed']
        write(DEST/'data/terminal_drain_correction.json',fixed)
    cases=[dict(id='baseline')] if a.baseline_only else study.cases()
    rows=[ledger(c['id'],ds,m) for c in cases for ds in DATASETS for m in METHODS]
    if not a.baseline_only:
        for origin in [study.OUT,proposed.OUT,timing.OUT]:assert load(origin/'verification.json')['passed']
        audit=load(study.OUT/'comparison_audit.json');assert audit['passed'] and audit['complete']
        write(DEST/'data/verification.json',{k:v for k,v in audit.items() if k!='input_sha256'})
    protocol=load(study.OUT/'protocol.json')
    write(DEST/'data/method_transfers.json',{k:v for k,v in protocol.items() if k!='source_sha256'})
    for execution in ['execution_batching.json','execution_parallel.json']:
        if (study.OUT/execution).exists():load(study.OUT/execution)
    configure()
    if not a.baseline_only:sweep(rows)
    convergence();docs(rows,not a.baseline_only)
    # Preserve only the already-published ablation bundle; never raw experiment archives.
    old=ROOT/'final_chart';allowed={'.png','.pdf','.svg','.csv','.json','.md','.html'}
    published=subprocess.check_output(['git','--git-dir=.git-push-ready/code-charts.git','--work-tree=.','ls-files','-z','final_chart'],cwd=ROOT).decode().split('\0')
    for item in published:
        if not item:continue
        p=ROOT/item
        if p.is_file() and p.suffix in allowed and not p.name.startswith('raw_'):
            dest=DEST/'ablation'/p.relative_to(old);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    write(DEST/'data/metrics.json',rows);write_csv(DEST/'data/metrics.csv',rows);write(DEST/'data/plot_series.json',SERIES)
    for p in [Path(__file__),Path(study.__file__),Path(study.radio.__file__),Path(study.lib.__file__),Path(proposed.__file__),Path(timing.__file__),Path(chart.__file__),Path(chart.style.__file__),Path(physical.__file__)]:SOURCES[str(p.relative_to(ROOT))]=sha(p)
    write(DEST/'data/manifest.json',dict(unix=time.time(),complete=not a.baseline_only,methods=METHODS,datasets=DATASETS,figures=FIGURES,
        compared_arms=len(rows),source_sha256=SOURCES,sensitivity_axes=study.AXES,raw_records_published=False))
    print(dict(figures=len(FIGURES),arms=len(rows),destination=str(DEST)),flush=True)


if __name__=='__main__':main()
