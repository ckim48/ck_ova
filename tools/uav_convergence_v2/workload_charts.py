"""Five workload/link sensitivities; actual arm ledgers and quality tradeoffs."""
import csv
import html
import math
from pathlib import Path
import time
import numpy as np
import workload_sweeps_v3 as study
import paper_uav_charts as prior
import workload_departure_fix as correction
import workload_correction_audit as correction_audit

ROOT=study.ROOT
DEST=ROOT/'final_final_charts/expanded_sweeps'
OLD_SOURCE=prior.source
AXES={a:dict(prior.study.AXES[a]) for a in ['bandwidth','downlink']}
AXES.update(study.AXES)
TICKS=dict(bandwidth=[1,2,3,4,5],downlink=[.25,1.,1.75,2.5,3.25,4.],
    request_bursts=[1,3,5,7,9,11,13,15],fl_rounds=[100,125,150,175,200],packet_loss=[0,10,20,30,40])


def cid(axis,value):
    return 'baseline' if value==AXES[axis]['base'] else axis+'_'+str(value).replace('.','p')


def cases():
    return ['baseline']+[cid(a,v) for a,s in AXES.items() for v in s['values'] if v!=s['base']]


def source(case,ds,m):
    if (case,ds,m)==(correction.CASE,correction.DATASET,correction.METHOD):
        return correction.OUT/'cases'/case/(ds.lower()+'_seed0')/m
    if case.startswith(('request_bursts_','fl_rounds_','packet_loss_')):
        return study.OUT/'cases'/case/(ds.lower()+'_seed0')/m
    return OLD_SOURCE(case,ds,m)


def comm(r,p):
    e=r['energy_bound_j']/p['pa_efficiency']+r['uplink_s']*(9*p['tx_circuit_w']+p['rx_circuit_w'])
    count=r.get('broadcast_attempts',int(r['uplink_s']>0 or r.get('broadcast_required',False)))
    d=count*(p['update_coordinates']*p['broadcast_bits_per_coordinate']+p['control_bits_per_update'])/p['downlink_rate_bps']
    return e+d*(p['downlink_power_w']/p['pa_efficiency']+p['tx_circuit_w']+9*p['rx_circuit_w']),d


def ledger(case,ds,m):
    row=prior.ledger(case,ds,m);arm=source(case,ds,m)
    tl=prior.load(arm/'mission_timeline.json');manifest=prior.load(arm/'manifest.json')
    tx=[prior.load(p) for p in sorted((arm/'transactions').glob('*.json'))]
    row.update(FL_rounds=manifest['completed_FL_rounds'],mission_delay_s=tl['mission_end_s'],
        completion_delay_s=tl['service_end_s'],
        broadcasts=sum(int(t['radio'].get('broadcast_attempts',int(t['radio']['uplink_s']>0 or t['radio'].get('broadcast_required',False)))>0) for t in tx),
        broadcast_attempts=sum(t['radio'].get('broadcast_attempts',int(t['radio']['uplink_s']>0 or t['radio'].get('broadcast_required',False))) for t in tx))
    row['broadcast_retries']=row['broadcast_attempts']-row['broadcasts']
    assert row['FU_cycles']==225
    return row


def draw(rows,axis,metric,ylabel,scale,name):
    chart=prior.chart;spec=AXES[axis];by={(r['case'],r['dataset'],r['method']):r for r in rows}
    fig,axes=chart.style.canvas(ylabel);panels=[]
    for ax,ds in zip(axes,study.DATASETS):
        values=[]
        for n,(m,label) in enumerate(study.METHODS.items()):
            x=spec.get('display_values',spec['values']);ids=[cid(axis,v) for v in spec['values']]
            y=[by[(c,ds,m)][metric]*scale for c in ids]
            chart.style.curve(ax,x,y,n,label);values+=y
            prior.SERIES.append(dict(figure=name,dataset=ds,method=m,x=x,y=y,metric=metric,scale=scale,cases=ids,axis=axis))
        ticks=TICKS[axis];chart.regular_axis(ax,'x',ticks,chart.tick_decimals(ticks));ax.set_xlabel(spec['label']);panels.append(values)
    chart.finish_axes(axes,panels,decimals=1 if metric=='retained_mR_percent' else None)
    prior.save(fig,axes,name,ylabel+' versus '+spec['label'])


def response_summary(rows):
    by={(r['case'],r['dataset'],r['method']):r for r in rows};out=[]
    for axis,spec in AXES.items():
        for ds in study.DATASETS:
            for m,label in study.METHODS.items():
                for metric in ['total_energy_j','delay_s','mission_delay_s']:
                    y=[by[(cid(axis,v),ds,m)][metric] for v in spec['values']];diff=np.diff(y)
                    direction='increasing' if np.all(diff>=-1e-9) else 'decreasing' if np.all(diff<=1e-9) else 'nonmonotonic'
                    out.append(dict(axis=axis,dataset=ds,method=m,label=label,metric=metric,
                        x=spec.get('display_values',spec['values']),values=y,first=y[0],last=y[-1],
                        endpoint_change_percent=100*(y[-1]/y[0]-1),span_percent=100*(max(y)/min(y)-1),direction=direction))
    return out


def docs(rows,responses):
    text=['# Expanded UAV/VLM sensitivity figures','',
      'CapERA (left), AirScape (right). Proposed, ISCC-P2 and AFL-OTA; the [paper-core transfer definitions and limitations](../README.md) remain unchanged. All three use the same intervention at each plotted point. These are scoped adaptations, not complete reproductions of the cited UAV papers.','',
      '## Figures','',
      '- **01–10:** five axes × total energy / active FL/FU delay.',
      '- **11–16:** retained mR, validation loss and forget KL versus round and active time, at the unchanged baseline.',
      '- **17–18:** final mR and KL versus FL-round budget; fewer rounds change utility and forgetting, not just cost.',
      '- **19–23:** mission delay for the five axes, kept separate from active FL/FU delay.',
      '- Earlier CPU-frequency, per-request FU-cycle and batch-cap results remain [in the parent bundle](../index.html), including nearly flat curves.','',
      '## Controlled interventions','',
      '| Axis | Values | Fixed / interpretation |','|---|---|---|',
      '| Uplink bandwidth | 1,2,3,4,5 MHz | Existing audited fresh simulations; unchanged channel/noise scaling and all full payloads. |',
      '| Downlink rate | 0.25,0.5,1,2,4 Mbit/s | Existing audited fresh simulations; all model broadcasts included. |',
      '| FU requests per release | 1,3,5,9,15 | Same45 requests and225 FU cycles, regrouped into45,15,9,5,3 releases across the five sites. Release grouping also changes which site serves a request, following the registered existing scenario. |',
      '| FL-round budget | 100,125,150,175,200 |20 common warmup +80/105/130/155/180 mission FL updates, equally divided over five sites. Same45 requests and225 FU cycles. Original cosine schedule is truncated, not retuned. Within-site release ordinals are compressed proportionally. |',
      '| Downlink packet loss | 0,10,20,30,40% | Independent exogenous erasures of whole multicast model+control packets. Actual attempts until success are recorded and charged. |','',
      '**ARQ assumptions:** one trial represents delivery to all intended peers; failed trials repeat the whole broadcast and consume real modeled airtime, communication energy and hover time. Shared deterministic uniform draws are coupled across loss settings and methods by transaction ordinal. Feedback/ACK has zero duration and energy in this simplified model. Uplink loss/CSI assumptions are unchanged. Packet loss is an imposed scenario parameter, not measured or SNR-calibrated reliability. Retries do not create additional gradient updates.','',
      '**Costs and clocks:** Total energy = active FL/FU communication + executed adapter computation + hovering. Main delay = sum of active service intervals. Mission delay includes travel, IoT collection, all service and return, with no preflight feature extraction. Feature caches are fixed; these are not full-VLM encoding costs. Energy, active delay and mission delay are independently recomputed from recorded work/transaction/timeline ledgers. Active hovering dominates this cached-adapter scenario, so energy and delay profiles are closely related; component energies remain available in the metrics CSV.','',
      '**Quality and reference:** all cases use the same data partitions,45-request forget union, warm20 and reporting reference200. Shorter-round cases are intentionally compared against that fixed reference; their KL must not be presented as equivalence to a newly matched shorter retraining oracle. The reference is reporting-only. Proposed retains its previously disclosed offline known-cohort proxy and adaptive controller; paper baselines retain their shared GA→FT FU extension.','',
      '**Integrity:** six zero-change control arms reproduce baseline final metrics exactly and preserve transaction order, times and RF energy. The expansion contains64 new intervention arms plus those6 controls and8 identical reused Proposed request-grouping arms. The combined five-axis bundle compares126 unique arms. Single development seed; no confidence intervals or statistical superiority claim. No curve smoothing, rank filtering, artificial separation or added slopes. QoS-cap exceedances remain in metrics; completion does not imply every original Proposed bound was met.','',
      'Both axes have equally spaced numeric endpoint ticks, matching precision across panels. No point labels or extra grey captions. Convergence plots retain all observed FU effects; time curves hold the last completed model to the common display horizon.','',
      'The two earlier launcher attempts stopped during dynamic function construction before any training arm started; the v3 launcher fixes source tracking for freshly loaded modules. They produced no selected or discarded model results. The100-round AirScape Proposed run later exposed a real regional-departure assertion: its late request had only4 of5 FU cycles when the third site FL quota was reached. A preserved correction performs the remaining FU-only transaction before departure, charges its full costs, and adds no FL updates. Its first48 transactions are identical to the failed attempt except for measured host optimizer wall time, which is excluded from modeled UAV energy/delay; all other completed arms already satisfy regional completion. There are71 new training attempts including this preserved failure,70 completed new arms and8 reused arms. See `data/regional_completion_correction.json`. Source-frozen experiment code, control checks, input hashes, full plot arrays and aggregate metrics are included; raw data/checkpoints/transaction traces remain local.','',
      '## Measured endpoint changes','',
      'Percent change is from the first to last x value. Nonmonotonic curves retain every intermediate point.','',
      '| Axis | Dataset | Method | Energy change | Active-delay change | Shape (energy) |','|---|---|---|---:|---:|---|']
    by={(r['axis'],r['dataset'],r['method'],r['metric']):r for r in responses}
    for a in AXES:
        for ds in study.DATASETS:
            for m,label in study.METHODS.items():
                e=by[(a,ds,m,'total_energy_j')];d=by[(a,ds,m,'delay_s')]
                text.append(f"| {a} | {ds} | {label} | {e['endpoint_change_percent']:+.2f}% | {d['endpoint_change_percent']:+.2f}% | {e['direction']} |")
    text+=['','Rebuild after local inputs are available: `python tools/uav_convergence_v2/workload_release.py run`. Audit/export existing completed results: `python tools/uav_convergence_v2/workload_release.py finish`. The existing parent campaign supplies the reused bandwidth/downlink and baseline arms.','']
    (DEST/'README.md').write_text('\n'.join(text))
    cards=['<section><h2>'+html.escape(f['title'])+'</h2><a href="'+f['name']+'.pdf"><img loading="lazy" src="'+f['name']+'.png"></a></section>' for f in prior.FIGURES]
    (DEST/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Expanded UAV VLM sweeps</title><style>body{font:16px system-ui;max-width:1150px;margin:35px auto;padding:0 20px;color:#202530}img{width:100%;height:auto}h2{font-size:20px}section{margin:45px 0}</style><h1>Expanded UAV/VLM sweeps</h1><p>CapERA · AirScape · <a href="README.md">Scenario definitions and results</a> · <a href="data/metrics.csv">Metrics CSV</a> · <a href="../index.html">Earlier figures</a></p>'+''.join(cards))


def main():
    study.register();check=study.read(study.OUT/'verification.json');assert check['passed'] and check['arms']==78
    audit=study.read(study.OUT/'comparison_audit.json');assert audit['passed'] and audit['compared_arms']==126
    DEST.mkdir(parents=True,exist_ok=True);(DEST/'data').mkdir(exist_ok=True)
    prior.DEST=DEST;prior.source=source;prior.comm=comm;prior.SOURCES={};prior.SERIES=[];prior.FIGURES=[]
    for name in ['protocol.json','execution.json','control_verification.json','verification.json','unit_checks.json','comparison_audit.json']:
        p=study.OUT/name;value=prior.load(p);study.write(DEST/'data'/name,value)
    rows=[ledger(c,ds,m) for c in cases() for ds in study.DATASETS for m in study.METHODS]
    assert len(rows)==126
    study.write(DEST/'data/regional_completion_correction.json',correction_audit.verify())
    prior.configure()
    for i,a in enumerate(AXES):
        draw(rows,a,'total_energy_j','Total energy (kJ)',.001,f'{1+2*i:02d}_{a}_energy_1x2')
        draw(rows,a,'delay_s','Delay (s)',1.,f'{2+2*i:02d}_{a}_delay_1x2')
    prior.convergence()
    draw(rows,'fl_rounds','retained_mR_percent','Retained mR (%)',1.,'17_fl_rounds_mr_1x2')
    draw(rows,'fl_rounds','forget_KL','Forget-set KL',1.,'18_fl_rounds_kl_1x2')
    for i,a in enumerate(AXES):draw(rows,a,'mission_delay_s','Mission delay (s)',1.,f'{19+i:02d}_{a}_mission_delay_1x2')
    responses=response_summary(rows);docs(rows,responses)
    study.write(DEST/'data/metrics.json',rows);prior.write_csv(DEST/'data/metrics.csv',rows)
    study.write(DEST/'data/responses.json',responses);prior.write_csv(DEST/'data/responses.csv',[{k:v for k,v in r.items() if k not in ['x','values']} for r in responses])
    study.write(DEST/'data/plot_series.json',prior.SERIES)
    for p in [Path(__file__),Path(study.__file__),Path(prior.__file__),Path(prior.chart.__file__),Path(prior.chart.style.__file__)]:prior.SOURCES[str(p.relative_to(ROOT))]=study.sha(p)
    study.write(DEST/'data/manifest.json',dict(unix=time.time(),complete=True,compared_arms=126,figures=prior.FIGURES,
        methods=study.METHODS,datasets=study.DATASETS,sensitivity_axes=AXES,source_sha256=prior.SOURCES,raw_records_published=False))
    print(dict(figures=len(prior.FIGURES),arms=len(rows),destination=str(DEST)),flush=True)


if __name__=='__main__':main()
