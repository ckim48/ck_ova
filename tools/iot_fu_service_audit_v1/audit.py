"""Audit recorded FU service latency without rerunning or changing algorithms.

The second timing calculation is a fixed-order accounting diagnostic, not a
new scheduler simulation or measured end-to-end latency.
"""
from pathlib import Path
import csv,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'results/iot_uav_remoteclip_v3_energy_overrun'
OUT=ROOT/'results/iot_fu_service_audit_v1'
sys.path.insert(0,str(ROOT/'tools/uav_real_vlm_v1'))
from energy import compute_cost,communication,PARAMETERS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HASHES={}
def read(path):
    payload=path.read_bytes();HASHES[str(path.relative_to(ROOT))]=hashlib.sha256(payload).hexdigest()
    return json.loads(payload)

def csv_write(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)

def stats(values,prefix):
    a=np.asarray(values,float)
    return {prefix+'_mean_s':float(a.mean()),prefix+'_median_s':float(np.median(a)),
            prefix+'_p95_s':float(np.percentile(a,95)),prefix+'_max_s':float(a.max())}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=read(SOURCE/'protocol.json');labels=protocol['plotting']['labels']
    rows=[];summary=[]
    for dataset in protocol['datasets']:
        request_ids=None
        for method,label in labels.items():
            folder=SOURCE/(dataset.lower()+'_seed0')/method
            manifest=read(folder/'manifest.json');assert manifest['status']=='completed_200FL_and_requests'
            requests=read(folder/'request_records.json')
            assert len(requests)==45 and len({r['burst'] for r in requests})==9
            ids={r['id'] for r in requests}
            if request_ids is None:request_ids=ids
            assert ids==request_ids
            tx=[read(p) for p in sorted((folder/'transactions').glob('*.json'))]
            work={}
            for p in sorted((folder/'work_attempts').glob('*.json')):
                r=read(p);assert r['status']=='transmitted';work[r['transaction']]=r
            assert manifest['completed_FL_rounds']==200
            assert all(r['deadline_s'] is None for r in requests)
            batch_rows=[]
            for r in requests:
                assert r['steps']==r['required_steps']==5
                serving=[t for t in tx if r['id'] in t['FU_request_ids']]
                assert len(serving)==5
                assert np.isclose(serving[0]['start_s'],r['first_service_s'],atol=1e-9,rtol=0)
                assert np.isclose(serving[-1]['end_s'],r['completed_s'],atol=1e-9,rtol=0)
                interval=[t for t in tx if t['start_s']>=r['arrival_s']-1e-9 and t['end_s']<=r['completed_s']+1e-9]
                assert sum(r['id'] in t['FU_request_ids'] for t in interval)==5
                compute=sum(float(compute_cost(work[t['sequence']]['compute']['device_macs'])[1].max()) for t in interval)
                downlink=sum(communication(t['radio'])[1] for t in interval)
                latency=r['completed_s']-r['arrival_s'];wait=r['first_service_s']-r['arrival_s']
                assert latency>=wait>=-1e-9
                quality=serving[-1]['evaluation']['per_request'][r['id']]
                fl_during_service=sum(t['kind']=='joint' for t in serving)
                record=dict(dataset=dataset,method=method,label=label,request_id=r['id'],burst=r['burst'],
                            arrival_s=r['arrival_s'],first_service_s=r['first_service_s'],completed_s=r['completed_s'],
                            logged_completion_latency_s=latency,logged_wait_s=wait,logged_service_span_s=latency-wait,
                            added_compute_critical_path_s=compute,added_downlink_s=downlink,
                            fixed_order_accounted_latency_s=latency+compute+downlink,
                            FU_steps=5,simultaneous_FL_updates=fl_during_service,
                            KL_to_retrained_reference_at_completion=quality['KL'],
                            completion_KL_at_most_existing_0p01_threshold=quality['KL']<=.01)
                rows.append(record);batch_rows.append(record)
            # Integrate the recorded pending-request curve; completions precede
            # arrivals at an identical time. This is an observed trace, not a
            # stationary queue-capacity experiment.
            events=sorted([(r['arrival_s'],1) for r in requests]+[(r['completed_s'],-1) for r in requests])
            count=peak=0;area=0.;last=events[0][0]
            for at,delta in events:
                area+=count*(at-last);count+=delta;assert count>=0
                peak=max(peak,count);last=at
            assert count==0 and np.isclose(area,sum(r['logged_completion_latency_s'] for r in batch_rows))
            bursts=sorted({r['burst'] for r in requests})
            overlap=sum(max(r['completed_s'] for r in requests if r['burst']==a)>min(r['arrival_s'] for r in requests if r['burst']==b)+1e-9 for a,b in zip(bursts,bursts[1:]))
            ul=sum(t['radio']['uplink_s'] for t in tx)
            s=dict(dataset=dataset,method=method,label=label,requests=45,bursts=9,FU_steps=225,
                   completed_requests=45,peak_pending_requests=peak,overlapping_burst_pairs=overlap,
                   deadlines_evaluated=False,
                   **stats([r['logged_completion_latency_s'] for r in batch_rows],'logged_latency'),
                   **stats([r['logged_wait_s'] for r in batch_rows],'logged_wait'),
                   **stats([r['fixed_order_accounted_latency_s'] for r in batch_rows],'fixed_order_accounted_latency'),
                   added_compute_mean_s=float(np.mean([r['added_compute_critical_path_s'] for r in batch_rows])),
                   added_downlink_mean_s=float(np.mean([r['added_downlink_s'] for r in batch_rows])),
                   total_adaptation_uplink_s=ul,
                   FU_carrying_uplink_s=sum(t['radio']['uplink_s'] for t in tx if t['FU_request_ids']),
                   observed_requests_per_all_workload_uplink_second=45/ul,
                   FL_updates_during_FU_transmissions=sum(t['kind']=='joint' for t in tx),
                   FU_only_transactions=sum(t['kind']=='fu' for t in tx),
                   completion_KL_at_most_0p01_requests=sum(r['completion_KL_at_most_existing_0p01_threshold'] for r in batch_rows),
                   mean_completion_KL=float(np.mean([r['KL_to_retrained_reference_at_completion'] for r in batch_rows])),
                   worst_completion_KL=max(r['KL_to_retrained_reference_at_completion'] for r in batch_rows),
                   final_worst_request_KL=manifest['evaluation']['worst_request_KL'])
            summary.append(s)
    csv_write(OUT/'request_latencies.csv',rows);csv_write(OUT/'summary.csv',summary)
    comparisons=[]
    for dataset in protocol['datasets']:
        subset={s['label']:s for s in summary if s['dataset']==dataset};p=subset['Proposed']
        for label,b in subset.items():
            if label=='Proposed':continue
            comparisons.append(dict(dataset=dataset,baseline=label,
                logged_mean_latency_reduction_percent=100*(1-p['logged_latency_mean_s']/b['logged_latency_mean_s']),
                fixed_order_mean_latency_reduction_percent=100*(1-p['fixed_order_accounted_latency_mean_s']/b['fixed_order_accounted_latency_mean_s'])))
    scope=dict(
        recorded_clock='Queue wait plus modeled uplink; excludes local compute and global-model downlink. Completion means five prescribed FU steps, not certified deletion.',
        fixed_order_accounting='Diagnostic only: preserve logged arrival/transaction order; add per-transaction maximum device compute time and complete model broadcast time to each request interval. Includes final broadcast to all UAVs. No overlap is assumed. Does not recompute periodic triggers, request arrivals, channel designs or model updates.',
        downlink='Existing model: (20480*32+512) bits / 1 Mbps = 0.655872 seconds per nonzero uplink transaction.',
        arrivals='Triggered by FL progress, not common external timestamps. Nine bursts of five requests, one seed. Requests in the same burst are correlated.',
        throughput='All methods completed the same 45 requests. Requests per occupied uplink second measures realized radio-work efficiency, not maximum sustainable load or end-to-end throughput.',
        deadline='Not evaluated: current deadline_s is null. Historical old_* deadlines are not active and are not reused.',
        quality='KL<=0.01 is an existing reference-similarity diagnostic, not a privacy/deletion certificate. Fast five-step completion is distinct from matching a quality threshold.',
        controls='Proposed/S-OF+ use the same CLIPErase-inspired objective; S-AC/AC-OF use the proxy objective. S-OF+ permits the previously authorized network-energy overruns.',
        inference='A real-time scheduler with compute/downlink, fixed external arrivals and a load sweep is needed to establish end-to-end latency, capacity and deadline compliance.')
    for path in [Path(__file__),ROOT/'tools/uav_real_vlm_v1/energy.py',ROOT/'tools/overnight_energy_v2/korea_joint_seed0_pilot_v2/round_schedule.py',ROOT/'tools/overnight_energy_v2/korea_round_lr_sync_rsitmd_seed0_v6/comparison_schedule.py']:
        HASHES[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT/'audit.json').write_text(json.dumps(dict(status='completed',source=str(SOURCE.relative_to(ROOT)),
        request_rows=len(rows),summary=summary,comparisons=comparisons,scope=scope,energy_parameters=PARAMETERS,input_sha256=HASHES),indent=2,allow_nan=False)+'\n')
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'font.size':9,'axes.linewidth':.65,'pdf.fonttype':42,'ps.fonttype':42})
    colors=['#167D8D','#4978A8','#D69B48','#8A769F']
    for field,ylabel,name in [('logged_latency_mean_s','Mean FU procedure latency (s)','fu_latency_recorded_1x2'),
                              ('fixed_order_accounted_latency_mean_s','Fixed-order accounted latency (s)','fu_latency_fixed_order_1x2')]:
        fig,axes=plt.subplots(1,2,figsize=(7.16,2.7),layout='constrained')
        for ax,dataset in zip(axes,protocol['datasets']):
            subset=[s for s in summary if s['dataset']==dataset]
            vals=[s[field] for s in subset]
            bars=ax.bar(range(4),vals,width=.6,color=colors,edgecolor='white',linewidth=.5)
            ax.set_xticks(range(4),[s['label'] for s in subset]);ax.set_title(dataset,pad=7);ax.set_ylabel(ylabel)
            ax.set_ylim(0,max(vals)*1.2);ax.spines[['top','right']].set_visible(False)
            ax.set_axisbelow(True);ax.yaxis.grid(True,color='#e6e9ed',linewidth=.55)
            ax.bar_label(bars,labels=[f'{v:.3f}' for v in vals],padding=3,fontsize=8,color='black')
        for extension in ['png','pdf','svg']:
            path=OUT/f'{name}.{extension}';fig.savefig(path,dpi=450,bbox_inches='tight')
            if extension=='svg':path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
        plt.close(fig)
    print(json.dumps(dict(summary=summary,comparisons=comparisons),indent=2))

if __name__=='__main__':main()
