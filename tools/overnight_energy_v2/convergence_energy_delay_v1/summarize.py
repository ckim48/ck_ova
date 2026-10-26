"""Convergence, communication energy, and observed target-hitting cost reports."""
import argparse,csv,json
from pathlib import Path


def dump_csv(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def summarize(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plan=json.loads((root/'plan.json').read_text())
    last=max(plan['config']['request_rounds']); methods=plan['methods']
    series={};status=[];targets=[];flat=[]
    for method in methods:
        folder=root/method
        mf=json.loads((folder/'manifest.json').read_text()) if (folder/'manifest.json').exists() else {'status':'not_started'}
        rows=[json.loads(x) for x in (folder/'rounds.jsonl').read_text().splitlines() if x.strip()] if (folder/'rounds.jsonl').exists() else []
        init=folder/'initial_evaluation.json'
        if init.exists():
            rows.insert(0,dict(round=0,validation=json.loads(init.read_text()),cumulative_communication_energy_j=0.,
                cumulative_uplink_s=0.,cumulative_delay_without_optimizer_or_overhead_s=0.,
                cumulative_delay_with_optimizer_without_overhead_s=0.,request_users_total=0,individual_request_deliveries_total=0))
        series[method]=rows
        final=rows[-1] if rows else {}
        eligible=final.get('request_users_total',0)==final.get('individual_request_deliveries_total',0)
        status.append(dict(method=method,status=mf['status'],completed_rounds=final.get('round',0),
            final_loss=final.get('validation',{}).get('loss'),final_accuracy=final.get('validation',{}).get('accuracy'),
            energy_j=final.get('cumulative_communication_energy_j'),uplink_s=final.get('cumulative_uplink_s'),
            delay_with_optimizer_s=final.get('cumulative_delay_with_optimizer_without_overhead_s'),
            required_request_deliveries=final.get('request_users_total',0),
            individual_request_deliveries=final.get('individual_request_deliveries_total',0),
            eligible_completed_task=bool(mf['status']=='completed' and eligible)))
        for x in rows:
            flat.append(dict(method=method,round=x['round'],loss=x['validation']['loss'],accuracy=x['validation']['accuracy'],
                energy_j=x['cumulative_communication_energy_j'],uplink_s=x['cumulative_uplink_s'],
                delay_without_optimizer_s=x['cumulative_delay_without_optimizer_or_overhead_s'],
                delay_with_optimizer_s=x['cumulative_delay_with_optimizer_without_overhead_s']))
        for metric,levels in plan['target_sweep'].items():
            for threshold in levels:
                for stage in ['any_round','after_all_requests']:
                    pool=[x for x in rows if stage=='any_round' or x['round']>=last]
                    for stable in [1,3]:
                        hit=None
                        for i in range(len(pool)-stable+1):
                            window=pool[i:i+stable]
                            if all((x['validation'][metric]<=threshold if metric=='loss' else x['validation'][metric]>=threshold) for x in window):
                                hit=window[-1];break
                        t=dict(method=method,metric=metric,target=threshold,stage=stage,consecutive_evaluations=stable,
                            status='reached' if hit else 'not_reached_in_observed_rounds',run_status=mf['status'],
                            eligible_completed_task=bool(mf['status']=='completed' and eligible))
                        if hit:
                            t.update(round=hit['round'],energy_j=hit['cumulative_communication_energy_j'],
                                uplink_s=hit['cumulative_uplink_s'],delay_without_optimizer_s=hit['cumulative_delay_without_optimizer_or_overhead_s'],
                                delay_with_optimizer_s=hit['cumulative_delay_with_optimizer_without_overhead_s'],
                                all_individual_requests_delivered=hit['request_users_total']==hit['individual_request_deliveries_total'])
                            for overhead in plan['overhead_seconds_sensitivity']:
                                t[f'delay_with_optimizer_and_{overhead:g}s_overhead_per_round']=t['delay_with_optimizer_s']+hit['round']*overhead
                        targets.append(t)
    dump_csv(root/'convergence.csv',flat);dump_csv(root/'target_costs.csv',targets);dump_csv(root/'method_summary.csv',status)
    (root/'summary.json').write_text(json.dumps(dict(methods=status,timing_scope=plan['timing_scope'],
        caveats=['single-seed pilot; no significance claim','failure or incomplete run is not an energy winner',
        'AirComp additive reference does not preserve individual request vectors',
        'RTX3080 and server CPU timing are proxies, not measured UAV time',
        'control/downlink overhead remains a parameter; neither evaluation time nor total server wall time is confused with deployment delay',
        'communication energy only; compute and flight energy not measured',
        'forget loss and directional request handling are not deletion certificates']),indent=2)+'\n')
    if flat:
        fig,axes=plt.subplots(2,2,figsize=(12,8))
        for method,rows in series.items():
            if not rows:continue
            x=[r['round'] for r in rows]
            for ax,key,label in [(axes[0,0],'loss','Validation cross-entropy'),(axes[0,1],'accuracy','Validation accuracy')]:
                ax.plot(x,[r['validation'][key] for r in rows],label=method);ax.set_ylabel(label)
            axes[1,0].plot(x,[r['cumulative_communication_energy_j'] for r in rows],label=method)
            axes[1,1].plot(x,[r['cumulative_delay_with_optimizer_without_overhead_s'] for r in rows],label=method)
        axes[1,0].set_ylabel('Cumulative communication energy bound (J)')
        axes[1,1].set_ylabel('Delay proxy incl. optimizer, excl. overhead (s)')
        for ax in axes.flat:
            ax.set_xlabel('Global round');ax.grid(alpha=.25)
            for k in plan['config']['request_rounds']:ax.axvline(k,color='grey',alpha=.2,linewidth=.7)
        axes[0,0].legend(fontsize=8);fig.tight_layout();fig.savefig(root/'convergence_energy_delay.png',dpi=180);fig.savefig(root/'convergence_energy_delay.pdf');plt.close(fig)
    return status


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
    print(json.dumps(summarize(a.root),indent=2))
