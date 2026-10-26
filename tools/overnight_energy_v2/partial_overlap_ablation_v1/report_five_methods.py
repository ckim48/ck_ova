"""Five-method utility target costs, with explicit incomplete-task qualifications."""
from pathlib import Path
import csv,json,time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
MAIN=ROOT/'results/overnight_energy_v2/partial_overlap_tx_v1/learning_comparison'
EXTRA=ROOT/'results/overnight_energy_v2/partial_overlap_ablation_v1'
OUT=EXTRA/'five_method_report'
METHODS=['proposed','aircomp_oma','aircomp_only','oma_only','hybrid_no_coupling']

def csv_write(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def first_hit(rows,metric,target,count):
    for i in range(len(rows)-count+1):
        w=rows[i:i+count]
        if [x['round'] for x in w]!=list(range(w[0]['round'],w[0]['round']+count)):continue
        if all(x[metric]<=target if metric=='loss' else x[metric]>=target for x in w):return w[-1]
    return None

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plan=json.loads((MAIN/'plan.json').read_text());summary=[];curves=[];targets=[]
    for method in METHODS:
        d=(EXTRA if method=='hybrid_no_coupling' else MAIN)/method
        if not (d/'manifest.json').exists():
            summary.append(dict(method=method,status='not_started',observed_rounds=0));continue
        manifest=json.loads((d/'manifest.json').read_text());r=[];e=ee=t=0.
        f=d/'rounds.jsonl'
        lines=f.read_text().splitlines() if f.exists() else []
        for i,line in enumerate(lines):
            try:x=json.loads(line)
            except json.JSONDecodeError:
                if i!=len(lines)-1:raise
                continue
            tx=x['transmission'];assert tx['status']=='feasible'
            e+=tx['energy_bound_j'];ee+=tx['expected_energy_j'];t+=tx['uplink_s']
            r.append(dict(method=method,round=x['round'],post_request_rounds=x['round']-60,
                          loss=x['validation']['loss'],accuracy=x['validation']['accuracy'],
                          tx_energy_bound_j=e,tx_energy_expected_j=ee,total_delay_s=t,
                          requests_required=x['request_users_total'],requests_individually_delivered=x['individual_request_deliveries_total']))
        if not r:
            summary.append(dict(method=method,status=manifest['status'],observed_rounds=0));continue
        curves+=r;summary.append(dict(status=manifest['status'],observed_rounds=len(r),**r[-1]))
        initial=json.loads((d/'initial_evaluation.json').read_text())
        levels={k:list(v) for k,v in plan['target_sweep'].items()}
        levels['loss']=sorted(set(levels['loss']+[initial['loss']-x for x in [0,.001,.002,.005,.01,.02,.05]]))
        levels['accuracy']=sorted(set(levels['accuracy']+[initial['accuracy']+x for x in [0,.001,.002,.005,.01,.02,.05]]))
        for metric,values in levels.items():
            for target in values:
                for count in [1,3]:
                    hit=first_hit(r,metric,target,count)
                    item=dict(method=method,metric=metric,target=target,consecutive_rounds=count,
                              already_met_before_request=initial[metric]<=target if metric=='loss' else initial[metric]>=target,
                              status='reached' if hit else 'not_reached_in_observed_window',run_status=manifest['status'])
                    if hit:item.update(hit)
                    targets.append(item)
    metadata=dict(updated_unix=time.time(),metric_scope='Cumulative uplink transmit energy and occupied uplink time only.',
                  aircomp_definition='Existing aircomp_only: learning and quantized deletion updates added into one analog aggregate, one transmission phase. Two separate AirComp phases are not this baseline.',
                  crossmodal_ablation='Proposed vs hybrid_no_coupling differs only in local visual-text cross-modal preconditioning.',
                  hypotheses=['More rounds do not by themselves imply greater cumulative energy; actual per-round energy is summed up to the same prespecified utility target.',
                              'Inability to recover individual updates does not itself prove slower accuracy convergence. Utility target cost does not certify matched forgetting quality.'],
                  qualifications=['Common historical training prefix excluded. A single seed and K=5 deletion burst; not a final full-study claim.',
                                  'Not reached in 20 rounds is right-censored, not infinite energy or proof of divergence.',
                                  'Incomplete/infeasible runs are not energy winners. AirComp-only does not satisfy individual-vector recovery; utility-only comparisons remain visible.'],summary=summary)
    (OUT/'summary.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (OUT/'target_costs.json').write_text(json.dumps(targets,indent=2)+'\n')
    csv_write(OUT/'summary.csv',summary);csv_write(OUT/'curves.csv',curves);csv_write(OUT/'target_costs.csv',targets)
    if curves:
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,3,figsize=(13,7))
        for method in METHODS:
            series=[x for x in curves if x['method']==method]
            if not series:continue
            for col,(xkey,xlabel,scale) in enumerate([('post_request_rounds','Post-request rounds',1),('tx_energy_bound_j','Cumulative TX energy bound (mJ)',1000),('total_delay_s','Cumulative uplink transmission time (s)',1)]):
                for row,ykey in enumerate(['loss','accuracy']):
                    axes[row,col].plot([x[xkey]*scale for x in series],[x[ykey] for x in series],label=method)
                    axes[row,col].set_xlabel(xlabel);axes[row,col].set_ylabel('Validation '+ykey);axes[row,col].grid(alpha=.25)
        axes[0,0].legend(fontsize=8);fig.suptitle('Partial-overlap pilot: utility vs rounds, transmit energy and uplink time')
        fig.tight_layout();fig.savefig(OUT/'five_method_curves.png',dpi=160);fig.savefig(OUT/'five_method_curves.pdf');plt.close(fig)
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
