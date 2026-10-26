"""Raw five-method 1x2 figures; missing/infeasible points remain gaps."""
from common import *
import csv, math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0,str(ROOT/'tools'))
from plot_event_mission_reference_style import configure
METRICS = {
    'total_energy_j':('Total energy (kJ)',1000),
    'FL_FU_completion_delay_s':('FL/FU completion delay (s)',1),
    'mission_delay_s':('Mission delay (s)',1),
    'total_delay_s':('Total delay (s)',1),
    'FU_mean_completion_latency_s':('Mean FU latency (s)',1),
    'FL_FU_active_service_s':('FL/FU active service (s)',1),
    'energy_hover_j':('Hovering energy (kJ)',1000),
    'energy_communication_j':('Communication energy (J)',1),
    'adaptation_RF_energy_j':('Adaptation RF energy (J)',1),
    'uplink_s':('Occupied uplink time (s)',1),
    'loss':('Retained loss',1), 'mR':('Retained mR (%)',.01),
}
COLORS=['#4C72B0','#9470B3','#EFAA54','#D65A50','#66B366']
MARKERS=['o','D','s','^','v'];STYLES=['-','--','-.',':',(0,(5,2))]
def generate():
    out=OUT/'figures';out.mkdir(parents=True,exist_ok=True)
    rows=[read(f) for f in sorted((OUT/'cases').glob('*/*/*/outcome.json'))]
    index={(r['case'],r['dataset'],r['method']):r for r in rows}
    flat=[dict(case=r['case'],dataset=r['dataset'],method=r['method'],status=r['status'],**{k:r.get('metrics',{}).get(k) for k in METRICS}) for r in rows]
    with (out/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['case','dataset','method','status']+list(METRICS),lineterminator='\n');w.writeheader();w.writerows(flat)
    write(out/'metrics.json',dict(rows=flat,source='actual per-arm energy_and_delay and outcome files',seed=0))
    configure();rendered=[]
    pairs=[('historical',['RSITMD','CapERA']),('uav_extra',EXTRA_DATASETS)]
    for pair,datasets in pairs:
        for axis,(xlabel,values,_) in AXES.items():
            # Render a sweep only after every arm is terminal for this pair.
            if not all((case_id(axis,v),d,m) in index for v in values for d in datasets for m in METHODS):continue
            for key,(ylabel,scale) in METRICS.items():
                name=f'{pair}_{axis}_{key}_1x2'
                if all((out/(name+'.'+ext)).exists() for ext in ['png','pdf','svg']):rendered.append(name);continue
                fig,axes=plt.subplots(1,2,figsize=(7.4,3.735))
                fig.subplots_adjust(left=.115,right=.985,bottom=.22,top=.80,wspace=.46)
                for ax,dataset in zip(axes,datasets):
                    for (method,label),color,marker,style in zip(METHODS.items(),COLORS,MARKERS,STYLES):
                        y=[index[(case_id(axis,v),dataset,method)].get('metrics',{}).get(key,float('nan'))/scale for v in values]
                        ax.plot(values,y,color=color,marker=marker,mfc='white',mew=1.,ms=5,linestyle=style,label=label)
                    ax.set_box_aspect(.9);ax.set_xlabel(xlabel);ax.set_ylabel(ylabel,labelpad=4)
                    ax.set_xticks(values)
                    if axis in ['bandwidth','downlink','power','network_cap','collection_slot']:ax.set_xscale('log',base=2);ax.set_xticks(values,[f'{v:g}' for v in values]);ax.minorticks_off()
                    ax.set_xlim(values[0],values[-1]);ax.grid(True,ls=':',lw=.5);ax.set_axisbelow(True)
                    ax.tick_params(direction='out',pad=3,labelsize=12)
                    assert not ax.texts
                handles,labels=axes[0].get_legend_handles_labels()
                fig.legend(handles,labels,ncol=5,loc='upper center',bbox_to_anchor=(.5,1.0),fontsize=12,handlelength=1.6,columnspacing=.75)
                for i,(ax,dataset) in enumerate(zip(axes,datasets)):
                    pos=ax.get_position();fig.text((pos.x0+pos.x1)/2,.015,f'({chr(97+i)}) {dataset}',ha='center',va='bottom',fontsize=14)
                name=f'{pair}_{axis}_{key}_1x2'
                for ext in ['png','pdf','svg']:fig.savefig(out/(name+'.'+ext),dpi=350,bbox_inches='tight')
                plt.close(fig);rendered.append(name)
    # Actual baseline loss / mR convergence; preserve repeated FL observations.
    for pair,datasets in pairs:
        if not all(('baseline',d,m) in index for d in datasets for m in METHODS):continue
        for metric,ylabel,scale in [('loss','Retained loss',1),('mR','Retained mR (%)',100)]:
            name=f'{pair}_{metric}_convergence_1x2'
            if all((out/(name+'.'+ext)).exists() for ext in ['png','pdf','svg']):rendered.append(name);continue
            fig,axes=plt.subplots(1,2,figsize=(7.4,3.735));fig.subplots_adjust(left=.115,right=.985,bottom=.22,top=.80,wspace=.46)
            for ax,d in zip(axes,datasets):
                dd=OUT/'cases/baseline'/(d.lower()+'_seed0')
                prefix=read(dd/'initial20/rounds.json')
                for (m,label),color,marker,style in zip(METHODS.items(),COLORS,MARKERS,STYLES):
                    p=[(r['round'],r['evaluation']['retained'][metric]*scale) for r in prefix if r.get('evaluation')]
                    for f in sorted((dd/m/'evaluations').glob('*.json')):
                        r=read(f);p.append((r['completed_FL_rounds'],r['evaluation']['retained'][metric]*scale))
                    if p:ax.plot(*zip(*p),color=color,marker=marker,markevery=max(1,len(p)//8),mfc='white',ms=4,linestyle=style,label=label)
                ax.set_box_aspect(.9);ax.set_xlim(0,200);ax.set_xticks([0,50,100,150,200]);ax.set_xlabel('Completed FL rounds');ax.set_ylabel(ylabel);ax.grid(ls=':')
            fig.legend(*axes[0].get_legend_handles_labels(),ncol=5,loc='upper center',fontsize=12,handlelength=1.6,columnspacing=.75)
            for i,(ax,d) in enumerate(zip(axes,datasets)):
                p=ax.get_position();fig.text((p.x0+p.x1)/2,.015,f'({chr(97+i)}) {d}',ha='center',fontsize=14)
            name=f'{pair}_{metric}_convergence_1x2'
            for ext in ['png','pdf','svg']:fig.savefig(out/(name+'.'+ext),dpi=350,bbox_inches='tight')
            plt.close(fig);rendered.append(name)
    summary=dict(terminal_arms=len(rows),completed=sum(r['status']=='completed' for r in rows),
                 infeasible=sum('infeasible' in r['status'] or 'no_feasible' in r['status'] for r in rows),
                 failed=sum(r['status']=='failed' for r in rows),rendered=rendered,
                 axes={k:dict(label=v[0],values=v[1],baseline=v[2]) for k,v in AXES.items()},
                 limitations=['Paired seed0 sensitivity; no confidence intervals.','One factor varied, common preflight learning checkpoints reused.','Missing or infeasible values are not zero or interpolated.','S-OF+ may exceed network TX budget; other methods retain hard cap.','Real visual-language replay on synthetic UAV/IoT mission, not flight measurements.'])
    write(out/'summary.json',summary);return summary

if __name__=='__main__':print(json.dumps(generate(),indent=2))
