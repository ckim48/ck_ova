"""Paper-style forgetting figures and complete, outcome-independent exports."""
import csv
import json
import math
from pathlib import Path
import shutil

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter
from matplotlib.backends.backend_pdf import PdfPages

import forgetting_quality_v1 as study

ROOT,OUT,read,write,sha=study.ROOT,study.OUT,study.read,study.write,study.sha
DEST=ROOT/'final_final_charts/forgetting_quality'
DATASETS,METHODS=study.DATASETS,study.METHODS
COLORS=['#4C72B0','#EFAA54','#66B366','#C65365']
SIZE=(7.4,3.735)
RECTS=[(.095,.22,.385,.65),(.595,.22,.385,.65)]
PLOT_ROWS=[]
AUDITS=[]
COLLECTION=None


def table(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');writer.writeheader();writer.writerows(rows)


def nice_range(values,include_zero=True):
    lo,hi=min(values),max(values)
    if include_zero:lo=min(lo,0);hi=max(hi,0)
    if hi==lo:hi=lo+1
    raw=(hi-lo)/4
    power=10**math.floor(math.log10(raw))
    step=next(v*power for v in [1,2,2.5,5,10] if v*power>=raw)
    lower=math.floor(lo/step)*step;upper=math.ceil(hi/step)*step
    if upper==hi and hi>0:upper+=step
    ticks=np.arange(lower,upper+step*.25,step)
    decimals=max(0,-math.floor(math.log10(step)))+int(abs(step/(10**math.floor(math.log10(step)))-2.5)<1e-8)
    return ticks,decimals


def layout(ylabel,values,legend=None):
    fig=plt.figure(figsize=SIZE);axes=[fig.add_axes(r) for r in RECTS]
    ticks,decimals=nice_range(values)
    for i,(ax,ds) in enumerate(zip(axes,DATASETS)):
        ax.set_xlim(-.4,3.4);ax.set_xticks(range(4),list(METHODS.values()),fontsize=12)
        ax.set_ylabel(ylabel,fontsize=14,labelpad=3)
        ax.set_ylim(ticks[0],ticks[-1]);ax.set_yticks(ticks)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.'+str(decimals)+'f'))
        ax.tick_params(labelsize=12,pad=3,direction='out',length=3)
        ax.set_axisbelow(True);ax.grid(axis='y',ls=':',lw=.5,color='#bfc2c8')
        for spine in ax.spines.values():spine.set_visible(True);spine.set_linewidth(.8)
        ax.text(.5,-.25,'('+chr(97+i)+') '+ds,transform=ax.transAxes,ha='center',va='top',fontsize=14)
        if ticks[0]<0:ax.axhline(0,color='#4a4a4a',lw=.6)
    if legend:
        fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.537,.885),
                   ncol=len(legend),frameon=False,fontsize=12,handlelength=1.5,
                   columnspacing=1.4,handletextpad=.45,borderaxespad=0)
    return fig,axes


def save(fig,name):
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    geometry=[]
    for ax in fig.axes:
        b=ax.get_window_extent(renderer)
        labels=[t.get_window_extent(renderer) for t in ax.get_xticklabels()]
        gaps=[(b.x0-a.x1)*72/fig.dpi for a,b in zip(labels,labels[1:])]
        assert min(gaps)>5,gaps
        for artist in [ax.yaxis.label,*ax.get_xticklabels(),*ax.get_yticklabels(),*ax.texts]:
            bounds=artist.get_window_extent(renderer)
            assert bounds.x0>=0 and bounds.y0>=0 and bounds.x1<=fig.bbox.x1 and bounds.y1<=fig.bbox.y1, (name,artist.get_text(),bounds)
        assert np.isclose(ax.get_yticks()[0],ax.get_ylim()[0])
        assert np.isclose(ax.get_yticks()[-1],ax.get_ylim()[1])
        assert np.allclose(np.diff(ax.get_yticks()),np.diff(ax.get_yticks())[0])
        geometry.append(dict(width_pt=b.width*72/fig.dpi,height_pt=b.height*72/fig.dpi,
                             category_font_pt=12,ylabel_font_pt=14,category_gaps_pt=gaps))
    assert np.allclose([g['width_pt'] for g in geometry],geometry[0]['width_pt'])
    assert np.allclose([g['height_pt'] for g in geometry],geometry[0]['height_pt'])
    for suffix in ['pdf','png','svg']:
        file=DEST/(name+'.'+suffix)
        fig.savefig(file,dpi=220,facecolor='white',bbox_inches=None)
        if suffix=='svg':file.write_text('\n'.join(line.rstrip() for line in file.read_text().splitlines())+'\n')
    if COLLECTION is not None:COLLECTION.savefig(fig,bbox_inches=None)
    AUDITS.append(dict(figure=name,panels=geometry,canvas_in=SIZE))
    plt.close(fig)


def metric(ds,m,condition,cohort,key):
    path=OUT/ds.lower()/m/condition/'metrics.json'
    return read(path)[cohort][key]


def reference(ds,m,key,cohort='forget'):
    return read(OUT/ds.lower()/'references'/study.POLICIES[m]/'metrics.json')[cohort][key]


def pair(name,ylabel,labels,fetch,scale=1):
    values={(ds,m):[scale*v for v in fetch(ds,m)] for ds in DATASETS for m in METHODS}
    legends=[Patch(facecolor='white',edgecolor='#222',hatch=h,label=l) for l,h in zip(labels,['//','..'])]
    fig,axes=layout(ylabel,[v for row in values.values() for v in row],legends)
    for ds,ax in zip(DATASETS,axes):
        for i,(m,color) in enumerate(zip(METHODS,COLORS)):
            for j,(v,h) in enumerate(zip(values[(ds,m)],['//','..'])):
                ax.bar(i+(j-.5)*.34,v,width=.32,color=color,hatch=h,edgecolor='#242424',lw=.7)
                PLOT_ROWS.append(dict(figure=name,dataset=ds,method=m,label=METHODS[m],series=labels[j],value=v))
    save(fig,name)


def paired_values(ds,m,key):
    plan=read(study.current.source('baseline',ds,m).parent/'plan.json')
    idx=study.indices(plan,'forget')
    a=np.load(OUT/ds.lower()/m/'exact_no_fu/per_video.npz')[key][idx]
    b=np.load(OUT/ds.lower()/m/'exact_fu/per_video.npz')[key][idx]
    return a-b


def bootstrap(values,seed=1030):
    rng=np.random.default_rng(seed)
    means=np.array([rng.choice(values,len(values),replace=True).mean() for _ in range(2000)])
    return np.quantile(means,[.025,.975]).tolist()


def association():
    name='22_forget_association_change_1x2'
    rows=[]
    for ds in DATASETS:
        for m in METHODS:
            v=1000*paired_values(ds,m,'association_margin');ci=bootstrap(v)
            rows.append(dict(dataset=ds,method=m,mean=float(v.mean()),low=ci[0],high=ci[1],
                        weakened_percent=100*float(np.mean(v>1e-9)),unchanged_percent=100*float(np.mean(np.abs(v)<=1e-9)),
                        strengthened_percent=100*float(np.mean(v< -1e-9)),videos=len(v)))
    fig,axes=layout(r'Association reduction ($10^{-3}$)',[r[k] for r in rows for k in ['low','high']])
    for ds,ax in zip(DATASETS,axes):
        for i,(m,color) in enumerate(zip(METHODS,COLORS)):
            r=next(r for r in rows if (r['dataset'],r['method'])==(ds,m))
            ax.bar(i,r['mean'],width=.58,color=color,edgecolor='#242424',hatch='//',lw=.7)
            ax.errorbar(i,r['mean'],yerr=[[r['mean']-r['low']],[r['high']-r['mean']]],fmt='none',ecolor='#222',capsize=3,lw=.85)
            PLOT_ROWS.append(dict(figure=name,series='Exact no-FU minus exact FU margin',**r))
    save(fig,name);write(DEST/'data/association_summary.json',rows)
    return rows


def request_box():
    name='25_request_retraining_kl_1x2'
    rows={(ds,m):read(OUT/ds.lower()/m/'actual/request_KL.json') for ds in DATASETS for m in METHODS}
    vals=[1000*r['matched_KL'] for group in rows.values() for r in group]+[10.]
    fig,axes=layout(r'Request KL ($10^{-3}$)',vals,[plt.Line2D([],[],color='#444',ls='--',lw=.8,label='Diagnostic tolerance (0.01)')])
    for ds,ax in zip(DATASETS,axes):
        ax.axhline(10.,color='#444',ls='--',lw=.8)
        for i,(m,color) in enumerate(zip(METHODS,COLORS)):
            values=[1000*r['matched_KL'] for r in rows[(ds,m)]]
            ax.boxplot([values],positions=[i],widths=.52,patch_artist=True,manage_ticks=False,
                       boxprops=dict(facecolor=color,edgecolor='#242424',lw=.8),
                       medianprops=dict(color='white',lw=1.5),whiskerprops=dict(color=color),
                       capprops=dict(color=color),flierprops=dict(marker='.',markersize=3,markeredgecolor=color))
            for r in rows[(ds,m)]:PLOT_ROWS.append(dict(figure=name,dataset=ds,method=m,series='Policy-matched',plot_value=1000*r['matched_KL'],**r))
    save(fig,name)


def collect():
    full=[];per=[];events=[];fresh=[]
    for ds in DATASETS:
        assert read(OUT/ds.lower()/'finished.json')['passed']
        plan=read(study.current.source('baseline',ds,'gate_select16').parent/'plan.json')
        forgotten=set(plan['forgotten_images']);retained=set(plan['validation_images'])
        for policy in ['common','priority','equal']:
            fresh.append(read(OUT/ds.lower()/'references'/policy/'finished.json'))
        for m in METHODS:
            for condition in ['actual','exact_fu','exact_no_fu']:
                summary=read(OUT/ds.lower()/m/condition/'metrics.json')
                for group,metrics in summary.items():
                    full.append(dict(dataset=ds,method=m,label=METHODS[m],condition=condition,cohort=group,**metrics))
                arrays=np.load(OUT/ds.lower()/m/condition/'per_video.npz')
                for i,vid in enumerate(arrays['video_ids']):
                    per.append(dict(dataset=ds,method=m,condition=condition,video_id=int(vid),
                                    cohort='forget' if int(vid) in forgotten else 'retained',
                                    **{k:float(arrays[k][i]) for k in arrays.files if k!='video_ids'}))
                if condition!='actual':
                    for r in read(OUT/ds.lower()/m/condition/'request_events.json'):
                        events.append(dict(dataset=ds,method=m,condition=condition,
                            **{k:v for k,v in r.items() if k not in ['before','after']},
                            **{'before_'+k:v for k,v in r['before'].items()},
                            **{'after_'+k:v for k,v in r['after'].items()}))
        for name in ['warm20','initialization']:
            for cohort,values in read(OUT/ds.lower()/name/'metrics.json').items():
                full.append(dict(dataset=ds,method='shared',label=name,condition=name,cohort=cohort,**values))
        for policy in ['common','priority','equal']:
            for cohort,values in read(OUT/ds.lower()/'references'/policy/'metrics.json').items():
                full.append(dict(dataset=ds,method='reference',label=policy,condition='retrain200',cohort=cohort,**values))
    table(DEST/'data/all_metrics.csv',full);table(DEST/'data/per_video_metrics.csv',per)
    table(DEST/'data/request_before_after.csv',events);write(DEST/'data/fresh_retraining_checks.json',fresh)
    return full


def docs(association_rows):
    rows=[]
    for ds in DATASETS:
        actual=read(OUT/ds.lower()/'summary.json')
        for r in actual:
            m=r['method'];a=next(x for x in association_rows if x['dataset']==ds and x['method']==m)
            rows.append(dict(dataset=ds,method=METHODS[m],retained_mR=100*r['metrics']['retained']['mR'],
                forget_mR=100*r['metrics']['forget']['mR'],matched_retrain_forget_mR=100*reference(ds,m,'mR'),
                common_KL=r['metrics']['forget']['common_KL'],matched_KL=r['metrics']['forget']['matched_KL'],
                exact_no_FU_matched_KL=metric(ds,m,'exact_no_fu','forget','matched_KL'),
                exact_FU_matched_KL=metric(ds,m,'exact_fu','forget','matched_KL'),
                worst_request_matched_KL=r['worst_request_matched_KL'],
                association_reduction=a['mean']/1000,association_weakened_percent=a['weakened_percent']))
    table(DEST/'data/performance_summary.csv',rows)
    cap=next(r for r in rows if r['dataset']=='CapERA' and r['method']=='Proposed')
    air=next(r for r in rows if r['dataset']=='AirScape' and r['method']=='Proposed')
    paired_statement=(f"Its exact matched KL changes from{cap['exact_no_FU_matched_KL']:.6f} to{cap['exact_FU_matched_KL']:.6f} on CapERA "
                      f"and from{air['exact_no_FU_matched_KL']:.6f} to{air['exact_FU_matched_KL']:.6f} on AirScape (full precision in the tables).")
    lines=['# Forgetting quality validation','',
      'Current Proposed = P-KS. All original numerical charts and model checkpoints are unchanged.',
      'Two datasets, four methods, 200 FL rounds and 45 deletion requests (225 FU steps). All forgotten videos and their native captions are evaluated: CapERA 247 videos (five captions each); AirScape 358 videos (one caption each). Each retained evaluation set contains 256 videos. This remains a seed0 development experiment, not a new independent test benchmark.','',
      '## Figures','',
      'Open `forgetting_quality_all.pdf` for all eight figures in one file. Individual PDF/PNG/SVG files and LaTeX snippets are also included.','',
      '| File | What it measures |','|---|---|',
      '| 20_forget_retrieval_1x2 | Actual wireless final-model forget mR and policy-matched retraining mR. Lower forget mR alone does not establish unlearning. |',
      '| 21_forget_retraining_kl_1x2 | Actual forget-set KL to the original common and policy-matched retraining references. Both counterfactuals are retained. |',
      '| 22_forget_association_change_1x2 | Paired reduction of positive-vs-negative cosine margin: exact no-FU minus exact FU. Positive means weaker associations due to the FU intervention; negative means strengthening. Error bars: 2,000 paired video-bootstrap replicates, conditional on the fitted models. |',
      '| 23_forget_retrieval_fu_control_1x2 | Same200-round no-FU vs FU comparison with exact aggregation and identical original transaction schedules. |',
      '| 24_wireless_error_forgetting_1x2 | KL to common retraining for exact vs actual wireless FU trajectories. This separates aggregation sensitivity from unlearning success. |',
      '| 25_request_retraining_kl_1x2 | Distribution of policy-matched KL over all45 requests. The0.01 line is the existing diagnostic tolerance, not a proven deletion criterion. |',
      '| 26_forget_association_before_after_1x2 | Actual deleted-pair cosine before requests (warm20) and after200-round FL/FU. Different training horizons: descriptive only; use22/23 for the paired intervention. |',
      '| 27_forget_kl_fu_control_1x2 | Paired exact no-FU and exact FU distance to policy-matched retraining, at the identical200-round horizon. This tests whether FU improves that particular unlearning criterion. |','',
      '## Results','',
      '| Dataset | Method | Retained mR (%) | Forget mR (%) | Retrain forget mR (%) | Common KL | Matched KL | Worst request matched KL | Association reduction |',
      '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append('| {dataset} | {method} | {retained_mR:.3f} | {forget_mR:.3f} | {matched_retrain_forget_mR:.3f} | {common_KL:.6f} | {matched_KL:.6f} | {worst_request_matched_KL:.6f} | {association_reduction:.6f} |'.format(**r))
    lines += ['',
      '## Interpretation and limits','',
      '**The current results do not validate strong forgetting.** In both datasets, every method has higher forget mR after explicit FU than its exact no-FU control, and its policy-matched retraining KL is also higher. Proposed changes forget mR from16.842% to17.301% on CapERA and from11.220% to11.546% on AirScape. '+paired_statement+' A low absolute KL threshold alone therefore does not demonstrate a benefit over simply excluding requested samples from subsequent FL.',
      'Proposed has only a very small average association-margin reduction on CapERA and a small increase on AirScape. Read the conditional intervals and all-video outcomes in22 rather than claiming uniformly weakened associations. The mean raw positive-pair cosine increases from warm20 on CapERA and decreases on AirScape;26 is descriptive because intervening FL also changes representations.',
      'Wireless vs exact trajectories have very similar forget mR and retraining KL in these runs. That observation points to the current learning/unlearning objectives and reference definition as the substantive issue, rather than establishing that AirComp distortion explains the remaining forgetting gap. No causal guarantee outside these runs is implied.',
      'Do not infer successful unlearning from retained mR, completed FU steps, a small aggregation MSE, or C9. C9 compares distorted and error-free aggregation; it does not bound distance to a deletion-trained reference. Figures21/25 test that distance empirically, while22/23 test the FU intervention at fixed exact transport.',
      'Retraining means resetting and retraining the rank10 visual/text adapters from their original initialization, using the frozen pretrained RemoteCLIP encoders and fixed temperature. It does not mean retraining the full pretrained VLM. The complete deletion union is excluded from every reference round. All six fresh200-round references reproduce their previously saved counterparts bitwise.',
      'The common reference preserves the original sampler. The policy-matched reference changes the retained-only sampling/weighting counterfactual for every method; it never changes a candidate model or feeds evaluation predictions into training. Publish both: agreement with a convenient reference alone is insufficient.',
      'The Proposed training proxy is the existing20-round retained-only proxy with the full future deletion cohort known. Its local request KL gate is not the200-round full-gallery reporting KL. That assumption remains unchanged and should be disclosed; these experiments do not validate unknown future deletion cohorts.',
      'The16 fresh control runs start from the same warm20 checkpoint as the original models. Requested samples leave subsequent FL at the original release times. Exact-no-FU does zero explicit unlearning updates; its request-step counters only preserve the original schedule. It must not be reported as successfully processing deletion requests. All original digital FU controls reproduce the original final parameter tensors bitwise. AFL-OTA has all10 ready and zero staleness in these particular traces.',
      'No energy or delay claim is made for exact-aggregation controls. They isolate FU conditional on the original schedule; comparisons with actual wireless models additionally change communication error. Request before/after intervals can include FL updates; both conditions are exported.',
      'Retrieval can remain above zero after valid retraining because of pretrained/generalizable knowledge. Excess retrieval and probability divergence relative to retraining are more informative than demanding zero recall. Conversely, suppressing all similarities can destroy utility without removing the learned influence of specific samples.',
      'All forget queries are included. No favorable-subset selection, modified baseline, evaluation-driven tuning, or withheld adverse result. Frozen feature caches and raw data remain local for audit; there is no claim of physical data erasure or certified unlearning.','',
      '## Reproduction','',
      'Run `tools/uav_convergence_v2/forgetting_quality_v1.py run` with the local native data/features and GPU, then `tools/uav_convergence_v2/forgetting_quality_report_v1.py`. Completed runs are reused; input hashes are checked. Raw features, models and logs are intentionally not published.',
      'Detailed per-video metrics, all request events, complete summaries, plot values and provenance accompany the figures in `data/`.','',
      'Background: [CLIPErase (ACL2025)](https://aclanthology.org/2025.acl-long.1469/) motivates explicitly assessing visual–textual associations, retention and consistency. [SalUn (ICLR2024)](https://openreview.net/pdf?id=gn0mIhQGNM) is the source of the existing F-SalUn transfer; no method has been modified for this audit.','']
    (DEST/'README.md').write_text('\n'.join(lines))
    captions=[
      ('20_forget_retrieval_1x2','Forget-set retrieval performance of the final FL/FU models and policy-matched models retrained without the requested data.'),
      ('21_forget_retraining_kl_1x2','Forget-set prediction divergence from common and policy-matched retraining references.'),
      ('22_forget_association_change_1x2','Change in deleted video--text associations induced by FU under exact aggregation. Positive values indicate a reduction relative to the matched no-FU control. Error bars show conditional 95\\% video-bootstrap intervals.'),
      ('23_forget_retrieval_fu_control_1x2','Forget-set retrieval performance with and without FU under exact aggregation at the same 200-round learning horizon.'),
      ('24_wireless_error_forgetting_1x2','Forget-set prediction divergence from the common retraining reference under exact and wireless aggregation.'),
      ('25_request_retraining_kl_1x2','Distribution of prediction divergence from policy-matched retraining over all deletion requests. The dashed line indicates the diagnostic tolerance.'),
      ('26_forget_association_before_after_1x2','Mean cosine similarity of deleted video--text pairs before the deletion workload and after joint FL/FU.')]
    captions.append(('27_forget_kl_fu_control_1x2','Forget-set divergence from policy-matched retraining with and without FU under exact aggregation at the same 200-round learning horizon.'))
    tex=['% Fresh seed0 audit; exact controls do not imply physical transmission feasibility.']
    for name,caption in captions:
        tex += ['\\begin{figure}[!t]','    \\centering',f'    \\includegraphics[width=\\linewidth]{{figures/{name}.pdf}}',
                '    \\caption{'+caption+'}',f'    \\label{{fig:{name}}}','\\end{figure}','']
    (DEST/'figure_snippets.tex').write_text('\n'.join(tex))
    return rows


def main():
    global COLLECTION
    DEST.mkdir(parents=True,exist_ok=True);(DEST/'data').mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'serif','font.serif':['Nimbus Roman','Times New Roman','DejaVu Serif'],
        'mathtext.fontset':'stix','font.size':12,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
        'savefig.bbox':None,'hatch.linewidth':.55,'axes.unicode_minus':True})
    protocol=study.register()
    for name,digest in protocol['input_sha256'].items():assert sha(ROOT/name)==digest,name
    collect()
    COLLECTION=PdfPages(DEST/'forgetting_quality_all.pdf')
    pair('20_forget_retrieval_1x2','Forget-set mR (%)',['FL + FU','Retrained'],
         lambda d,m:[metric(d,m,'actual','forget','mR'),reference(d,m,'mR')],100)
    pair('21_forget_retraining_kl_1x2',r'Forget-set KL ($10^{-3}$)',['Common reference','Policy-matched reference'],
         lambda d,m:[metric(d,m,'actual','forget','common_KL'),metric(d,m,'actual','forget','matched_KL')],1000)
    assoc=association()
    pair('23_forget_retrieval_fu_control_1x2','Forget-set mR (%)',['Without FU','With FU'],
         lambda d,m:[metric(d,m,'exact_no_fu','forget','mR'),metric(d,m,'exact_fu','forget','mR')],100)
    pair('24_wireless_error_forgetting_1x2',r'Forget-set KL ($10^{-3}$)',['Exact aggregation','Wireless aggregation'],
         lambda d,m:[metric(d,m,'exact_fu','forget','common_KL'),metric(d,m,'actual','forget','common_KL')],1000)
    request_box()
    pair('26_forget_association_before_after_1x2','Positive-pair cosine',['Before requests','After FL + FU'],
         lambda d,m:[read(OUT/d.lower()/'warm20/metrics.json')['forget']['positive_cosine'],metric(d,m,'actual','forget','positive_cosine')])
    pair('27_forget_kl_fu_control_1x2',r'Forget-set KL ($10^{-3}$)',['Without FU','With FU'],
         lambda d,m:[metric(d,m,'exact_no_fu','forget','matched_KL'),metric(d,m,'exact_fu','forget','matched_KL')],1000)
    COLLECTION.close();COLLECTION=None
    rows=docs(assoc)
    table(DEST/'data/plot_values.csv',PLOT_ROWS);write(DEST/'data/layout_checks.json',dict(passed=True,figures=AUDITS))
    write(DEST/'data/protocol.json',protocol)
    inherited=study.quality.register()['source_sha256']
    write(DEST/'data/provenance.json',dict(training_source_sha256=sha(Path(study.__file__)),
          report_source_sha256=sha(Path(__file__)),original_inputs_unchanged=True,
          inherited_training_sources_verified=inherited,
          completed=[read(OUT/d.lower()/'finished.json') for d in DATASETS]))
    aliases=ROOT/'final_final_charts/figures';aliases.mkdir(exist_ok=True)
    for file in DEST.glob('*.pdf'):shutil.copy2(file,aliases/file.name)
    print(json.dumps(dict(figures=len(AUDITS),destination=str(DEST),summary=rows),indent=2))


if __name__=='__main__':main()
