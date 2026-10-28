"""Publish measured learning diagnostics and a fourth literature comparator."""
import argparse
import html
import math
from pathlib import Path
import time
import numpy as np
import learning_quality_v3 as quality
import workload_charts as plots
import learning_selection_fix_v2 as correction

ROOT=quality.ROOT
prior=plots.prior
ORIGINAL_SOURCE=plots.source
ORIGINAL_METHODS=dict(plots.study.METHODS)
SALUN='f_salun'
METHODS={**ORIGINAL_METHODS,SALUN:'F-SalUn'}
read,write,sha=quality.read,quality.write,quality.sha


def source(case,ds,method):
 if method==SALUN:return quality.OUT/'salun/cases'/case/(ds.lower()+'_seed0')/quality.DIGITAL
 return ORIGINAL_SOURCE(case,ds,method)


def variant_source(ds,variant):
 origin=correction.OUT if quality.VARIANTS[variant]['selection'] else quality.OUT
 return origin/variant/'cases/baseline'/(ds.lower()+'_seed0')/(quality.DIGITAL if variant=='salun' else quality.PROPOSED)


def styles(methods):
 palette={quality.PROPOSED:('#4C72B0','o','-'),'control':('#4C72B0','o','-'),'current':('#4C72B0','o','-'),
  'iscc_p2_gaft':('#EFAA54','s','--'),'iscc':('#EFAA54','s','--'),
  'uav_afl_ota_gaft':('#66B366','v','-.'),'afl':('#66B366','v','-.'),
  'salun':('#C65365','^',':'),SALUN:('#C65365','^',':'),
  'select16':('#9467BD','s','--'),'gate_select16':('#92703B','d',(0,(5,2)))}
 chosen=[palette.get(m,('#777777','o','-')) for m in methods]
 prior.chart.COLORS=prior.chart.style.COLORS=[v[0] for v in chosen]
 prior.chart.MARKERS=prior.chart.style.MARKERS=[v[1] for v in chosen]
 prior.chart.STYLES=prior.chart.style.STYLES=[v[2] for v in chosen]


def setup(dest,methods,resolve):
 dest.mkdir(parents=True,exist_ok=True);(dest/'data').mkdir(exist_ok=True)
 prior.DEST=dest;prior.METHODS=methods;prior.source=resolve;prior.comm=plots.comm
 prior.SOURCES={};prior.SERIES=[];prior.FIGURES=[]
 prior.configure()
 styles(methods)
 # Preserve axes assertions and the established canvas; only fit the legend.
 if not hasattr(setup,'save'):setup.save=prior.save
 prior.save=quality.infra.cached_clone(setup.save,[('ncol=3','ncol=len(METHODS)',1),('fontsize=12,handlelength','fontsize=10,handlelength',1)],{},live_globals=prior.__dict__)


def finish(dest,rows,description,extra=None):
 write(dest/'data/metrics.json',rows);prior.write_csv(dest/'data/metrics.csv',rows)
 write(dest/'data/plot_series.json',prior.SERIES)
 for file in [Path(__file__),Path(quality.__file__),Path(correction.__file__),Path(prior.__file__),Path(plots.__file__)]:prior.SOURCES[str(file.relative_to(ROOT))]=sha(file)
 manifest=dict(complete=True,unix=time.time(),methods=prior.METHODS,figures=prior.FIGURES,
  source_sha256=prior.SOURCES,raw_data_published=False,description=description)
 if extra:manifest.update(extra)
 write(dest/'data/manifest.json',manifest)
 cards=['<section><h2>'+html.escape(f['title'])+'</h2><a href="'+f['name']+'.pdf"><img loading="lazy" src="'+f['name']+'.png"></a></section>' for f in prior.FIGURES]
 (dest/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>UAV VLM learning comparisons</title><style>body{font:16px system-ui;max-width:1150px;margin:35px auto;padding:0 20px;color:#202530}img{width:100%}h2{font-size:20px}section{margin:45px 0}</style><h1>UAV/VLM learning comparisons</h1><p>'+html.escape(description)+' <a href="README.md">Definitions and limitations</a> · <a href="data/metrics.csv">All metrics</a> · <a href="../index.html">Previous figures</a></p>'+''.join(cards))


def four_methods():
 quality.register();correction.register()
 dest=ROOT/'final_final_charts/four_methods'
 setup(dest,METHODS,source);plots.study.METHODS=METHODS;plots.source=source
 rows=[];audit=[];cache={}
 def digest(p):
  key=str(p.resolve())
  if key not in cache:cache[key]=sha(p)
  return cache[key]
 for case in plots.cases():
  for ds in quality.infra.DATASETS:
   reference=source(case,ds,quality.PROPOSED)
   for m in METHODS:
    arm=source(case,ds,m)
    for name in ['plan.json','requests.json','initial20/final.pt','feature_cache/features.pt','reference200/final.pt']:
     assert digest(arm.parent/name)==digest(reference.parent/name),(case,ds,m,name)
    assert read(arm.parents[1]/'settings.json')['energy_model']==read(reference.parents[1]/'settings.json')['energy_model']
    row=plots.ledger(case,ds,m);rows.append(row)
    if m==SALUN:
     tx=[read(p) for p in sorted((arm/'transactions').glob('*.json'))];masks={};steps={};charges=0
     for t in tx:
      assert t['radio']['max_power_w']<=.32*(1+2e-6)
      assert t['radio']['conditional_rmse']<=t['radio']['epsilon']*(1+3e-5)+1e-9
      for local in t['local']:
       rid=local['request'];steps[rid]=steps.get(rid,0)+1
       assert local['FU_variant']=='F-SalUn' and local['step']==steps[rid]
       assert local['random_label_shift']>0 and .49<=local['mask_fraction']<=.51
       assert local['mask_created']==(local['step']==1)
       if rid in masks:assert masks[rid]==local['mask_sha256']
       masks[rid]=local['mask_sha256'];charges+=local['extra_loss_MAC_equivalents']
     assert len(masks)==45 and sum(steps.values())==225 and set(steps.values())=={5} and charges>0
    audit.append(dict(case=case,dataset=ds,method=m,shared_inputs=True,complete_workload=True,source=str(arm.relative_to(ROOT))))
 assert len(rows)==168
 for i,a in enumerate(plots.AXES):
  plots.draw(rows,a,'total_energy_j','Total energy (kJ)',.001,f'{1+2*i:02d}_{a}_energy_1x2')
  plots.draw(rows,a,'delay_s','Delay (s)',1.,f'{2+2*i:02d}_{a}_delay_1x2')
 prior.convergence()
 plots.draw(rows,'fl_rounds','retained_mR_percent','Retained mR (%)',1.,'17_fl_rounds_mr_1x2')
 plots.draw(rows,'fl_rounds','forget_KL','Forget-set KL',1.,'18_fl_rounds_kl_1x2')
 for i,a in enumerate(plots.AXES):plots.draw(rows,a,'mission_delay_s','Mission delay (s)',1.,f'{19+i:02d}_{a}_mission_delay_1x2')
 write(dest/'data/audit.json',dict(passed=True,arms=len(rows),rows=audit))
 write(dest/'data/method_protocol.json',read(quality.OUT/'protocol.json'))
 write(dest/'data/responses.json',plots.response_summary(rows))
 text=['# Four-method UAV/VLM comparison','',
  'The original Proposed, ISCC-P2 and AFL-OTA results are unchanged. F-SalUn is added as a real FU-algorithm transfer; it is not a weakened version of Proposed. See [the existing transport and workload definitions](../expanded_sweeps/README.md).','',
  '**F-SalUn:** [SalUn, ICLR 2024, Algorithm 1](https://proceedings.iclr.cc/paper_files/paper/2024/file/ec4d2e436794d1bf55ca83f5ebb31887-Paper-Conference.pdf), transferred to federated rank10 VLM adapters with ISCC-P2 digital FDMA. At each request start, form a median hard mask from the forget-set contrastive gradient; keep that mask for five prescribed FU cycles. Wrong-label classification is adapted to randomly shifted video–text assignments, combined with retained CE. The backbone and temperature stay frozen. Equal-client FedSGD and dense FP32 transmissions remain unchanged. Saliency, backward passes and mask operations are charged. SalUn is an ICLR FU paper, not a UAV-FL paper; this is an explicit adaptation, not an official benchmark reproduction.','',
  'All four methods use identical data, initial weights, partitions, requests, routes and physical coefficients within each condition. All 45 requests and225 FU cycles finish. A digital QoS overrun is retained and reported; completion does not certify every original bound. The fifth sweep changes total FL budget, not the reporting reference200.','',
  'Total energy is active communication + computation + hovering. Fixed cached features imply adapter-only computation. The main delay is active FL/FU service time; mission delay is separate. No raw samples, checkpoints or transaction records are published.','',
  'Single development seed. SalUn has higher retained mR but also larger retrain KL in the baseline; its CapERA worst-request KL exceeds0.01. Mean KL below0.01 must not be mistaken for every-request compliance. All measured outcomes and original comparators remain visible. A separate [learning diagnosis and continuation bundle](../learning_quality/index.html) tests changes to Proposed; they are not silently substituted into these fixed-workload curves.','',
  '| Dataset | Method | mR (%) | Loss | Mean KL | Worst request KL | Energy (kJ) | Delay (s) |','|---|---|---:|---:|---:|---:|---:|---:|']
 for r in rows:
  if r['case']=='baseline':text.append(f"| {r['dataset']} | {r['label']} | {r['retained_mR_percent']:.3f} | {r['retained_loss']:.5f} | {r['forget_KL']:.6g} | {r['worst_request_KL']:.6g} | {r['total_energy_j']/1000:.3f} | {r['delay_s']:.3f} |")
 text+=['','Rebuild: `python tools/uav_convergence_v2/learning_quality_report.py four`. Requires local registered source experiments.','']
 (dest/'README.md').write_text('\n'.join(text));finish(dest,rows,'Four methods; five sensitivity axes; original Proposed preserved.',dict(compared_arms=len(rows)))
 print(dict(bundle='four_methods',figures=len(prior.FIGURES),arms=len(rows)),flush=True)


def diagnostics():
 dest=ROOT/'final_final_charts/learning_quality'
 all_methods={v:s['label'] for v,s in quality.VARIANTS.items()}
 resolve=lambda c,ds,v:variant_source(ds,v)
 setup(dest,all_methods,resolve)
 rows=[prior.ledger('baseline',ds,v) for ds in quality.infra.DATASETS for v in quality.VARIANTS]
 # Four readable curves; the table contains every diagnostic, including losses.
 prior.METHODS={'control':'Proposed','select16':'P-LS','gate_select16':'P-KS','salun':'F-SalUn'}
 styles(prior.METHODS)
 prior.convergence()
 write(dest/'data/protocol.json',read(quality.OUT/'protocol.json'))
 write(dest/'data/selection_correction.json',correction.register())
 text=['# Learning diagnosis and saved-time continuation','',
  'All ten predeclared diagnostics are reported below. The current Proposed is preserved; the KL-gate and selective-FL settings are developmental candidates. P-LS changes only FL sampling; P-KS adds the KL-gate objective. No validation or reporting-reference predictions select samples, gradients or weights. Single seed; this is diagnosis, not held-out statistical confirmation.','',
  'The exact-aggregation diagnostic removes received-update noise while retaining its original cost. It barely changes utility, whereas changing the FU objective recovers much of the retained performance. The old adaptive retained-CE coefficient was usually about0.5. Fixed1 and gated variants explicitly change that learning tradeoff.','',
  '**KL gate:** `3*relu(local_forget_proxy_KL-0.01) + retained_CE + 3*request_teacher_KL`. This avoids continuing to minimize a local proxy term below the declared tolerance. The training proxy and request teacher are the original disclosed assumptions. This local surrogate is not the final retrain KL and is not a deletion certificate.','',
  '**Selection:** half of each local batch explores uniformly; the remainder prioritizes high cached per-example training loss, scored only during earlier gradient computations. Deleted samples are removed as requests arrive. The fixed16 and explicit8 caps are separate experiments. Extra selection/scoring work is included. Smaller batches still send the same20480-coordinate model update, so reduced computation alone need not reduce total energy.','',
  'The original selection attempts and two setup corrections failed before the first transaction. They are preserved locally. The final correction only retains the factory’s seeded RNG call sites and binds its registered selection configuration; it does not change the sampling rule.','',
  '| Variant | Dataset | mR (%) | Loss | Mean KL | Worst KL | Energy (kJ) | Delay (s) |','|---|---|---:|---:|---:|---:|---:|---:|']
 for r in rows:text.append(f"| {r['label']} | {r['dataset']} | {r['retained_mR_percent']:.3f} | {r['retained_loss']:.5f} | {r['forget_KL']:.6g} | {r['worst_request_KL']:.6g} | {r['total_energy_j']/1000:.3f} | {r['delay_s']:.3f} |")
 text+=['','## Reproduction','',
  'Requires the original local feature caches and scenario inputs. Use `learning_quality_v3.py --variant control`, then `learning_quality_campaign.py control` and `learning_quality_unit.py`. Run the remaining non-selection variants with `learning_quality_runner.py run --variants exact equal gaft retain1 gate salun`. Run selection with `learning_quality_runner_v2.py run --corrected --variants select16 gate_select16 gate_select8`. Add the40 SalUn sensitivity arms with `learning_quality_runner.py run --variants salun --all-sweeps`. All named scripts are under `tools/uav_convergence_v2`.','',
  'Next run `learning_time_budget.py --dataset CapERA` and repeat for AirScape; run `learning_continuation_reference.py` for both datasets. Export with `learning_quality_report.py four`, `learning_quality_report.py diagnostics`, and `learning_time_report.py`, then verify using `learning_release_check.py`. The final corrected sampler is used without rerunning the intentionally preserved setup failures. Raw data, checkpoints and transaction logs remain local.','']
 (dest/'README.md').write_text('\n'.join(text)+'\n')
 finish(dest,rows,'All registered learning diagnostics; no outcome filtering.',dict(diagnostic_arms=len(rows)))
 print(dict(bundle='learning_quality',figures=len(prior.FIGURES),arms=len(rows)),flush=True)


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['four','diagnostics']);a=p.parse_args()
 (four_methods if a.action=='four' else diagnostics)()
