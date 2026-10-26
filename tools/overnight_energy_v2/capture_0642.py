"""Freeze completed PUF/dual/precision evidence and active follow-up states."""
import base64
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import subprocess
import time
import zipfile

root=Path('/home/wnlab/Documents/Codex/2026-10-20/files-pasted-by-the-user-standalone/outputs/ova_CK')
results=root/'results/overnight_energy_v2';tools=root/'tools/overnight_energy_v2';suite=root/'results/comparisons_seed0_solver_v2'
def read(p):
    for attempt in range(10):
        try:return json.loads(p.read_text())
        except json.JSONDecodeError:
            if attempt==9:raise
            time.sleep(.02)
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
expected=read(suite/'suite_metadata.json')['source_sha256'];assert all(sha(root/n)==v for n,v in expected.items())
quality=read(root/'results/energy_quality_v1/energy_quality_report.json')
puf=next(x for x in quality['methods'] if x['method']=='puf_regular_adapted_hybrid')
assert puf['status']=='completed' and puf['round']==100 and puf['quality_status']=='evaluated'
assert read(results/'dual_analog_v9/status.json')['head_designs']==140
assert read(results/'dual_analog_v9/physical_certificates.json')['observed_head_designs']==140
assert read(results/'u_precision_v12/status.json')['new_variants']==8
now=datetime.now().isoformat(timespec='seconds')
retain_path=suite/'retain_only_ideal/rounds.jsonl'
retain_rows=[json.loads(x) for x in retain_path.read_text().splitlines() if x.strip()] if retain_path.exists() else []
radio_heads=read(results/'dual_analog_v9/heads.json');dual={}
for relative in [.2,.6]:
    eligible=[x for x in radio_heads if x['relative_tolerance']==relative and x['status']=='feasible_radio_only']
    dual[str(relative)]=min(eligible,key=lambda x:x['energy_bound_j'])
initializer=read(results/'initialization_v11/results.json')
initializer_status=read(results/'initialization_v11/status.json')
snapshot=dict(captured_at_kst=now,source_sha256=expected,puf=puf,
              puf_protocol_note='Frozen adapted baseline omits retained-learning on requesting clients and uses20*forget_count/modified_denominator for ascent. Thus each requester has U but zero L in request rounds, explaining near equality of energy bound and expected random-codeword energy. This is an algorithmic baseline, not a matched-fixed-update radio ablation.',
              dual_completed_status=read(results/'dual_analog_v9/status.json'),dual_best=dual,
              dual_physical_certificates=read(results/'dual_analog_v9/physical_certificates.json'),
              precision_summary=read(results/'u_precision_v12/summary.json'),
              precision_rows=read(results/'u_precision_v12/results.json'),
              recent_projection_screen=read(results/'public_history_v13/recent_screen/results.json'),
              projection_radio_status=read(results/'public_history_v13/radio/status.json'),
              initialization_status=initializer_status,initialization_completed_cases=len(initializer),
              initialization_selected=[{k:x.get(k) for k in ['bits','relative_tolerance','mode','analog_fraction','status','selected_solver','selected_energy_bound_j']} for x in initializer],
              observed_frontier=quality['observed_frontier'],selection_status=quality['selection_status'],
              retain_completed_rounds=len(retain_rows),retain_last=retain_rows[-1] if retain_rows else None,
              recovery_status=read(results/'queue_recovery_v1/status.json'),
              gpu_processes=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True).stdout,
              processes=subprocess.run(['ps','-p','3546625,3546626,3520581,3534100,3597714,3591060,3599136','-o','pid,ppid,stat,etime,args'],capture_output=True,text=True).stdout)
write(results/'heartbeat_0642_snapshot.json',snapshot)
state=read(tools/'campaign_state.json');state.update(last_heartbeat_research_kst=now,
    puf_completed=True,puf_final_quality_evaluated=True,current_gpu_method='retain_only_ideal',current_gpu_pid=3597714,
    intentionally_held_idle_pids=[],dual_analog_v9_completed=True,u_precision_v12_completed=True,
    public_history_v13_observer_completed=True,initialization_v11_completed=initializer_status['status'].startswith('completed'),
    last_completed_evidence_snapshot='results/overnight_energy_v2/heartbeat_0642_snapshot.json',
    notes='PUF completed100 and cached final-reference audit finished07:00:21; existing retain-only is now the single GPU training job, burst waits it. Prior intentional holds released; do not stop/relaunch them. Dual v9 full140/126 feasible checked; v12 sees tiny one-step precision effects and does not justify tightening U QoS. v13 recent-public-history paid-coefficient radio and v11 initialization refinement continue CPU only, no promotion.')
write(tools/'campaign_state.json',state)
with (tools/'campaign_plan.txt').open('a') as f:
    f.write('\nHeartbeat06:42 snapshot '+now+' KST\n'
        'PUF completed100 and all5 request rounds, final reference audit finished07:00:21. Ebound0.200245690 J, val loss0.96776219/acc72.9%, retainedloss0.94620692/acc74.4%, forgetKL0.02616829, pred disagreement5.4%, val loss-MIA AUC0.503364/gap0.001192. Mixed remains lower E/val loss/forgetKL in this seed; no formal all-metric/statistical winner. PUF requesters skip L and have different FU weighting, so algorithmic rather than fixed-update radio comparison. Recovery resumed original waiters; retain_only_ideal GPU3597714 now active, burst waits. No duplicate cache inference.\n'
        'Dual v9 complete140 head designs/14cases at06:53:29:126 feasible designs pass independent physical-unit checks,14 no feasible design found. Best0.2 9.034770mJ/head0/f0.125; best0.6 0.581236mJ/head3/f0.125. Finite AO incumbents; all failures retained.\n'
        'U precision v12 complete8 new vectors at06:57:15 plus reused ideal: all validation73.5%, forget71.5%, forget-reference disagreement4%. Codec8/6 U-only referenceKL0.02498927/0.02498971; analogU-only mean0.02498727, fullDual mean0.02498387, three noise draws, ideal0.02498938. Differences tiny and not proof of useful precision/deletion; oracleL controls never physical energy competitors. This DOES NOT establish an application need for extra strict U QoS and does not justify excluding the original Mixed/dual results. No new MIA from tiny pilot, no automatic GPU promotion.\n'
        'v13 public-state observer completed five states96..100 without read errors. Actual round100 learning-only CPU replay at public theta99 and96..99 history completed07:02:50: rank1 normsum reduction1.255%, rank3 4.886%, coefficients288/864 bits per head; exact reconstruction~3.4e-22. Paid-control radio CPU3599136 still runs with all10 heads and SAME167.936ms total, control RFenergy/peak/battery paid before any saving claim. All original modes identical here because U=0; generic FL improvement only, not Hybrid superiority. Decode/projection compute energy unknown; report break-even. Preserve failed long-history controls.\n'
        'Initialization v11 still partial unless its status now completed; inspect before duplicating. Current completed H8 strict/loose rows unchanged from v3 despite always-convex start andAO30. No need to alter frozen training/burst sources. Morning09 report flag still false; noon policy unchanged.\n')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(12,4.6))
names=['coupled_mixed_aircomp','coupled_hybrid_q6','puf_regular_adapted_hybrid','coupled_hybrid','coupled_fixed_head']
rows={x['method']:x for x in quality['methods']};values=[rows[n]['energy_bound_j'] for n in names]
bars=axes[0].bar(['Mixed','Hybrid6','PUF-adapted8','Hybrid8','Fixed head8'],values,color=['#427c96','#5b946e','#9972ad','#b05a35','#9b9b9b'])
for bar,value in zip(bars,values):axes[0].text(bar.get_x()+bar.get_width()/2,value+.006,f'{value:.4f}',ha='center',fontsize=9)
axes[0].set_ylim(0,.4);axes[0].set_ylabel('Communication energy upper bound (J)')
axes[0].set_title('Completed100-round physical runs (one seed)')
summary=snapshot['precision_summary'];ideal=snapshot['precision_rows'][0]['forget']['gap_to_final_retraining']['kl_reference_to_candidate']
changes=[(x['forget_kl_to_final_retraining_mean']-ideal)*1e6 for x in summary]
spread=[x['forget_kl_to_final_retraining_noise_std']*1e6 for x in summary]
axes[1].errorbar(range(4),changes,yerr=spread,fmt='o',capsize=5,color='#427c96')
axes[1].axhline(0,color='#999999',linewidth=1)
axes[1].set_xticks(range(4),['Codec8\n(ideal L)','Codec6\n(ideal L)','Analog U\n(ideal L)','Full dual\nanalog'])
axes[1].set_ylabel('Forget KL change from ideal step (x1e-6)')
axes[1].set_title('U precision: tiny effects in one fixed step')
for ax in axes:ax.spines[['top','right']].set_visible(False)
fig.text(.5,.015,'Right:200 frozen forget samples; bars are noise SD across3 draws, not training seeds or CI. Ideal-L controls have no physical energy rank. No deletion guarantee.',ha='center',fontsize=7.5)
fig.tight_layout(rect=(0,.045,1,1))
for ext in ['png','svg']:fig.savefig(results/f'heartbeat_0642_quality_energy.{ext}',dpi=170)
plt.close(fig)

paths=[results/'heartbeat_0642_snapshot.json',results/'heartbeat_0642_quality_energy.png',results/'heartbeat_0642_quality_energy.svg',
       tools/'campaign_plan.txt',tools/'campaign_state.json',tools/'capture_0642.py',
       root/'results/energy_quality_v1/energy_quality_report.json',root/'results/energy_quality_v1/energy_quality.csv',
       root/'results/energy_quality_v1/puf_regular_adapted_hybrid.cache.json',
       suite/'puf_regular_adapted_hybrid/latest.pt',suite/'puf_regular_adapted_hybrid/rounds.jsonl',suite/'puf_regular_adapted_hybrid/manifest.json',
       results/'queue_recovery_v1/status.json',results/'retain_control_status.json',results/'retain_control_plan.json']
if retain_path.exists():paths.extend([retain_path,retain_path.parent/'manifest.json'])
for name in ['dual_analog_v9','initialization_v11','u_precision_v12','public_history_v13']:
    for parent in [results,tools]:paths.extend(p for p in (parent/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts and not p.name.endswith('.tmp'))
contents={};omitted=[]
for p in dict.fromkeys(paths):
    for attempt in range(5):
        try:
            data=p.read_bytes()
            if p.suffix=='.json':json.loads(data)
            if p.suffix=='.npz':
                with zipfile.ZipFile(io.BytesIO(data)) as candidate:assert candidate.testzip() is None
            contents[str(p.relative_to(root))]=data;break
        except (OSError,json.JSONDecodeError,zipfile.BadZipFile,AssertionError):time.sleep(.02)
    else:omitted.append(str(p.relative_to(root)))
manifest=dict(captured_at_kst=now,ongoing_files_are_partial=True,in_progress_files_omitted=omitted,files={n:dict(sha256=hashlib.sha256(v).hexdigest(),bytes=len(v)) for n,v in contents.items()})
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as archive:
    for name,data in contents.items():archive.writestr(name,data)
    archive.writestr('capture_manifest.json',json.dumps(manifest,indent=2)+'\n')
print(base64.b64encode(buf.getvalue()).decode())
