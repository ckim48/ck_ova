"""Freeze a partial radio snapshot and completed component-fidelity diagnostics."""
import base64
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import zipfile

root=Path('/home/wnlab/Documents/Codex/2026-10-20/files-pasted-by-the-user-standalone/outputs/ova_CK')
results=root/'results/overnight_energy_v2';tools=root/'tools/overnight_energy_v2'
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
expected=read(root/'results/comparisons_seed0_solver_v2/suite_metadata.json')['source_sha256']
assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==v for n,v in expected.items())
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
verified=subprocess.run([str(root/'.venv/bin/python'),str(tools/'dual_analog_v9/verify_physical.py'),'--project',str(root)],env=env,capture_output=True,text=True)
assert verified.returncode==0,verified.stderr
heads=read(results/'dual_analog_v9/heads.json');oracle=read(results/'component_fidelity_v10/oracle_results.json')
best={}
for relative in [.2,.6]:
    good=[x for x in heads if x['relative_tolerance']==relative and x['status']=='feasible_radio_only']
    if good:best[str(relative)]=min(good,key=lambda x:x['energy_bound_j'])
puf_dir=root/'results/comparisons_seed0_solver_v2/puf_regular_adapted_hybrid'
puf=[json.loads(x) for x in (puf_dir/'rounds.jsonl').read_text().splitlines() if x.strip()]
state=read(tools/'campaign_state.json');now=datetime.now().isoformat(timespec='seconds')
state.update(last_heartbeat_research_kst=now,component_fidelity_v10_completed=True,
             component_fidelity_v10_tests_passed=6,component_fidelity_v10_gpu_promoted=False,
             last_radio_snapshot='results/overnight_energy_v2/heartbeat_0612_snapshot.json')
write(tools/'campaign_state.json',state)
snapshot=dict(captured_at_kst=now,core_sha256=expected,
              dual_analog_status=read(results/'dual_analog_v9/status.json'),
              dual_analog_completed_head_designs=len(heads),dual_analog_best_so_far=best,
              dual_analog_physical_verification=json.loads(verified.stdout),
              oracle_summary=[{k:v for k,v in x.items() if k!='blocks'} for x in oracle['heads']],
              puf_completed_rounds=len(puf),puf_last_round=puf[-1],
              puf_request_rounds=[x for x in puf if x['newly_deleted']],
              gpu_processes=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True).stdout,
              note='Dual analog grid remains active. Existing full-trajectory queue unchanged. Component-U precision is an explicit problem revision with no application need or deletion guarantee established. No architecture or task-qualified energy winner is declared.')
write(results/'heartbeat_0612_snapshot.json',snapshot)
with (tools/'campaign_plan.txt').open('a') as f:
    f.write('\nHeartbeat06:12 diagnostic snapshot '+now+' KST\n'
        'Dual-analog v9 CPU3587649 still active: '+str(len(heads))+'/140 head designs. At original sum tolerance0.2, feasible incumbent9.03477mJ (head0,timefraction0.125; full grid pending) vs existing joint H8 55.01878mJ, phase8 32.82269mJ, Mixed5.05909mJ. More AO work on new strong baseline; finite incumbents, not global energy optima. Independent physical-unit verifier checks saved designs. U actual RMS at this dual incumbent1.21796e-4, not equal to digital U fidelity. No task-quality inference/new GPU.\n'
        'Component_fidelity_v10 completed6 tests plus all10-head calculation. Separate-U targets set to existing codec global conservative bounds tau6=5.539354e-5,tau8=1.352128e-5 (NOT actual quant errors). Optimistic minimum conservative analog bound exceeds both at all heads. A second offline full-geometry oracle relaxes learning quality/battery, grants maxU time, signed gains and full true updates; convex primal and explicit nonnegative quadratic-dual numerical certificate agree in aggregate. Best numerical oracle lower U RMS5.765073e-5, about1.04075*tau6 and4.26370*tau8. Singular padded-block geometry falls back to a valid zero lower bound; some individual relative block gaps are1, so do not hide them or claim all-block tightness. These are results only for a separately recovered single-linear-receiver U analog class with current time/power/update fixture, not all analog communication. No hardware measurement, interval proof, application-need validation or forgetting guarantee. Original additive problem does NOT contain separate-U tau; keep its Mixed/dual results visible. Optimized digital-phase also meets codec fidelity, so this is NOT NOMA-over-OMA energy evidence.\n'
        'Keep current PUF/retain/burst queue intact, no duplicate inference/training. Next heartbeat should read dual v9 status and rerun only verify_physical.py for newly completed incumbents; do not rerun completed tests/oracle or edit frozen v9 run/solver. Full application controls still pending before promotion.\n')

# Standard static scientific plot. Values come from this recorded fixed state.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(12,4.7))
reference=read(results/'joint_power_v3/results.json')
energies=[]
for mode in ['hybrid','orthogonal','mixed_aircomp']:
    candidates=[x for x in reference if x['bits']==8 and x['relative_tolerance']==.2 and x['mode']==mode and x['status']=='feasible']
    energies.append(min(candidates,key=lambda x:x['energy_bound_j'])['energy_bound_j']*1000)
energies.append(best['0.2']['energy_bound_j']*1000)
labels=['Hybrid8\n(joint AO)','Digital-phase8\n(best grid)','Mixed\nAirComp','Dual analog\n(partial grid)']
bars=axes[0].bar(labels,energies,color=['#b05a35','#8d79b0','#427c96','#5b946e'],width=.63)
for bar,value in zip(bars,energies):axes[0].text(bar.get_x()+bar.get_width()/2,value+1,f'{value:.2f}',ha='center',fontsize=10)
axes[0].set_ylim(0,max(energies)*1.19);axes[0].set_ylabel('Modeled communication energy (mJ)')
axes[0].set_title('Original additive sum: same RMS budget')
axes[0].spines[['top','right']].set_visible(False)
oracle_rows=oracle['heads'];values=[x['numerical_dual_lower_u_rmse']*1e6 for x in oracle_rows]
axes[1].bar(range(len(values)),values,color='#5b946e',alpha=.8,label='Optimistic oracle numerical lower')
codecs=read(results/'component_fidelity_v10/results.json')['reference_codecs']
for bits,color in [('6','#b05a35'),('8','#427c96')]:
    tau=codecs[bits]['global_all_client_quantization_rms_bound']*1e6
    axes[1].axhline(tau,color=color,linestyle='--',label=f'{bits}-bit global codec bound ({tau:.2f})')
axes[1].set_xticks(range(10));axes[1].set_xlabel('Candidate head');axes[1].set_ylabel('Separate U RMS error (micro-units)')
axes[1].set_title('Extra U-fidelity constraint: changed problem')
axes[1].legend(loc='upper left',bbox_to_anchor=(0,1.0),fontsize=8,frameon=False)
axes[1].set_ylim(0,max(values)*1.4);axes[1].spines[['top','right']].set_visible(False)
fig.text(.5,.02,'CPU fixed legacy round91; finite AO/grid. Oracle grants full gradient geometry and relaxes learning quality. No VLM/deletion guarantee.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.055,1,1))
for ext in ['png','svg']:fig.savefig(results/f'heartbeat_0612_energy_precision.{ext}',dpi=170)
plt.close(fig)

paths=[results/'heartbeat_0612_snapshot.json',results/'heartbeat_0612_energy_precision.png',results/'heartbeat_0612_energy_precision.svg',
       tools/'campaign_state.json',tools/'campaign_plan.txt',tools/'capture_0612.py',puf_dir/'rounds.jsonl',puf_dir/'manifest.json',
       root/'results/energy_quality_v1/energy_quality_report.json',root/'results/energy_quality_v1/energy_quality.csv',
       results/'joint_power_v3/results.json',results/'joint_power_v3/manifest.json']
for name in ['dual_analog_v9','component_fidelity_v10']:
    for parent in [results,tools]:paths.extend(p for p in (parent/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
# Freeze file bytes before hashing and zipping, since the diagnostic is active.
contents={str(p.relative_to(root)):p.read_bytes() for p in paths}
manifest=dict(captured_at_kst=now,ongoing_results_are_partial=True,
              files={name:dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest()) for name,data in contents.items()})
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as archive:
    for name,data in contents.items():archive.writestr(name,data)
    archive.writestr('capture_manifest.json',json.dumps(manifest,indent=2)+'\n')
print(base64.b64encode(buf.getvalue()).decode())
