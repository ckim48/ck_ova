"""Four requested primary axes; keep raw deletion diagnostics separate."""
import json,time
from prepare_campaign import OUT,ARMS,LABELS,THRESHOLD,write

def scalar_point(row):
    ev=row.get('evaluation') or {}
    r=ev.get('retained',{})
    return dict(FL_round=row['completed_FL_rounds'],mR=r.get('mR'),loss=r.get('loss'),
       TX_energy_bound_j=row['cumulative_tx_energy_bound_j'],
       uplink_occupied_s=row['cumulative_uplink_occupied_s'],
       elapsed_service_s=row['cumulative_elapsed_service_s'],
       FU_procedures_completed=row['requests']['procedure_completed'])

def summarize(name):
    dest=OUT/name;row=dict(method=name,label=LABELS[name],status='pending')
    samples=[]
    for p in sorted((dest/'evaluations').glob('*.json')):
        samples.append(scalar_point(json.loads(p.read_text())))
    row['curve']=samples
    regular={p['FL_round']:p for p in samples if p['FL_round']%5==0}
    streak=[];target=None;first=None
    for p in regular.values():
        ok=p['FU_procedures_completed']==45 and p['mR'] is not None and p['mR']>=THRESHOLD
        if ok and first is None:first=p
        streak=streak+[p] if ok else []
        if len(streak)==3 and target is None:target=dict(first_of_three=streak[0],confirmation=streak[-1])
    row['utility_first_crossing']=first;row['utility_confirmed_target']=target
    m=dest/'manifest.json'
    latest=dest/'latest.json'
    if m.exists():
        manifest=json.loads(m.read_text());row['status']=manifest['status']
        row['final']=scalar_point(manifest)
        row['completed_same_workload']=bool(manifest['status']=='completed_200FL_and_requests' and
            row['final']['FL_round']==200 and row['final']['FU_procedures_completed']==45)
    elif latest.exists():
        state=json.loads(latest.read_text());row['status']='running';row['progress']=scalar_point(state)
    failure=dest/'failure.json'
    if failure.exists():
        row['status']='failed';row['failure']=json.loads(failure.read_text());row['completed_same_workload']=False
    timing=dest/'runtime_seconds.json'
    if timing.exists():row['server_wall_clock']=json.loads(timing.read_text())
    return row

def report():
    rows=[summarize(n) for n in ARMS]
    write('summary.json',dict(updated_unix=time.time(),methods=rows,
        utility_target_mR=THRESHOLD,
        main_energy_scope='Cumulative TX upper bound; not flight/compute/RX energy.',
        main_time_scope='Pure occupied uplink; service clock additionally includes idle waiting. Server runtime is separate, not UAV end-to-end latency.',
        scope='Four complete systems; utility convergence is not deletion certification. Unfinished methods are not given completion-cost ranks.'))

if __name__=='__main__':report()
