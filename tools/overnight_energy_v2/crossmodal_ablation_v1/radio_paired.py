"""Four-access comparison for nested round-61 multi-request workloads."""
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BASE_OUT = ROOT / "results/overnight_energy_v2/crossmodal_ablation_v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def oma_time_energy(bits, channel, symbols, config):
    """Continuous-symbol, per-block OMA energy minimum under peak power."""
    bits = np.asarray(bits, float)
    channel = np.asarray(channel, float)
    symbols = np.asarray(symbols, float)
    n, blocks = bits.shape
    W = config["symbol_rate_hz"]
    pmax = config["power_limit_w"]
    energy = np.zeros(n)
    used = np.zeros(blocks)
    required = np.zeros(blocks)
    for m in range(blocks):
        ix = np.flatnonzero(bits[:, m] > 0)
        if not len(ix):
            continue
        h2 = channel[ix] ** 2
        if np.any(h2 <= 0):
            return None
        b = bits[ix, m]
        cap = np.log2(1 + h2 * pmax)
        lmin = b / cap
        required[m] = lmin.sum()
        budget = symbols[m]
        if lmin.sum() > budget * (1 + 1e-10):
            return dict(status="peak_power_time_infeasible", min_symbols=float(lmin.sum()), budget_symbols=float(budget), block=m)

        def minus_derivative(L):
            x = math.log(2) * b / L
            return (1 - np.exp(x) * (1 - x)) / (W * h2)

        def lengths(mu):
            out = lmin.copy()
            for j in range(len(out)):
                if mu >= minus_derivative(lmin)[j]:
                    continue
                lo, hi = lmin[j], max(lmin[j] * 2, lmin[j] + 1)
                def val(x):
                    z = math.log(2) * b[j] / x
                    return (1 - math.exp(z) * (1 - z)) / (W * h2[j])
                while val(hi) > mu:
                    hi *= 2
                for _ in range(55):
                    mid = 0.5 * (lo + hi)
                    if val(mid) > mu:
                        lo = mid
                    else:
                        hi = mid
                out[j] = 0.5 * (lo + hi)
            return out

        high = max(minus_derivative(lmin))
        low = 0.0
        for _ in range(70):
            mid = 0.5 * (low + high)
            current = lengths(mid)
            if current.sum() > budget:
                low = mid
            else:
                high = mid
        L = lengths(high)
        p = np.expm1(np.log(2) * b / L) / h2
        if np.any(p > pmax * (1 + 1e-6)):
            return dict(status="peak_power_numerical_failure", max_power=float(p.max()))
        energy[ix] += L * p / W
        used[m] = L.sum()
    return dict(status="feasible", energy_by_user=energy, used_symbols=float(used.sum()),
                required_symbols=float(required.sum()), per_block_used=used.tolist(),
                per_block_minimum=required.tolist())


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["crossmodal", "vanilla"], required=True)
    args = parser.parse_args()
    INPUT = BASE_OUT / "prepared" / args.variant
    OUT = BASE_OUT / "radio" / args.variant
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(ROOT), str(ROOT / "tools/overnight_energy_v2"),
                   str(ROOT / "tools/overnight_energy_v2/initialization_v11")]
    from wireless_init_v11 import optimize_head, quantize
    from phase_split_grid import make_inputs, split_symbols, necessary_power

    baseline = json.loads((ROOT / "results/overnight_energy_v2/public_history_fu_v19/screen/plan.json").read_text())
    config0 = dict(baseline["config"], max_ao_iterations=30, receive_initializations=4)
    modes = ["hybrid_noma_aircomp", "aircomp_only_additive_reference",
             "aircomp_learning_plus_phase_separated_noma",
             "aircomp_learning_plus_orthogonal_unlearning", "oma_only_individual"]
    fractions8 = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]
    fractions4 = [0.25, 0.375, 0.5, 0.625, 0.75]
    summary = dict(status="running", cases=[], protocols={
        "hybrid_noma_aircomp": "learning AirComp and individual 8-bit deletion payloads via NOMA overlap in the same frame",
        "aircomp_only_additive_reference": "all local L+Q8(U) updates aggregated by AirComp; cannot implement the per-request controller, reported as task-ineligible reference",
        "aircomp_learning_plus_phase_separated_noma": "learning AirComp and deletion NOMA use separate phases; strong incumbent from the previous comparison",
        "aircomp_learning_plus_orthogonal_unlearning": "learning AirComp; each deletion packet sent in its own OMA slot, with optimized per-block time/power",
        "oma_only_individual": "each remote UAV sends its learning vector (16-bit if its error bound fits, otherwise exact float32) and Q8 deletion vector in an orthogonal digital packet",
        "limits": ["single round-61 counterfactual at frozen theta60; not a new training trajectory or forgetting guarantee",
                   "within each ablation variant, all radio methods use the same channel, head set, .32W peak, .1J battery, Q8 request updates, alpha=1 request-control rule, and absolute application-update RMSE budget where eligible",
                   "the vanilla variant reuses the crossmodal variant's absolute RMSE budget; its per-request controller target is recomputed from its own update vectors, so target values can differ as a consequence of removing coupling",
                   "NOMA and phase-grid OMA use the frozen joint analog/digital AO solver; analog/OMA fractions use a predeclared 7-point grid, not a global time optimum",
                   "OMA-only per-block time/power is an ideal-Shannon convex continuous-symbol allocation; it is a lower-energy relaxation, with coding/control/broadcast energy excluded for all methods",
                   "AirComp-only cannot preserve individual requests and is excluded from same-task energy ranking; its additive-sum reference is shown separately",
                   "results are communication-model upper bounds/pilot estimates, not measured hardware energy or full retained/forget VLM evaluation"]})
    allrows=[]
    for K, symbol_sweeps in [(5, [8]), (8, [8, 4])]:
        source = INPUT / f"k{K}" / "prepared.npz"
        geom = json.loads((INPUT / f"k{K}" / "geometry.json").read_text())
        with np.load(source) as z:
            learning, unlearning, channels, g, active = [z[k].copy() for k in
                ["learning", "unlearning", "channels", "gradient", "active"]]
            target = z["target"].copy()
            quantized = z["quantized"].copy()
        rows=[]
        for spp in symbol_sweeps:
            c=dict(config0, symbols_per_coordinate=spp)
            l,u,decoded,delta,payload,rho,dims,total=make_inputs(learning,quantized,c)
            assert np.array_equal(decoded, u), "Canonical Q8 request anchors must be codec-idempotent"
            delta=np.zeros_like(delta)
            eps=float(geom["application_rmse_budget"])
            lip=float(geom["controller_lipschitz"])
            analog_eps=eps/lip
            frame=int(total.sum()); deadline=frame/c["symbol_rate_hz"]
            c["round_uplink_deadline_s"]=deadline
            c["coherence_time_s"]=max(c["coherence_time_s"],float(total.max()/c["symbol_rate_hz"])+1e-9)
            zeros=np.zeros_like(payload)
            # Proposed overlapping AirComp+NOMA.
            best=None; heads=[]
            t0=time.monotonic()
            for head in range(10):
                d=optimize_head(head,channels,rho,delta,payload,dims,total,analog_eps,c,
                                c["sic_residual_fraction"],"hybrid",digital_symbols=total)
                item=dict(head=head,status="feasible" if d is not None else "no_feasible_design_found")
                if d is not None:
                    item.update(energy_j=float(d.energy_j),error_bound=float(d.error_bound),
                                app_error_bound=float(d.error_bound*lip))
                    if best is None or d.energy_j<best[0]: best=(d.energy_j,head,d)
                heads.append(item)
            rows.append(dict(K=K,symbols_per_coordinate=spp,frame_symbols=frame,frame_ms=deadline*1000,
                method="hybrid_noma_aircomp",status="feasible" if best else "no_feasible_design_found",
                best_energy_j=None if best is None else best[0],best_head=None if best is None else best[1],
                app_error_bound=None if best is None else float(best[2].error_bound*lip),heads=heads,
                elapsed_s=time.monotonic()-t0,task_eligible=True))

            # Strong phase-separated baseline: AirComp phase + digital NOMA phase.
            best=None; heads=[]; grid=fractions8 if spp==8 else fractions4; t0=time.monotonic()
            for head in range(10):
                headrows=[]
                for frac in grid:
                    analog,digital,_=split_symbols(total,payload,head,frac)
                    if np.any(analog<dims) or np.any(digital<=0):
                        headrows.append(dict(fraction=frac,status="dimension_or_frame_failed")); continue
                    floor=necessary_power(channels,payload,head,digital,c)
                    if floor is None or np.max(floor,initial=0)>c["power_limit_w"]*(1+1e-6):
                        headrows.append(dict(fraction=frac,status="peak_power_or_sic_failed")); continue
                    d=optimize_head(head,channels,rho,delta,payload,dims,analog,analog_eps,c,
                                    c["sic_residual_fraction"],"orthogonal",digital_symbols=digital)
                    item=dict(fraction=frac,status="feasible" if d is not None else "no_feasible_design_found")
                    if d is not None:
                        item.update(energy_j=float(d.energy_j),error_bound=float(d.error_bound),app_error_bound=float(d.error_bound*lip))
                        if best is None or d.energy_j<best[0]: best=(d.energy_j,head,frac,d)
                    headrows.append(item)
                heads.append(dict(head=head,candidates=headrows))
            rows.append(dict(K=K,symbols_per_coordinate=spp,frame_symbols=frame,frame_ms=deadline*1000,
                method="aircomp_learning_plus_phase_separated_noma",status="feasible" if best else "no_feasible_design_found",
                best_energy_j=None if best is None else best[0],best_head=None if best is None else best[1],
                best_analog_fraction=None if best is None else best[2],
                app_error_bound=None if best is None else float(best[3].error_bound*lip),
                heads=heads,elapsed_s=time.monotonic()-t0,task_eligible=True))

            # Strict OMA for deletion payloads, with AirComp for learning.
            best=None; heads=[]; t0=time.monotonic()
            for head in range(10):
                users=np.delete(np.arange(10),head)
                headrows=[]
                for frac in grid:
                    analog=np.floor(total*frac).astype(int)
                    analog=np.maximum(analog,dims)
                    digital=total-analog
                    if np.any(digital<=0):
                        headrows.append(dict(fraction=frac,status="dimension_or_frame_failed")); continue
                    d_oma=oma_time_energy(payload[users],channels[users,head],digital,c)
                    if not d_oma or d_oma["status"]!="feasible":
                        headrows.append(dict(fraction=frac,status="oma_peak_power_time_infeasible",
                            detail=d_oma)); continue
                    analog_design=optimize_head(head,channels,rho,np.zeros_like(delta),zeros,dims,analog,
                        analog_eps,c,0.0,"orthogonal",digital_symbols=np.maximum(digital,1))
                    if analog_design is None:
                        headrows.append(dict(fraction=frac,status="analog_aircomp_no_design")); continue
                    analog_by=np.zeros(10)
                    analog_by[analog_design.users]=(analog[:,None]/c["symbol_rate_hz"]*
                        analog_design.analog_rms**2).sum(axis=1)
                    total_by=d_oma["energy_by_user"].copy()+analog_by
                    overhead=np.full(10,c["base_overhead_j"]); overhead[head]+=c["head_overhead_j"]
                    total_by+=overhead
                    if np.any(total_by>c["available_energy_j"]*(1+1e-6)):
                        headrows.append(dict(fraction=frac,status="battery_limit_failed")); continue
                    value=float(total_by.sum())
                    item=dict(fraction=frac,status="feasible",energy_j=value,
                        oma_user_energy_j=float(d_oma["energy_by_user"].sum()),
                        analog_aircomp_energy_j=float(analog_by.sum()),
                        analog_error_bound=float(analog_design.error_bound*lip),
                        oma_min_symbols=float(d_oma["required_symbols"]),
                        radio_payload_bits=int(payload[users].sum()))
                    if best is None or value<best[0]: best=(value,head,frac,analog_design,d_oma)
                    headrows.append(item)
                heads.append(dict(head=head,candidates=headrows))
            rows.append(dict(K=K,symbols_per_coordinate=spp,frame_symbols=frame,frame_ms=deadline*1000,
                method="aircomp_learning_plus_orthogonal_unlearning",status="feasible" if best else "no_feasible_design_found",
                best_energy_j=None if best is None else best[0],best_head=None if best is None else best[1],
                best_analog_fraction=None if best is None else best[2],
                analog_app_error_bound=None if best is None else float(best[3].error_bound*lip),
                heads=heads,elapsed_s=time.monotonic()-t0,task_eligible=True))

            # Full OMA: individually convey each remote learning vector and its Q8 request vector.
            best=None; heads=[]; ql16, dl16, pl16=quantize(l,16,32)
            l16_app=float(lip*np.linalg.norm(dl16.sum(axis=0)))
            use16=l16_app<=eps
            if use16: payload_l=pl16; l_app=l16_app; learn_bits=16
            else:
                payload_l=np.where(np.linalg.norm(l,axis=-1)>0,dims[None,:]*32,0)
                l_app=0.0; learn_bits=32
            full_payload=payload_l+payload
            t0=time.monotonic()
            for head in range(10):
                users=np.delete(np.arange(10),head)
                bits=full_payload[users]
                sched=oma_time_energy(bits,channels[users,head],total,c)
                item=dict(head=head,status="feasible" if sched and sched["status"]=="feasible" else "oma_peak_power_time_infeasible")
                if sched and sched["status"]=="feasible":
                    total_by=sched["energy_by_user"].copy()
                    overhead=np.full(10,c["base_overhead_j"]); overhead[head]+=c["head_overhead_j"]
                    total_by+=overhead
                    if np.any(total_by>c["available_energy_j"]*(1+1e-6)):
                        item["status"]="battery_limit_failed"
                    else:
                        value=float(total_by.sum()); item.update(energy_j=value,
                            oma_min_symbols=float(sched["required_symbols"]),
                            learning_codec_bits=learn_bits,learning_app_error_bound=l_app,
                            payload_bits=int(bits.sum()))
                        if best is None or value<best[0]: best=(value,head,sched)
                heads.append(item)
            rows.append(dict(K=K,symbols_per_coordinate=spp,frame_symbols=frame,frame_ms=deadline*1000,
                method="oma_only_individual",status="feasible" if best else "no_feasible_design_found",
                best_energy_j=None if best is None else best[0],best_head=None if best is None else best[1],
                learning_codec_bits=learn_bits,learning_app_error_bound=l_app,
                heads=heads,elapsed_s=time.monotonic()-t0,task_eligible=True))

            # AirComp-only reference for the original additive objective; it cannot
            # meet the per-request controller and is deliberately excluded from ranking.
            combined=learning+quantized
            lcomb=np.pad(combined,((0,0),(0,(-combined.shape[1])%c["block_dimension"]))).reshape(10,len(dims),-1)
            rhocomb=np.linalg.norm(lcomb,axis=-1)
            best=None; heads=[]; t0=time.monotonic()
            for head in range(10):
                d=optimize_head(head,channels,rhocomb,np.zeros_like(delta),zeros,dims,total,eps,c,0.0,
                    "orthogonal",digital_symbols=total)
                item=dict(head=head,status="feasible" if d is not None else "no_feasible_design_found")
                if d is not None:
                    item.update(energy_j=float(d.energy_j),update_error_bound=float(d.error_bound))
                    if best is None or d.energy_j<best[0]: best=(d.energy_j,head,d)
                heads.append(item)
            rows.append(dict(K=K,symbols_per_coordinate=spp,frame_symbols=frame,frame_ms=deadline*1000,
                method="aircomp_only_additive_reference",status="feasible" if best else "no_feasible_design_found",
                best_energy_j=None if best is None else best[0],best_head=None if best is None else best[1],
                task_eligible=False,controller_target_gap_norm=float(np.linalg.norm(target-(learning.sum(axis=0)+quantized.sum(axis=0)))),
                heads=heads,elapsed_s=time.monotonic()-t0))
            print("CASE",K,spp,json.dumps([{k:r.get(k) for k in ["method","status","best_energy_j","best_head","best_analog_fraction","task_eligible"]} for r in rows[-5:]]),flush=True)
        case=dict(K=K,rows=rows)
        outcase = OUT / f"k{K}"
        outcase.mkdir(parents=True, exist_ok=True)
        write(outcase/"radio_results.json",case)
        allrows.extend(rows)
    summary["status"]="completed"
    summary["variant"] = args.variant
    summary["crossmodal_enabled"] = (args.variant == "crossmodal")
    summary["cases"]=[{"K":5,"symbols_per_coordinate":[8]},
                      {"K":8,"symbols_per_coordinate":[8,4]}]
    summary["prepared_cases"]=[5,8]
    summary["rows"]=allrows
    write(OUT/"radio_summary.json",summary)
    print("RADIO_COMPLETED",OUT/"radio_summary.json",flush=True)


if __name__ == "__main__":
    main()



