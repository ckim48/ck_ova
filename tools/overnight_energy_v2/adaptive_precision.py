"""Client-local block precision under an explicit global quantization budget."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np


BIT_DEPTHS = (2, 4, 6, 8, 10, 12)
DEPTH_METADATA_BITS = 3


def budget_quantize(updates, epsilon, fraction, scale_metadata_bits=32, active_client_bound=None):
    """No cross-client update geometry is needed to select a local bit depth.

    With N nonzero clients and M blocks, each block gets delta <= fraction *
    epsilon / (N sqrt(M)). Therefore ||sum_u delta_u||_2 <= fraction * epsilon
    for any choice of head. The 3-bit depth header is included in each payload.
    """
    from wireless import quantize
    if epsilon <= 0 or not 0 < fraction < 1:
        raise ValueError("Positive epsilon and quantization-budget fraction in (0,1) required.")
    active = np.max(np.abs(updates), axis=-1) > 0
    actual_clients = int(np.any(active, axis=1).sum())
    clients = actual_clients if active_client_bound is None else active_client_bound
    if clients < actual_clients:
        raise ValueError("The known requesting-client bound is smaller than the active client count.")
    decoded = np.zeros_like(updates)
    delta = np.zeros(active.shape, dtype=float)
    depths = np.zeros(active.shape, dtype=int)
    payload = np.zeros(active.shape, dtype=int)
    if actual_clients == 0:
        return decoded, delta, payload, depths
    target = fraction * epsilon / (clients * np.sqrt(updates.shape[1]))
    for bits in BIT_DEPTHS:
        candidate, bound, _ = quantize(updates, bits, scale_metadata_bits)
        select = active & (depths == 0) & (bound <= target)
        decoded[select] = candidate[select]
        delta[select] = bound[select]
        depths[select] = bits
    if np.any(active & (depths == 0)):
        raise ValueError("Maximum supported precision cannot meet the assigned budget.")
    payload[active] = (updates.shape[-1] * depths[active]
                       + scale_metadata_bits + DEPTH_METADATA_BITS)
    assert np.linalg.norm(delta.sum(axis=0)) <= fraction * epsilon * (1 + 1e-12)
    return decoded, delta, payload, depths


def design_prequantized(learning, channels, delta, payload, config, epsilon, mode, fraction=0.5):
    from wireless import optimize_head
    from phase_split_grid import split_symbols, necessary_power
    rho = np.linalg.norm(learning, axis=-1)
    dims = np.full(learning.shape[1], learning.shape[2])
    symbols = dims * config["symbols_per_coordinate"]
    best = None; heads = []
    for head in range(config["users"]):
        item = {"head": head, "status": "not_attempted"}
        if mode == "hybrid":
            analog, digital = symbols.copy(), symbols.copy()
        elif mode == "orthogonal":
            analog, digital, _ = split_symbols(symbols, payload, head, fraction)
        else:
            raise ValueError("Unsupported mode")
        lower = necessary_power(channels, payload, head, digital, config)
        if lower is None or np.max(lower, initial=0) > config["power_limit_w"] * (1+1e-6):
            item["status"] = "necessary_digital_power_condition_failed"
            heads.append(item); continue
        if np.any(analog < dims):
            item["status"] = "symbol_dimension_condition_failed"
            heads.append(item); continue
        design = optimize_head(head, channels, rho, delta, payload, dims, analog,
            epsilon, config, config["sic_residual_fraction"], mode, digital_symbols=digital)
        item["status"] = "feasible" if design is not None else "no_feasible_design_found"
        if design is not None:
            item["energy_bound_j"] = design.energy_j
            if best is None or design.energy_j < best.energy_j:
                best = design
        heads.append(item)
    return best, heads


def exact_statistics(design, learning, unlearning, decoded, dimension):
    """Conditional bias + noise MSE in the implemented Gaussian recovery model."""
    bias = learning[design.head] + unlearning[design.head]
    variance = 0.0
    for block in range(learning.shape[1]):
        rho = design.rho[:, block]
        gain = design.receiver[block] * design.normalized_channels[:, block]
        gain *= design.analog_rms[:, block] / np.where(rho > 0, rho, 1)
        bias[block] += (gain[:, None] * learning[design.users, block]).sum(axis=0)
        bias[block] += decoded[design.users, block].sum(axis=0)
        residual = np.sum(design.chi * (design.normalized_channels[:, block]
                                       * design.digital_sqrt_power[:, block])**2)
        coordinates = max(0, min(learning.shape[2], dimension - block*learning.shape[2]))
        variance += coordinates * design.receiver[block]**2 * (1+residual) / (2*design.symbols[block])
    ideal = (learning + unlearning).sum(axis=0).reshape(-1)[:dimension]
    error = bias.reshape(-1)[:dimension] - ideal
    quantized_u = unlearning[design.head] + decoded[design.users].sum(axis=0)
    true_u = unlearning.sum(axis=0)
    return {"analytic_total_rmse": float(np.sqrt(error@error + variance)),
            "unlearning_component_error": float(np.linalg.norm((quantized_u-true_u).reshape(-1)[:dimension]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--budget-fractions", type=float, nargs="+", default=[0.1, 0.25, 0.5, 0.75])
    parser.add_argument("--analog-fractions", type=float, nargs="+", default=[0.125,0.25,0.375,0.5,0.625,0.75,0.875])
    args = parser.parse_args()
    root = args.project.resolve(); sys.path.insert(0, str(root))
    from transport import energy_accounting
    from phase_split_grid import make_inputs
    source = root / "results/failed_round_diagnostic"
    config = json.loads((source/"metadata.json").read_text())["config"]
    with np.load(source/"updates.npz") as data:
        raw_l, raw_u, channels = [data[k].copy() for k in ("learning","unlearning","channels")]
    learning, unlearning, *_ = make_inputs(raw_l, raw_u, config)
    epsilon = max(config["absolute_tolerance_floor"], config["relative_tolerance"]
                  * np.linalg.norm(np.linalg.norm(learning,axis=-1).sum(axis=0)))
    expected_time = learning.shape[1]*learning.shape[2]*config["symbols_per_coordinate"]/config["symbol_rate_hz"]
    output = root/"results/overnight_energy_v2/adaptive_precision"
    output.mkdir(parents=True,exist_ok=True)
    manifest = {"input_sha256":hashlib.sha256((source/"updates.npz").read_bytes()).hexdigest(),
        "core_source_sha256":{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ("wireless.py","transport.py")},
        "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "bit_depths":list(BIT_DEPTHS),"depth_header_bits_per_active_block":DEPTH_METADATA_BITS,
        "budget_fractions":args.budget_fractions,"analog_fractions":args.analog_fractions,
        "known_requesting_client_bound":config.get("requesting_users",config["users"]),
        "epsilon":float(epsilon),"config":config,
        "interpretation":"Exploratory CPU fixed-update diagnostic. Shared adaptive quantizer is applied to BOTH Hybrid and phase-separated links. Per-block depth metadata included; global epsilon and active-client count are the existing protocol's common control information. No test-set tuning, no VLM-quality or certified unlearning claim. Tail padding is retained identically to baseline."}
    f=output/"manifest.json"
    if f.exists() and json.loads(f.read_text())!=manifest:
        raise ValueError("Existing experiment differs; use a new version.")
    f.write_text(json.dumps(manifest,indent=2)+"\n")
    rfile=output/"results.json"
    rows=json.loads(rfile.read_text()) if rfile.exists() else []
    done={(r["quantization_budget_fraction"],r["mode"],r.get("analog_fraction")) for r in rows}
    for budget in args.budget_fractions:
        try:
            decoded, delta, payload, depths=budget_quantize(unlearning,epsilon,budget,config["scale_metadata_bits"],
                active_client_bound=config.get("requesting_users",config["users"]))
        except ValueError as error:
            row={"quantization_budget_fraction":budget,"mode":"quantizer","status":"quantization_budget_unmet","error":str(error)}
            if (budget,"quantizer",None) not in done:
                rows.append(row);rfile.write_text(json.dumps(rows,indent=2)+"\n")
            continue
        for mode,fraction in [("hybrid",None)]+[("orthogonal",a) for a in args.analog_fractions]:
            if (budget,mode,fraction) in done:
                continue
            began=time.monotonic()
            design,heads=design_prequantized(learning,channels,delta,payload,config,epsilon,mode,fraction or 0.5)
            values,counts=np.unique(depths[depths>0],return_counts=True)
            row={"quantization_budget_fraction":budget,"mode":mode,"analog_fraction":fraction,
                "status":"no_feasible_design_found","heads":heads,
                "depth_histogram_all_clients":{str(int(b)):int(c) for b,c in zip(values,counts)},
                "quantization_only_bound":float(np.linalg.norm(delta.sum(axis=0))),
                "all_client_payload_bits":int(payload.sum())}
            if design is not None:
                row.update(energy_accounting(design,config),status="feasible")
                row.update(exact_statistics(design,learning,unlearning,decoded,raw_l.shape[1]))
                row["remote_payload_bits"]=int(payload[design.users].sum())
                assert abs(row["uplink_s"]-expected_time)<1e-10
                assert row["analytic_total_rmse"]<=row["error_bound"]*(1+1e-4)+1e-12
                assert row["error_bound"]<=epsilon*(1+1e-5)
            row["seconds"]=time.monotonic()-began
            rows.append(row)
            temporary=rfile.with_suffix(".partial.json")
            temporary.write_text(json.dumps(rows,indent=2)+"\n");temporary.replace(rfile)
            print(json.dumps({k:v for k,v in row.items() if k!="heads"}),flush=True)
    best=[]
    for mode in ("hybrid","orthogonal"):
        feasible=[r for r in rows if r["mode"]==mode and r["status"]=="feasible"]
        if feasible:
            best.append({k:v for k,v in min(feasible,key=lambda r:r["energy_bound_j"]).items() if k!="heads"})
    (output/"best_observed.json").write_text(json.dumps(best,indent=2)+"\n")
    print("BEST_OBSERVED",json.dumps(best),flush=True)


if __name__=="__main__":
    main()
