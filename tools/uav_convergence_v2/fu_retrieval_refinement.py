"""Bounded development follow-up between infeasible RM0.1 and feasible RM1."""
from pathlib import Path
import argparse
import os
import subprocess
import sys
import time

import fu_retrieval_alignment as first
from fu_retrieval_alignment import ROOT, SOURCE, DATASETS, PROPOSED, read, write, sha, link

PREVIOUS = first.OUT
OUT = ROOT / "results/fu_retrieval_refinement_v2_20261027"
VARIANTS = {
    f"fu_proxy_{cm}_rm{tag}": dict(label=f"P-{'R' if cm=='rank' else 'E'}{tag}", proxy=True,
        relation=cm=="rank", retention=value, adaptive=False, request_lr=False, ideal=False, fusion=False)
    for cm in ["embed", "rank"] for tag, value in [("025", .25), ("05", .5)]
}


def register():
    path = OUT/"protocol.json"
    if path.exists():
        result = read(path)
        for group in ["source_sha256", "input_sha256"]:
            for name, digest in result[group].items(): assert sha(ROOT/name) == digest, name
        return result
    old = read(PREVIOUS/"protocol.json")
    assert read(PREVIOUS/"finished.json")["passed"]
    sources = dict(old["source_sha256"])
    sources[str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
    inputs = dict(old["input_sha256"])
    for dataset in DATASETS:
        for method in old["variants"]:
            path = PREVIOUS/"cases/baseline"/(dataset.lower()+"_seed0")/method/"outcome.json"
            inputs[str(path.relative_to(ROOT))] = sha(path)
    result = dict(old, registered_unix=time.time(), stage=2, variants=VARIANTS, new_arms=8,
        reason="All RM0/0.1 proxy arms were radio-infeasible under the unchanged FU-relative error guard and hard TX cap; RM1 completed. Test the fixed intermediate weights0.25 and0.5 for both consistency definitions. Preserve and publish every first-stage and follow-up outcome.",
        followup_limits="This is adaptive validation-set development, not a new independent test or confirmatory experiment. Stop this bounded refinement after these eight runs; do not continue a ranking-driven coefficient search or revise baselines/limits.",
        source_sha256=sources, input_sha256=inputs)
    write(OUT/"protocol.json", result)
    write(OUT/"unit_checks.json", first.unit_checks())
    link(SOURCE/"prepared/airscape_seed0", OUT/"prepared/airscape_seed0")
    return result


def state(dest=OUT, **kwargs):
    write(Path(dest)/"status.json", dict(unix=time.time(), pid=os.getpid(), **kwargs))


first.OUT = OUT; first.VARIANTS = VARIANTS; first.register = register; first.state = state
first.parent.OUT = OUT; first.parent.VARIANTS = VARIANTS
first.parent.register = register; first.parent.state = state
verify = first.verify


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("action", choices=["run", "variant", "verify"])
    parser.add_argument("--variant", choices=list(VARIANTS)); args = parser.parse_args()
    if args.action == "variant": first.parent.run_variant(args.variant); return
    if args.action == "verify": print(verify()); return
    register()
    for variant in VARIANTS:
        subprocess.run([sys.executable, str(Path(__file__)), "variant", "--variant", variant], cwd=ROOT, check=True)
    result = verify(); write(OUT/"finished.json", dict(unix=time.time(), **result))
    state(stage="completed", outcomes=result["outcomes"]); print(result, flush=True)


if __name__ == "__main__": main()
