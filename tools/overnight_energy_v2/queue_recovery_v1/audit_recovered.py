"""Reuse verified prediction caches and include recovered comparison jobs."""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--include-restart',action='store_true')
    parser.add_argument('--summary-only',action='store_true')
    args=parser.parse_args();root=args.project.resolve()
    sys.path[:0]=[str(root),str(root/'tools/energy_quality_v1')]
    import energy_quality as eq
    suite=root/'results/comparisons_seed0_solver_v2'
    extras=[]
    if (suite/'retain_only_ideal/manifest.json').exists():extras.append('retain_only_ideal')
    if args.include_restart:extras.append('uncoupled_hybrid_restart_v1')
    eq.METHODS=eq.METHODS+tuple(extras)
    settings=argparse.Namespace(project=root,suite=suite,output=root/'results/energy_quality_v1',
        device='cuda',cpu_threads=4,samples=1000,batch_size=32,
        validation_loss_max=None,forget_kl_max=None,attack_gap_max=None)
    if args.summary_only:eq.summarize(settings)
    else:eq.collect(settings)
    # Preserve this derived view independently of later standard summaries.
    out=root/'results/overnight_energy_v2/queue_recovery_v1'
    out.mkdir(parents=True,exist_ok=True)
    report=settings.output/'energy_quality_report.json'
    (out/'recovered_quality_report.json').write_bytes(report.read_bytes())


if __name__=='__main__':
    main()
