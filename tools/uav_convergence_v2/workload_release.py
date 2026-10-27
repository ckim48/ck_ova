"""Complete declared workload campaign, independently audit, then export figures."""
import argparse
from pathlib import Path
import subprocess
import sys
import workload_sweeps_v3 as study


def run(name,*args):
    subprocess.run([sys.executable,str(Path(__file__).with_name(name)),*args],cwd=study.ROOT,check=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','finish']);args=p.parse_args()
    if args.action=='run':
        run('workload_sweeps_v3.py','unit')
        run('workload_sweeps_v3.py','case','--case','control')
        run('workload_completion.py','run')
    else:run('workload_completion.py','verify')
    run('workload_audit.py')
    run('workload_charts.py')
    run('workload_artifact_check.py')


if __name__=='__main__':main()
