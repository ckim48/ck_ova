"""Release our idle downstream waiter if its repair supervisor exits."""
import argparse
import json
import os
from pathlib import Path
import signal
import time


def identity(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()
        return None if fields[0]=='Z' else fields[19]
    except FileNotFoundError:
        return None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--supervisor',type=int,required=True)
    p.add_argument('--waiter',type=int,action='append',required=True)
    p.add_argument('--waiter-identity',action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();supervisor_identity=identity(a.supervisor)
    if len(a.waiter)!=len(a.waiter_identity):raise ValueError('Waiter identities mismatch.')
    while supervisor_identity is not None and identity(a.supervisor)==supervisor_identity:
        time.sleep(30)
    released=[]
    for pid,expected in zip(a.waiter,a.waiter_identity):
        if identity(pid)==expected:
            try:os.kill(pid,signal.SIGCONT);released.append(pid)
            except ProcessLookupError:pass
    (a.output/'waiter_release_guard.json').write_text(json.dumps({
        'supervisor_exited':True,'waiters':a.waiter,'released_if_still_same_process':released},indent=2)+'\n')


if __name__=='__main__':
    main()
