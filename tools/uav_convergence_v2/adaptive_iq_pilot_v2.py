"""Restart the same pilot after a reference-only zero-LR endpoint failure."""
import argparse
from pathlib import Path
import time
import adaptive_iq_pilot_v1 as original
import adaptive_iq_math_v2 as controller

previous, parent, learning, audit = original.previous, original.parent, original.learning, original.audit
ROOT, read, write, sha = original.ROOT, original.read, original.write, original.sha
OUT = ROOT/'results/adaptive_iq_pilot_v2_20261030'
MODES = original.MODES


def register():
    path = OUT/'protocol.json'
    if path.exists():
        p = read(path)
        for f, h in p['source_sha256'].items(): assert sha(ROOT/f) == h, f
        return p
    p = dict(read(ROOT/'results/adaptive_iq_pilot_v1_20261030/protocol.json'))
    p.update(registered_unix=time.time(), correction='v1 failed at reference round200 because cosine LR is exactly0. No candidate outcome was observed. Treat a zero-LR/zero-update/zero-noise endpoint as an exact no-op. Preserve v1 and restart identical candidates.',
        source_sha256={str(Path(f).relative_to(ROOT)):sha(Path(f)) for f in [__file__, original.__file__,
            controller.__file__, controller.original.__file__, previous.__file__, learning.__file__]})
    write(path, p); write(OUT/'math_checks.json', controller.checks())
    return p


def install():
    original.OUT = OUT; original.controller = controller; original.register = register


def retrain(*args, **kwargs):
    install()
    return original.retrain(*args, **kwargs)


def run():
    register(); install(); audit.runtime('CapERA'); original.run()


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['register', 'run'])
    a = p.parse_args(); {'register':register, 'run':run}[a.action]()
