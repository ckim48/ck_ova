"""Initialize the legacy module paths before the frozen pilot imports train."""
from pathlib import Path
import time
import adaptive_iq_pilot_v1 as pilot

if __name__ == '__main__':
    pilot.register()
    pilot.write(pilot.OUT/'startup_correction.json', dict(unix=time.time(),
        source_sha256=pilot.sha(Path(__file__)),
        original_failure='ModuleNotFoundError: train, before any model/reference run began.',
        correction='Initialize audit.runtime before importing train; no learning or search change.'))
    pilot.audit.runtime('CapERA')
    pilot.run()
