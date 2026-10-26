"""Layout-only repair using frozen07:42 snapshot; no experiment/state changes."""
import base64
import io
import json
from pathlib import Path
import zipfile

root = Path('/home/wnlab/Documents/Codex/2026-10-20/files-pasted-by-the-user-standalone/outputs/ova_CK')
results = root/'results/overnight_energy_v2'
snapshot = json.loads((results/'heartbeat_0742_snapshot.json').read_text())
source = (root/'tools/overnight_energy_v2/capture_0742.py').read_text()
plot_code = source[source.index('import matplotlib\n'):source.index('\nreport = f')]
assert plot_code.count('axes[0].set_ylim(0, 1000)') == 1
plot_code = plot_code.replace('axes[0].set_ylim(0, 1000)', 'axes[0].set_ylim(0, 1150)')
exec(compile(plot_code, '<frozen-snapshot-layout-only>', 'exec'), dict(results=results,
     rows15=snapshot['public_history_fp16_results'], rows13=snapshot['public_history_float32_results']))
buffer = io.BytesIO()
with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
    for ext in ['png', 'svg']:
        path = results/f'heartbeat_0742_paid_codec.{ext}'
        archive.writestr(str(path.relative_to(root)), path.read_bytes())
    archive.writestr(str(Path(__file__).relative_to(root)), Path(__file__).read_bytes())
print(base64.b64encode(buffer.getvalue()).decode())
