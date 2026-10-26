"""Publish only the completed comparison figures and numeric chart summaries."""
import subprocess


def publish_figures(root,out):
    git=['git','--git-dir=.git-push-ready/code-charts.git','--work-tree=.']
    paths=sorted(p for p in (out/'figures').glob('*') if p.suffix in {'.png','.pdf','.svg','.csv','.json'})
    paths+=sorted((root/'results/figures').glob('mission_satellite_iot_uav_actual_heads_1x2_global_iot_five_sites_reference_square_cluster_motion.*'))
    if not paths:return dict(status='no_files')
    try:
        remote=subprocess.check_output(git+['remote','get-url','origin'],cwd=root,text=True).strip()
        assert remote=='https://github.com/ckim48/ck_ova.git'
        # Avoid incorporating someone else's staged work into this automatic commit.
        staged=subprocess.check_output(git+['diff','--cached','--name-only'],cwd=root,text=True).strip()
        if staged:return dict(status='deferred',reason='Existing staged changes')
        subprocess.run(git+['add','--']+[str(p.relative_to(root)) for p in paths],cwd=root,check=True,capture_output=True)
        check=subprocess.run(git+['diff','--cached','--quiet'],cwd=root)
        if check.returncode==0:return dict(status='unchanged')
        subprocess.run(git+['commit','-m','Add completed IoT UAV VLM comparison figures'],cwd=root,check=True,capture_output=True)
        pushed=subprocess.run(git+['push','origin','main'],cwd=root,capture_output=True,text=True)
        if pushed.returncode:return dict(status='push_failed',reason=pushed.stderr[-1000:])
        local=subprocess.check_output(git+['rev-parse','HEAD'],cwd=root,text=True).strip()
        remote=subprocess.check_output(git+['ls-remote','origin','refs/heads/main'],cwd=root,text=True).split()[0]
        assert local==remote
        return dict(status='pushed',commit=local)
    except Exception as error:return dict(status='failed',error=str(error))
