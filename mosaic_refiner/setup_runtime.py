"""Explicit setup: missing wheels/models only; no server or core/model update."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import tempfile
import re
from urllib.error import URLError
from .runtime_http import urlopen

from .checkpoint import digest_file
from .model_setup import SetupError, catalog, install, inspect
from .runtime_settings import RuntimeConnection

REQUIRED_NODES = ('MiniMaxH3ReferenceToVideo', 'MosaicH3LatentUpscale2x', 'H3TrialChunkedMLP',
                  'H3V2VGranularFractionalDenoise', 'MiniMaxH3SetAVNoiseMask',
                  'UnloadAllModels', 'VHS_LoadVideoPath', 'LoadAudio',
                  'UNETLoader','CLIPLoader','VAELoader','LoraLoaderBypassModelOnly','ImageToMask','LoadImageMask','MiniMaxH3SigmaShift',
                  'BasicGuider','KSamplerSelect','ManualSigmas','RandomNoise','SamplerCustomAdvanced',
                  'VAEDecodeTiled','CreateVideo','SaveVideo')
REQUIRED_FILES = ('comfy_extras/nodes_minimax_h3.py', 'comfy/ldm/minimax/model.py',
                  'comfy_extras/nodes_lora_debug.py',
                  'custom_nodes/mosaic_h3_public.py',
                  'custom_nodes/h3_trial_chunked_mlp.py',
                  'custom_nodes/ComfyUI-H3-Motion-Context-MultiRef/h3_v2v_fractional.py')


def idle(connection):
    """Refuse an occupied/unidentified listener. Never stop/restart it."""
    import psutil
    listeners = {c.pid for c in psutil.net_connections('tcp') if c.laddr.port == connection.port and c.status == 'LISTEN'}
    if not listeners:
        return 'server_stopped'
    from .runtime_observation import identity
    identity(connection)
    with urlopen(connection.endpoint+'/queue', timeout=10) as response:
        q = json.load(response)
    if q.get('queue_running') or q.get('queue_pending'):
        raise SetupError('ComfyUIが使用中です。処理の完了後にセットアップしてください')
    return 'server_idle'


def inspect_nodes(connection):
    connection.validate()
    state = idle(connection)
    files = [name for name in REQUIRED_FILES if not (connection.root/name).is_file()]
    if state == 'server_stopped':
        return {'server': state, 'nodes_verified': False, 'missing_nodes': [], 'missing_runtime_files': files,
                'next': '選択したComfyUIを通常の起動方法で起動し、再確認してください'}
    with urlopen(connection.endpoint+'/object_info', timeout=30) as response:
        nodes = json.load(response)
    missing = [name for name in REQUIRED_NODES if name not in nodes]
    return {'server': state, 'nodes_verified': not missing and not files, 'missing_nodes': missing,
            'missing_runtime_files': files}


def install_reference_node(connection,source=None):
    connection.validate()
    idle(connection)
    source = Path(source) if source else Path(__file__).with_name('comfy_reference_reuse.py')
    if not source.is_file():raise SetupError('参照再利用adapterはこの版に含まれていません')
    target = (connection.root/'custom_nodes/mosaic_refiner_reference_reuse.py').resolve()
    if connection.root not in target.parents:
        raise SetupError('custom_nodesの配置先が選択rootの外です')
    expected = digest_file(source)
    if target.exists():
        if target.read_bytes().replace(b'\r\n', b'\n') != source.read_bytes().replace(b'\r\n', b'\n'):
            raise SetupError('参照再利用nodeに同名の変更があります。上書きしません')
        return {'state': 'verified', 'sha256': digest_file(target), 'restart_required': False}
    with source.open('rb') as src, target.open('xb') as dst:
        shutil.copyfileobj(src, dst)
    return {'state': 'installed', 'sha256': expected, 'restart_required': True}


def resources(connection, lora_profile='unlocked_v2'):
    from .restoration_models import BASIC, model_directory as restoration_dir
    from .learned_detection import MODEL, model_directory as detector_dir
    from .lora_setup import model as lora_model
    return ([(m, connection.root/'models') for m in catalog()]
            + [(BASIC, restoration_dir()), (MODEL, detector_dir()), (lora_model(lora_profile), connection.root/'models')])


def resource_status(connection, verify=True, cancelled=lambda: False, progress=lambda **kw: None, lora_profile='unlocked_v2'):
    rows = []
    for model, target in resources(connection, lora_profile):
        if cancelled():raise KeyboardInterrupt
        idle(connection)
        rows.extend(inspect(target, models=[model], verify=verify, cancelled=cancelled, progress=progress))
    return rows


def acquire_resources(connection, cancelled=lambda: False, progress=lambda **kw: None, credentials=None, lora_profile='unlocked_v2', accept_terms=False):
    if not accept_terms:raise SetupError('各配布元の利用条件を本人が確認してから取得してください')
    from .lora_setup import preflight_selection
    credentials = preflight_selection(lora_profile,credentials)
    connection.validate()
    idle(connection)
    result = []
    for model, target in resources(connection, lora_profile):
        if cancelled():raise KeyboardInterrupt
        idle(connection)
        result.extend(install(target, models=[model], cancelled=cancelled, progress=progress,
                              credentials=credentials,terms_confirmed=accept_terms)['models'])
    return result


def import_unlocked(connection, source, cancelled=lambda:False):
    raise SetupError('必須LoRAは本人のCivitai API認証で取得してください。手動ファイルの取り込みには対応していません')


def run_python(python, args, cwd, cancelled=lambda:False):
    import time
    if cancelled():raise KeyboardInterrupt
    args=list(args)
    if args[:2]==['-m','pip'] and '--isolated' not in args:
        args.insert(2,'--isolated')
    run=subprocess.Popen([str(python),'-B',*args],cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
            text=True,encoding='utf-8',errors='replace',env=pip_environment(),
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    started=time.monotonic()
    while True:
        if cancelled() or time.monotonic()-started>1800:
            import psutil
            try:
                owner=psutil.Process(run.pid); children=owner.children(recursive=True)
                for child in reversed(children):
                    try:child.terminate()
                    except psutil.NoSuchProcess:pass
                owner.terminate()
                _,alive=psutil.wait_procs([*children,owner],timeout=5)
                for child in alive:child.kill()
            except psutil.NoSuchProcess:pass
            run.communicate()
            if cancelled():raise KeyboardInterrupt
            raise SetupError('依存確認の時間制限を超えました')
        try:stdout,stderr=run.communicate(timeout=.2);break
        except subprocess.TimeoutExpired:continue
    if run.returncode:raise SetupError('依存の確認/取得に失敗しました: '+stderr[-1800:])
    return stdout


def pip_environment():
    """Explicit official indexes must not inherit extra-index or pip.ini settings."""
    paths=[]
    for value in os.environ.get('PATH','').split(os.pathsep):
        try:
            if value and Path(value).is_dir():paths.append(value)
        except OSError:pass
    return {**{k:v for k,v in os.environ.items() if not k.upper().startswith('PIP_')},
            'PIP_CONFIG_FILE':os.devnull,'PYTHONUTF8':'1','PYTHONNOUSERSITE':'1','PATH':os.pathsep.join(paths)}


def distributions(connection):
    code = "import importlib.metadata as m,json,re;print(json.dumps({re.sub(r'[-_.]+','-',d.metadata['Name'].lower()):d.version for d in m.distributions() if d.metadata['Name']}))"
    return json.loads(run_python(connection.python, ['-c', code], connection.root))


def validate_report(report, installed):
    installed = {re.sub(r'[-_.]+', '-', name.lower()):version for name,version in installed.items()}
    pins = []
    for row in report.get('install', []):
        metadata = row['metadata']
        name = re.sub(r'[-_.]+', '-', metadata['name'].lower())
        if name.startswith('opencv-') and any(n.startswith('opencv-') for n in installed):
            raise SetupError('既存のcv2配布物と競合します。OpenCVを重複導入しません')
        if name in installed:
            raise SetupError('既存依存の置換が必要です。自動更新しません: '+name)
        if name in ('torch', 'torchvision', 'torchaudio'):
            raise SetupError('GPU用TorchはComfyUI公式導入手順で準備してください。CPU版へ自動置換しません')
        pins.append(metadata['name']+'=='+metadata['version'])
    return pins


def prepare_libraries(connection, apply=False, cancelled=lambda:False):
    """pip resolves against every existing installed version before any install.

    Wheels only; no arbitrary setup.py/source builds. Applying the exact plan
    with --no-deps cannot upgrade an existing library as a resolver side effect.
    """
    connection.validate()
    idle(connection)
    existing = distributions(connection)
    if not existing.get('torch','').endswith('+cu130'):
        raise SetupError('このH3導入構成はCUDA 13.0版Torchが必要です。既存GPU依存は置換しません')
    from .setup_packages import require_secure_torch
    require_secure_torch(existing.get('torch',''))
    from .setup_environment import recipe
    extra=[r for r in recipe()['requirements']
           if not (r.startswith('opencv-') and any(n.startswith('opencv-') for n in existing))]
    requirements = connection.root/'requirements.txt'
    if not requirements.is_file():
        raise SetupError('選択したComfyUIのrequirements.txtがありません')
    with tempfile.TemporaryDirectory(prefix='mosaic-setup-') as folder:
        constraints = Path(folder)/'constraints.txt'
        constraints.write_text('\n'.join(name+'=='+version for name, version in sorted(existing.items())), encoding='utf-8')
        report_path = Path(folder)/'report.json'
        effective = requirements
        if any(name.startswith('opencv-') for name in existing):
            run_python(connection.python, ['-c', "import cv2; assert all(hasattr(cv2,n) for n in ('imdecode','cvtColor','calcOpticalFlowPyrLK'))"],
                       connection.root,cancelled)
            effective = Path(folder)/'requirements.txt'
            effective.write_text('\n'.join(
                line for line in requirements.read_text(encoding='utf-8').splitlines()
                if not re.match(r'^\s*opencv-(?:contrib-)?python(?:-headless)?(?:[<>=!~;\s]|$)',line,re.I)),
                encoding='utf-8')
        run_python(connection.python, ['-m', 'pip', 'install', '--dry-run', '--only-binary=:all:',
                   '--index-url','https://pypi.org/simple',
                   '--report', str(report_path), '-c', str(constraints), '-r', str(effective),
                   'safetensors', 'einops', 'psutil',*extra], connection.root,cancelled)
        pins = validate_report(json.loads(report_path.read_text(encoding='utf-8')), existing)
        if apply and pins:
            idle(connection)
            # Recheck after resolution in case another setup changed this env.
            if distributions(connection) != existing:
                raise SetupError('依存の確認後に環境が変更されました。取得計画を作り直してください')
            run_python(connection.python, ['-m', 'pip', 'install', '--no-deps', '--only-binary=:all:',
                       '--index-url','https://pypi.org/simple',*pins], connection.root,cancelled)
        after = distributions(connection) if apply else existing
        if any(after.get(name) != version for name, version in existing.items()):
            raise SetupError('既存依存の版が変化しました。生成を開始せず確認してください')
    return {'missing_libraries': pins, 'applied': apply, 'existing_versions_preserved': True,
            'gpu_libraries_replaced': False, 'node_libraries_included': True}
