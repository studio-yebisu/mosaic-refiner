"""Explicit complete setup. Never update an existing core/GPU stack or cancel jobs."""
from hashlib import sha256
from contextlib import contextmanager
from contextvars import ContextVar
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

from .checkpoint import atomic_json, job_lock
from .model_setup import SetupError, checked_hash, default_cache
from .runtime_settings import RuntimeConnection, save
from .setup_archives import publish_archive, verify_existing_source, verify_published
from .setup_hardware import preflight, storage_check

ASSETS = Path(__file__).with_name('setup_assets')
_ACTIVE_RECIPE = ContextVar('mosaic_setup_recipe', default=None)


def recipe():
    if _ACTIVE_RECIPE.get() is not None:
        return _ACTIVE_RECIPE.get()
    data=json.loads((ASSETS/'recipe.json').read_text(encoding='utf-8'))
    if data['schema_version']!=1: raise SetupError('導入構成の版が不明です')
    return data


@contextmanager
def using_recipe(data):
    token = _ACTIVE_RECIPE.set(data)
    try:
        yield
    finally:
        _ACTIVE_RECIPE.reset(token)


def resolve_recipe(connection, new, cancelled=lambda:False, progress=lambda **kw:None):
    """Resolve only on an explicit setup action; interrupted installs reuse a snapshot."""
    folder = state_directory(connection)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/'resolved-recipe.json'
    with job_lock(folder):
        if path.exists():
            saved = json.loads(path.read_text(encoding='utf-8'))
            changed=(saved.get('root') != str(connection.root) or saved.get('new') is not new
                    or saved.get('connection') != connection.to_dict()
                    or saved.get('python_abi') != list(sys.version_info[:2])
                    or saved.get('recipe',{}).get('profile') != recipe()['profile'])
            if changed and not (folder/'state.json').exists():
                # Planning has not installed anything. Preserve the old plan and
                # let the user correct the endpoint/Python before installation.
                backup=path.with_name(f'resolved-recipe.plan-{time.time_ns()}.json')
                path.rename(backup)
            elif changed:
                raise SetupError('保存された導入計画と環境・方式が異なります。既存記録を保全して確認してください')
            else:
                validate_resolved_recipe(saved['recipe'],new)
                validate_source_layout(saved['recipe'],cancelled,progress)
                return saved['recipe']
        from .setup_latest import resolve_sources
        from .setup_packages import resolve_packages
        resolved = resolve_sources(recipe(), default_cache().parent/'setup-resolution', cancelled, progress)
        validate_source_layout(resolved,cancelled,progress)
        if new:
            resolved = resolve_packages(resolved, sys.version_info[:3])
        else:
            # Match the chosen shared core instead of upgrading its kernels to
            # the requirements of a different ComfyUI release.
            from .setup_runtime import distributions
            from .setup_latest import _pins
            existing = distributions(connection)
            from .setup_packages import require_secure_torch
            require_secure_torch(existing.get('torch',''))
            resolved['requirements'], pins = _pins(
                (connection.root/'requirements.txt').read_text(encoding='utf-8'), resolved['requirements'])
            resolved['requirements'] = [
                'triton-windows=='+existing['triton-windows'] if r.startswith('triton-windows==') and 'triton-windows' in existing else r
                for r in resolved['requirements']]
            resolved['resolution']['existing_environment'] = {'versions_preserved': True, 'kernel_pins': pins}
        if cancelled():raise KeyboardInterrupt
        validate_resolved_recipe(resolved,new)
        atomic_json(path, {'schema_version':1, 'root':str(connection.root), 'new':new,
                           'connection':connection.to_dict(),
                           'python_abi':list(sys.version_info[:2]), 'recipe':resolved})
        return resolved


def state_directory(connection):
    key=sha256((str(connection.root).casefold()+'\0'+recipe()['profile']).encode()).hexdigest()[:24]
    return default_cache().parent/'setup-state'/key


def lora_profile_from_identity(value):
    """Match immutable model identity; display URLs are not execution pins."""
    if not isinstance(value,dict):return None
    from .lora_setup import model
    for profile in ('unlocked_v2','restoration'):
        selected=model(profile)
        expected={'id':selected.id,'model_id':selected.model_id,'version_id':selected.version_id,
                  'file_id':selected.file_id,'sha256':selected.sha256}
        if all(value.get(k)==v for k,v in expected.items()):return profile
    return None


def installed_lora_profile(connection=None):
    """Use the completed setup choice for new videos; never change saved jobs."""
    if connection is None:
        from .runtime_settings import RUNTIME
        connection=RUNTIME
    path=state_directory(connection)/'state.json'
    if not path.exists():return 'unlocked_v2'
    try:
        state=json.loads(path.read_text(encoding='utf-8'))
        if (not isinstance(state,dict) or state.get('root')!=str(connection.root)
                or state.get('profile')!=recipe()['profile']):raise ValueError()
        selected=state.get('completed_main_lora')
        if selected is None and state.get('state')=='dependencies_ready':selected=state.get('main_lora')
        if selected is None:return 'unlocked_v2'
        profile=lora_profile_from_identity(selected)
        if profile:return profile
    except (OSError,ValueError,TypeError):
        pass
    raise SetupError('導入済みLoRAの選択記録を確認できません。セットアップ画面で選択した環境と主LoRAを確認してください')


def validate_source_layout(data,cancelled=lambda:False,progress=lambda **kw:None):
    """Reject missing required source files before creating a GPU environment."""
    from .setup_runtime import REQUIRED_FILES
    from .setup_archives import acquire
    for source in data['sources']:
        prefix=source['target'].rstrip('/')+'/'
        required=[name if source['target']=='.' else name[len(prefix):]
                  for name in REQUIRED_FILES
                  if (not name.startswith('custom_nodes/') if source['target']=='.' else name.startswith(prefix))]
        if not required:continue
        archive=acquire(source,cancelled,progress)
        with zipfile.ZipFile(archive) as stream:
            names={name.split('/',1)[1] for name in stream.namelist() if '/' in name and not name.endswith('/')}
        missing=[name for name in required if name not in names]
        if missing:
            raise SetupError('取得したコードに必要なファイルがありません。GPU依存の導入前に停止します: '
                             +source['repo']+' / '+', '.join(missing))


def kernel_requirements(data):
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    names={'triton-windows','comfy-kitchen','comfy-aimdo'}
    pins={}
    for text in data['requirements']:
        value=Requirement(text);name=canonicalize_name(value.name)
        if name in names:
            specs=list(value.specifier)
            if name in pins or value.url or value.marker or len(specs)!=1 or specs[0].operator!='==' or '*' in specs[0].version:
                raise SetupError('GPUカーネルには重複のないexact pinが必要です: '+name)
            pins[name]=name+'=='+specs[0].version
    if set(pins)!=names:raise SetupError('必要なGPUカーネルのpinが不足しています')
    return [pins[name] for name in sorted(pins)]


def validate_resolved_recipe(data,new):
    """Recheck persisted choices against the current security floor, without I/O."""
    from .setup_packages import require_secure_torch, _official
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    from .setup_latest import ALLOWED_REPOSITORIES
    try:
        kernel_requirements(data)
        if not data.get('resolution'):raise ValueError()
        expected={(s['repo'],s['target']) for s in recipe()['sources']}
        if ({(s['repo'],s['target']) for s in data['sources']}!=expected
                or len(data['sources'])!=len(expected)
                or data['sources'][0]['repo']!='Comfy-Org/ComfyUI'):raise ValueError()
        for source in data['sources']:
            revision=source['revision']
            if (source['repo'] not in ALLOWED_REPOSITORIES or not re.fullmatch('[a-f0-9]{40}',revision)
                    or source['url']!='https://codeload.github.com/'+source['repo']+'/zip/'+revision
                    or not re.fullmatch('[a-f0-9]{64}',source['sha256']) or type(source['size_bytes']) is not int
                    or source['size_bytes']<=0):raise ValueError()
        tool=data['ffmpeg']
        from urllib.parse import quote
        expected='https://github.com/GyanD/codexffmpeg/releases/download/'+quote(tool['tag'],safe='')+'/ffmpeg-'+tool['tag'].removeprefix('v')+'-essentials_build.zip'
        if tool['url']!=expected or not re.fullmatch('[a-f0-9]{64}',tool['sha256']) or tool['size_bytes']<=0:raise ValueError()
        requirements={canonicalize_name(r.name):r for r in map(Requirement,data['gpu_requirements']+kernel_requirements(data))}
        torch=list(requirements['torch'].specifier)
        if len(torch)!=1 or torch[0].operator!='==':raise ValueError()
        require_secure_torch(torch[0].version)
        if new:
            packages=data['resolution']['packages']
            for name in ('torch','torchvision','torchaudio','triton-windows','pip'):
                row=packages[name]
                if not _official(row['url']) or not re.fullmatch('[a-f0-9]{64}',row['sha256']):raise ValueError()
                if name!='pip' and row['version'] not in requirements[name].specifier:raise ValueError()
            if packages['pip']['version']!=data['bootstrap_pip_version']:raise ValueError()
    except (KeyError,TypeError,ValueError) as error:
        if isinstance(error,SetupError):raise
        raise SetupError('保存された導入版のURL/hash/pinを検証できません。旧版へ戻さず停止します') from None


def discovered_servers():
    """Only process metadata; don't probe, stop, or read any media."""
    import psutil
    result=[]
    for p in psutil.process_iter(['pid','name','cmdline','cwd']):
        try:info=p.info
        except (psutil.NoSuchProcess,psutil.AccessDenied):continue
        args=info.get('cmdline') or []
        if not info.get('cwd') or not any(Path(a).name.casefold()=='main.py' for a in args): continue
        cwd=Path(info['cwd']).resolve()
        roots=[(Path(a) if Path(a).is_absolute() else cwd/a).resolve().parent
               for a in args if Path(a).name.casefold()=='main.py']
        root=next((r for r in roots if (r/'comfy').is_dir() and (r/'main.py').is_file()),None)
        if root is None:continue
        if any(r['root']==str(root) for r in result): continue
        try:
            python=Path(p.exe()).resolve()
            parent=p.parent()
            parent_python=Path(parent.exe()).resolve() if parent else None
            expected=root/'.venv/Scripts/python.exe'
            if expected in (python,parent_python):python=expected
            port=int(args[args.index('--port')+1]) if '--port' in args else 8188
        except (ValueError,IndexError,psutil.NoSuchProcess,psutil.AccessDenied):
            python=root/'.venv/Scripts/python.exe';port=8188
        result.append({'pid':info['pid'],'root':str(root),'python':str(python),'endpoint':f'http://127.0.0.1:{port}'})
    return result


def plan(connection, new=False, cancelled=lambda:False, progress=lambda **kw:None, lora_profile='unlocked_v2'):
    result = _plan(connection,new,lora_profile)
    if _ACTIVE_RECIPE.get() is not None or not result['hardware']['can_install']:
        return result
    resolved = resolve_recipe(connection,new,cancelled,progress)
    with using_recipe(resolved):
        result = _plan(connection,new,lora_profile)
    result['resolution'] = resolved['resolution']
    result['snapshot_reused_on_resume'] = True
    return result


def _plan(connection, new=False, lora_profile='unlocked_v2'):
    from .setup_runtime import idle, resources
    data=recipe()
    from .lora_setup import model as lora_model
    selected = lora_model(lora_profile)
    state_file=state_directory(connection)/'state.json'
    state=json.loads(state_file.read_text(encoding='utf-8')) if state_file.exists() else None
    if state and (state.get('root')!=str(connection.root) or state.get('profile')!=data['profile']):
        raise SetupError('以前の導入構成と異なります。記録と既存環境を保全して確認してください')
    if new:
        running=discovered_servers()
        if any(Path(row['root'])!=connection.root for row in running):
            raise SetupError('既存のComfyUIが動いています。新設せず、その環境を選択してください')
        if connection.root.exists() and any(connection.root.iterdir()) and not (state and state.get('managed_new')):
            raise SetupError('新規導入先には空の場所を指定してください。既存ファイルを上書きしません')
        # Port ownership is checked even before main.py/Python exist.
        import psutil
        listeners=[c for c in psutil.net_connections('tcp') if c.status=='LISTEN' and c.laddr.port==connection.port]
        if listeners:
            if not (state and state.get('managed_new')): raise SetupError('指定APIポートは使用中です。別サーバーを止めず、既存環境を選択してください')
            idle(connection.validate())
    else:
        connection.validate()
        idle(connection)
    # A new install reserves both retained model cache and installed weights.
    missing=sum(m.size_bytes for m,t in resources(connection,lora_profile) if not m.destination(t).exists())/1024**3
    storage=storage_check([(connection.root,(25 if new else 5)+missing),
                           (default_cache(),(15 if new else 5)+missing)])
    hardware=preflight(connection.root,0)
    for volume in storage:
        if volume['free_gib']<volume['required_gib']:
            hardware['blockers'].append(f"{volume['path']} に約{volume['required_gib']:.0f}GiBの空きが必要です（取得cacheを含む）")
    hardware['storage']=storage
    hardware['can_install']=not hardware['blockers']
    return {'mode':'new' if new else 'existing','profile':data['profile'],'root':str(connection.root),
            'python':str(connection.python),'endpoint':connection.endpoint,'hardware':hardware,
            'models':len(resources(connection,lora_profile)), 'main_lora':lora_profile,
            'lora_model':{'id':selected.id, 'model_id':selected.model_id, 'version_id':selected.version_id,
                          'file_id':selected.file_id, 'sha256':selected.sha256, 'terms_url':selected.terms_url},
            'civitai_authentication_required':True,
            'model_terms_review_required':True,
            'core_update':False,'generation_or_benchmark':False,
            'download_licenses':[{'repo':s['repo'],'revision':s['revision']} for s in data['sources']],
            'driver_install':False}


def run_command(args,cwd,log,cancelled=lambda:False,progress=lambda **kw:None):
    """Child command only. Stdout/stderr remain in private setup logs."""
    if cancelled():raise KeyboardInterrupt
    log=Path(log); log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a',encoding='utf-8') as output:
        args=[str(a) for a in args]
        if '-m' in args and args[args.index('-m')+1:args.index('-m')+2]==['pip'] and '--isolated' not in args:
            args.insert(args.index('-m')+2,'--isolated')
        from .setup_runtime import pip_environment
        p=subprocess.Popen(args,cwd=cwd,stdout=output,stderr=subprocess.STDOUT,
                env=pip_environment(),
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        started=time.monotonic()
        while p.poll() is None:
            if cancelled() or time.monotonic()-started>3600:
                import psutil
                try:
                    owned=psutil.Process(p.pid)
                    children=owned.children(recursive=True)
                    for c in reversed(children):
                        try:c.terminate()
                        except psutil.NoSuchProcess:pass
                    owned.terminate()
                    _,alive=psutil.wait_procs([*children,owned],timeout=5)
                    for c in alive:c.kill()
                except psutil.NoSuchProcess:pass
                p.wait()
                if cancelled():raise KeyboardInterrupt
                raise SetupError('導入コマンドが時間制限を超えました。ログ: '+str(log))
            time.sleep(.1)
    if p.returncode:
        raise SetupError(f'導入コマンドが失敗しました（終了値{p.returncode}）。途中データは保持します。ログ: {log}')


def patch_core(connection):
    spec=recipe()['core_patch']
    if spec is None:return {'state':'upstream','local_patch':False}
    path=connection.root/spec['target']
    content=path.read_bytes().replace(b'\r\n',b'\n')
    digest=sha256(content).hexdigest()
    if digest in [spec['after_sha256'],*spec.get('accepted_existing_sha256',[])]:
        return {'state':'verified','sha256':digest}
    if digest!=spec['before_sha256']:
        raise SetupError('H3 VAEのコードが固定版と異なります。利用者の変更を上書きしません')
    source=content.decode('utf-8')
    for anchor,addition in spec['inserts']:
        if source.count(anchor)!=1:raise SetupError('H3 VAEの互換差分を適用できません')
        source=source.replace(anchor,anchor+addition)
    after=source.encode()
    if sha256(after).hexdigest()!=spec['after_sha256']:raise SetupError('H3 VAE差分のhashが一致しません')
    backup=state_directory(connection)/'backup/comfy-sd.py'
    backup.parent.mkdir(parents=True,exist_ok=True)
    if backup.exists() and backup.read_bytes()!=path.read_bytes():raise SetupError('既存backupが異なります')
    if not backup.exists():shutil.copyfile(path,backup)
    # No race with a newly started/queued ComfyUI.
    from .setup_runtime import idle
    idle(connection)
    if sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=spec['before_sha256']:
        raise SetupError('互換差分の確認後にコードが変更されました')
    temporary=path.with_name(path.name+'.mosaic-setup.partial')
    owned=False
    try:
        with temporary.open('xb') as f:
            owned=True;f.write(after)
        os.replace(temporary,path)
    finally:
        if owned and temporary.exists():
            temporary.unlink()
    return {'state':'patched','sha256':spec['after_sha256'],'backup':str(backup)}


def install_nodes(connection,cancelled=lambda:False,progress=lambda **kw:None,managed_new=False):
    from .setup_runtime import idle
    result=[]
    for spec in recipe()['sources'][1:]:
        idle(connection)
        target=connection.root/spec['target']
        progress(phase='必要ノードを導入',model=spec['id'])
        if target.exists():
            if not target.is_dir():raise SetupError('必要ノードの配置先がdirectoryではありません: '+str(target))
            if managed_new:
                if connection.root not in target.resolve().parents:
                    raise SetupError('管理下ノードの配置先が選択rootの外です')
                result.append(verify_published(spec,target))
                continue
            # The user may own a compatible older/Manager installation. Preserve
            # it and verify required registrations before model acquisition.
            result.append({'target':spec['target'],'state':'preserved-existing',
                           'node_registration_check_required':True})
        else:
            if connection.root not in target.resolve().parents:raise SetupError('必要ノードの配置先が選択rootの外です')
            result.append(publish_archive(spec,target,cancelled,progress))
    managed=list(recipe()['managed_nodes'])
    if not (connection.root/'custom_nodes/ComfyUI-Unload-Model').exists():
        managed.append({'source':'mosaic_h3_memory.py','target':'custom_nodes/mosaic_h3_memory.py'})
    for spec in managed:
        if cancelled():raise KeyboardInterrupt
        source=ASSETS/spec['source']; target=connection.root/spec['target']
        if connection.root not in target.resolve().parents:raise SetupError('adapterの配置先が選択rootの外です')
        content=source.read_bytes().replace(b'\r\n',b'\n')
        if spec.get('sha256') and sha256(content).hexdigest()!=spec['sha256']:
            raise SetupError('同梱adapterのhashが不一致です')
        if target.exists():
            if target.read_bytes().replace(b'\r\n',b'\n')!=content:
                raise SetupError('同名adapterの変更があります。上書きしません: '+str(target))
        else:
            idle(connection)
            with target.open('xb') as f:f.write(content)
        result.append({'file':spec['target'],'sha256':sha256(content).hexdigest()})
    return result


def install_tools(cancelled=lambda:False,progress=lambda **kw:None):
    """Versioned app-owned tools; never overwrite an earlier installation."""
    spec=recipe()['ffmpeg']
    target=default_cache().parent/'tools/ffmpeg-versions'/spec['sha256']
    receipt=target/'.mosaic-tools.json'
    if target.exists() and receipt.is_file():
        record=json.loads(receipt.read_text(encoding='utf-8'))
        if record['archive_sha256']!=spec['sha256']:raise SetupError('既存FFmpegの版が異なります')
        for name,digest in record['executables'].items():
            if checked_hash(target/name,cancelled)!=digest:raise SetupError('FFmpeg/ffprobeが変更されています')
        atomic_json(default_cache().parent/'tools/ffmpeg-active.json',{'sha256':spec['sha256']})
        return {'state':'verified','bin':str(target/'bin')}
    if target.exists():verify_published(spec,target)
    else:publish_archive(spec,target,cancelled,progress)
    executables={}
    for name in ('bin/ffmpeg.exe','bin/ffprobe.exe'):
        path=target/name
        if not path.is_file():raise SetupError('取得したFFmpegに必要な実行ファイルがありません')
        executables[name]=checked_hash(path,cancelled)
    atomic_json(receipt,{'archive_sha256':spec['sha256'],'executables':executables,
                        'license':spec['license'],'source':spec['url']})
    atomic_json(default_cache().parent/'tools/ffmpeg-active.json',{'sha256':spec['sha256']})
    return {'state':'installed','bin':str(target/'bin')}


def validate_lora(connection,source=None,cancelled=lambda:False,lora_profile='unlocked_v2'):
    from .lora_setup import model
    if source is not None:
        raise SetupError('必須LoRAは本人のCivitai API認証で取得してください。手動ファイルの取り込みには対応していません')
    # Validate the pinned selection before download; the file need not exist yet.
    return model(lora_profile)


def create_core(connection,state,cancelled,progress):
    spec=recipe()['sources'][0]
    if connection.root.exists() and (connection.root/'main.py').is_file():
        patch=recipe().get('core_patch')
        allowed={patch['target']:[patch['after_sha256']]} if patch else None
        return verify_published(spec,connection.root,allowed)
    if connection.root.exists():
        if any(connection.root.iterdir()):raise SetupError('新規導入先に既存ファイルがあります')
        connection.root.rmdir()  # A verified empty selected directory only.
    return publish_archive(spec,connection.root,cancelled,progress)


def create_gpu_environment(connection,cancelled,progress,log):
    venv=connection.root/'.venv'
    expected=venv/'Scripts/python.exe'
    if connection.python!=expected.resolve():raise SetupError('新規導入のPythonは選択rootの.venvへ固定します')
    owner={'schema_version':1,'root':str(connection.root),'python':str(connection.python),
           'base_python':str(Path(sys.executable).resolve()),'base_abi':list(sys.version_info[:2])}
    marker=venv/'.mosaic-venv.json'
    if not venv.exists():
        # Publish an exclusive ownership marker together with the new directory.
        import tempfile
        with tempfile.TemporaryDirectory(prefix='.mosaic-venv-',dir=connection.root) as temporary:
            payload=Path(temporary)/'payload';payload.mkdir()
            atomic_json(payload/'.mosaic-venv.json',owner)
            os.rename(payload,venv)
    if not marker.is_file() or json.loads(marker.read_text(encoding='utf-8'))!=owner:
        raise SetupError('所有を確認できない.venvがあります。上書きせず停止します')
    if not expected.exists():
        run_command([sys.executable,'-B','-m','venv',venv],connection.root,log,cancelled,progress)
    data=recipe()
    from .setup_runtime import distributions
    before=distributions(connection)
    for requirement in data['gpu_requirements']+kernel_requirements(data):
        name,version=requirement.split('==',1)
        if name in before and before[name]!=version:
            raise SetupError('導入済みGPU依存の版が構成と異なります。上書きしません: '+name)
    progress(phase='新規環境のpipを更新',model='pip')
    packages=data.get('resolution',{}).get('packages',{})
    if 'pip' not in packages:raise SetupError('解決済み公式pip wheelのURL/hashが不足しています')
    item=packages['pip']
    run_command([expected,'-B','-m','pip','install','--only-binary=:all:','--upgrade',
                 'pip @ '+item['url']+'#sha256='+item['sha256'],'--index-url','https://pypi.org/simple'],
                connection.root,log,cancelled,progress)
    from .setup_wheels import install_locked_dependencies
    result=install_locked_dependencies(connection,data,state_directory(connection),run_command,cancelled,progress,log)
    return {**result,'torch':data['gpu_requirements'],'kernel_requirements':kernel_requirements(data),
            'python':str(expected)}


def write_launcher(connection,hardware):
    # Only app-owned/new environments receive a launcher. Existing ones keep theirs.
    if connection.endpoint!='http://127.0.0.1:'+str(connection.port):
        raise SetupError('新規導入のAPIはhttp://127.0.0.1:<port>を指定してください')
    total=next(g for g in hardware['hardware']['gpus'] if g['index']==0)['vram_gib']
    reserve='13' if total>=23.9 else ('3' if total>=15.9 else '1')
    headroom='5' if total>=23.9 else '2'
    args=['-X','utf8','main.py','--listen','127.0.0.1','--port',str(connection.port),
          '--disable-pinned-memory','--fp16-intermediates','--reserve-vram',reserve,
          '--vram-headroom',headroom,'--disable-cuda-malloc','--disable-comfy-compiler','--cache-none']
    path=connection.root/'mosaic-launch.json'
    content={'python':str(connection.python),'root':str(connection.root),'args':args}
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=content:
        raise SetupError('既存の起動設定が異なります。上書きしません')
    atomic_json(path,content)
    return content


def verify_gpu(connection,cancelled=lambda:False):
    """Tiny synthetic kernel check only, before models. No weights or media."""
    from .setup_runtime import idle, run_python
    idle(connection)
    folder=state_directory(connection)/'kernel-probe'
    folder.mkdir(parents=True,exist_ok=True)
    output=run_python(connection.python,[str(ASSETS/'gpu_probe.py'),str(folder)],connection.root,cancelled)
    return json.loads(output.strip().splitlines()[-1])


def start_managed(connection,cancelled=lambda:False):
    from .setup_runtime import idle
    import psutil
    listeners=[c for c in psutil.net_connections('tcp') if c.status=='LISTEN' and c.laddr.port==connection.port]
    if listeners:
        # Opening the application must also work while its server is processing.
        # Verify ownership without touching its queue or allocating another GPU.
        from .runtime_observation import identity
        identity(connection)
        return {'state':'already_running'}
    status=idle(connection)
    if status=='server_idle':return {'state':'already_running'}
    cfg=json.loads((connection.root/'mosaic-launch.json').read_text(encoding='utf-8'))
    state_file=state_directory(connection)/'state.json'
    journal=json.loads(state_file.read_text(encoding='utf-8')) if state_file.is_file() else {}
    if not journal.get('managed_new'):
        raise SetupError('アプリが新規導入した環境だけを起動できます')
    if cfg.get('python')!=str(connection.python) or cfg.get('root')!=str(connection.root):
        raise SetupError('起動設定と選択環境が異なります')
    if journal.get('stages',{}).get('起動設定を保存')!=cfg:
        raise SetupError('保存後に起動設定が変更されています。上書きせず停止します')
    if cancelled():raise KeyboardInterrupt
    log=state_directory(connection)/'comfy-startup.log'
    from .runtime_settings import local_tool_bin
    toolbin=local_tool_bin()
    with log.open('a',encoding='utf-8') as stream:
        p=subprocess.Popen([str(connection.python),*cfg['args']],cwd=connection.root,stdout=stream,stderr=subprocess.STDOUT,
                env={**os.environ,'PYTHONUTF8':'1','PATH':(str(toolbin)+os.pathsep if toolbin else '')+os.environ.get('PATH','')},
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    from .runtime_http import urlopen
    for _ in range(120):
        if cancelled():raise KeyboardInterrupt
        if p.poll() is not None:raise SetupError('ComfyUIが起動できませんでした。ログ: '+str(log))
        try:
            with urlopen(connection.endpoint+'/system_stats',timeout=1) as r:json.load(r)
        except OSError:
            time.sleep(1);continue
        idle(connection) # A racing/unrelated owner fails immediately; never stop it.
        return {'state':'started','launcher_pid':p.pid,'log':str(log)}
    raise SetupError('ComfyUIの起動を確認できません。所有processとログを確認してください: '+str(log))


def install_all(connection,new=False,accept_terms=False,acknowledge_warnings=False,lora_file=None,
                cancelled=lambda:False,progress=lambda **kw:None,credentials=None,lora_profile='unlocked_v2'):
    if not accept_terms:raise SetupError('コード・モデル・FFmpegの配布元の利用条件を確認してください')
    validate_lora(connection,lora_file,cancelled,lora_profile)
    from .lora_setup import preflight_selection
    credentials = preflight_selection(lora_profile,credentials)
    proposed=plan(connection,new,cancelled,progress,lora_profile)
    if not proposed['hardware']['can_install']:raise SetupError('\n'.join(proposed['hardware']['blockers']))
    resolved=resolve_recipe(connection,new,cancelled,progress)
    with using_recipe(resolved):
        return _install_all(connection,new,accept_terms,acknowledge_warnings,lora_file,cancelled,progress,credentials,lora_profile)


def _install_all(connection,new=False,accept_terms=False,acknowledge_warnings=False,lora_file=None,
                 cancelled=lambda:False,progress=lambda **kw:None,credentials=None,lora_profile='unlocked_v2'):
    """One explicit action, resumable journal; no video or model inference."""
    from .setup_runtime import idle, acquire_resources, inspect_nodes, prepare_libraries
    if not accept_terms:raise SetupError('コード・モデル・FFmpegの配布元の利用条件を確認してください')
    proposed=plan(connection,new,lora_profile=lora_profile)
    hw=proposed['hardware']
    if not hw['can_install']:raise SetupError('\n'.join(hw['blockers']))
    if hw['requires_warning_acknowledgement'] and not acknowledge_warnings:
        raise SetupError('\n'.join(hw['warnings'])+'\n未検証条件を確認してから開始してください')
    folder=state_directory(connection); folder.mkdir(parents=True,exist_ok=True)
    with job_lock(folder):
        plan(connection,new,lora_profile=lora_profile) # Recheck ownership/queue after obtaining the writer lock.
        path=folder/'state.json'
        state=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {
            'root':str(connection.root),'profile':recipe()['profile'],'managed_new':new,'stages':{},'started_at':time.time()}
        if bool(state['managed_new'])!=new:raise SetupError('前回と導入方式が異なります。記録を確認してください')
        previous = state.get('main_lora')
        if (previous is not None and lora_profile_from_identity(previous)!=lora_profile_from_identity(proposed['lora_model'])
                and state.get('state')!='dependencies_ready'):
            old=lora_profile_from_identity(previous) or '固定版不明'
            raise SetupError('前回の未完成導入と必須LoRAの固定版が異なります。前回の選択（'+old+'）で再開してください')
        if state.get('state')=='dependencies_ready' and previous is not None:
            state.setdefault('completed_main_lora',previous)
        state['main_lora'] = proposed['lora_model']
        atomic_json(path,{**state,'state':'installing'})
        def stage(name,action):
            if cancelled():raise KeyboardInterrupt
            progress(phase=name,model='setup')
            if cancelled():raise KeyboardInterrupt
            state['state']='installing';state['phase']=name;atomic_json(path,state)
            result=action()
            state['stages'][name]=result;atomic_json(path,state)
            return result
        try:
            if new:
                from .setup_runtime import idle
                if connection.python.exists():idle(connection)
                state['core']=stage('ComfyUI本体を導入',lambda:create_core(connection,state,cancelled,progress))
                atomic_json(path,state)
                stage('GPU依存を導入',lambda:create_gpu_environment(connection,cancelled,progress,folder/'libraries.log'))
            connection.validate()
            idle(connection)
            # Core changes are only for the newly owned environment.
            if new:stage('H3 VAEの互換設定',lambda:patch_core(connection))
            stage('必要ノードを導入',lambda:install_nodes(connection,cancelled,progress,managed_new=new))
            if not new:stage('不足ライブラリを導入',lambda:prepare_libraries(connection,apply=True,cancelled=cancelled))
            stage('GPUとカーネルを検査',lambda:verify_gpu(connection,cancelled))
            stage('FFmpegを導入',lambda:install_tools(cancelled,progress))
            if new:
                stage('起動設定を保存',lambda:write_launcher(connection,hw))
                stage('ComfyUIを起動',lambda:start_managed(connection,cancelled))
            result=stage('必要ノードの登録を確認',lambda:inspect_nodes(connection))
            if result.get('missing_runtime_files'):
                raise SetupError('必要な本体・ノードファイルがありません。再起動だけでは解決しません。環境を保全して確認してください: '
                                 +', '.join(result['missing_runtime_files']))
            if not result['nodes_verified']:
                state.update(state='restart_required',next='選択した既存ComfyUIを通常の方法で再起動し、この導入操作をもう一度実行してください')
                atomic_json(path,state);return state
            stage('不足モデルを取得',lambda:acquire_resources(connection,cancelled,progress,credentials=credentials,lora_profile=lora_profile,accept_terms=accept_terms))
            stage('環境の選択を保存',lambda:{'settings':str(save(connection))})
            state.update(state='dependencies_ready',completed_main_lora=proposed['lora_model'],finished_at=time.time(),generation_tested=False,
                         benchmark_resumed=False,hardware=hw)
            atomic_json(path,state)
            return state
        except BaseException as exc:
            state.update(state='cancelled' if isinstance(exc,KeyboardInterrupt) else 'failed',
                         error=str(exc),automatic_resume=False,updated_at=time.time())
            atomic_json(path,state)
            raise
