"""Cheap metadata preflight; no CUDA context, benchmark, or media access."""
import csv
import io
import platform
from pathlib import Path
import shutil
import subprocess
import psutil

GIB = 1024**3
MIN_VRAM_GIB = 12
MIN_RAM_GIB = 32
RECOMMENDED_VRAM_GIB = 24
RECOMMENDED_RAM_GIB = 64
MIN_DRIVER_MAJOR = 580


def snapshot(target):
    ancestor = Path(target).resolve()
    while not ancestor.exists(): ancestor = ancestor.parent
    devices, error = [], None
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,driver_version,compute_cap',
                '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if result.returncode: raise ValueError('nvidia-smiがGPU情報を返しませんでした')
        for row in csv.reader(io.StringIO(result.stdout)):
            if len(row) != 5: raise ValueError('GPU情報の形式が不明です')
            index, name, mib, driver, capability = [s.strip() for s in row]
            devices.append({'index':int(index), 'name':name, 'vram_gib':float(mib)/1024,
                            'driver':driver, 'compute_capability':capability})
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        error = str(exc)
    return {'os':platform.system(), 'machine':platform.machine(), 'os_build':platform.version(),
            'ram_gib':psutil.virtual_memory().total/GIB,
            'disk_free_gib':shutil.disk_usage(ancestor).free/GIB, 'disk_path':str(ancestor),
            'gpus':devices, 'gpu_error':error}


def assess(data, required_disk_gib=100):
    blockers, warnings = [], []
    if data['os'] != 'Windows' or data['machine'].lower() not in ('amd64', 'x86_64'):
        blockers.append('自動導入は64bit Windows 10/11向けです。ARM、macOS、Linuxは対象外です')
    else:
        try: build=int(data.get('os_build','10.0.0').split('.')[-1])
        except ValueError:build=0
        if build<19045:blockers.append('Windows 10 22H2以降、またはWindows 11を用意してください')
    if data['ram_gib'] < MIN_RAM_GIB - 1:
        blockers.append('RAMは32GB以上が必要です。H3のCPU退避用メモリが不足します')
    elif data['ram_gib'] < RECOMMENDED_RAM_GIB - 1:
        warnings.append('RAMは64GBを推奨します。32〜63GBではメモリ不足やディスクへの退避で遅くなる可能性があります')
    if data['disk_free_gib'] < required_disk_gib:
        blockers.append(f'導入先に約{required_disk_gib:.0f}GB以上の空きが必要です。モデル・取得cache・依存を保存します')
    gpu = next((g for g in data['gpus'] if g['index']==0), None)
    if not gpu:
        blockers.append('NVIDIA GPUと動作するドライバーが必要です。AMD/Intel/CPU推論はこのH3導入構成の対象外です')
    else:
        try:
            cap = tuple(int(v) for v in gpu['compute_capability'].split('.'))
            driver = int(gpu['driver'].split('.')[0])
        except (ValueError, KeyError):
            blockers.append('GPU世代またはドライバーを判定できません。NVIDIAのドライバーを確認してください')
        else:
            if cap < (8,0): blockers.append('NVIDIA Ampere世代以降（compute capability 8.0以上）が必要です')
            if driver < MIN_DRIVER_MAJOR:
                blockers.append('CUDA 13.0用のNVIDIA R580以降のドライバーが必要です。公式ドライバーを先に導入してください')
        if gpu['vram_gib'] < MIN_VRAM_GIB - .1:
            blockers.append('VRAM 12GB未満はこのH3構成の自動導入対象外です。モデル取得を始めません')
        elif gpu['vram_gib'] < 16 - .1:
            warnings.append('VRAM 12〜15GBは未検証です。メモリ不足で停止する可能性が高く、画面サイズを下げても完走は保証できません')
        elif gpu['vram_gib'] < RECOMMENDED_VRAM_GIB - .1:
            warnings.append('VRAM 16〜23GBは未検証です。解像度・生成窓・他のGPU使用量によってメモリ不足や速度低下が起こりえます')
    return {'can_install':not blockers, 'requires_warning_acknowledgement':bool(warnings),
            'blockers':blockers, 'warnings':warnings, 'hardware':data,
            'tested_configuration':'RTX 4090 24GB・RAM 64GB。別GPU/12GB実機での完走は未確認',
            'minimums_are_setup_policy_not_measured_inference_guarantee':True}


def preflight(target, required_disk_gib=100):
    return assess(snapshot(target), required_disk_gib)


def storage_check(requirements):
    """Sum reservations on each volume, including the separate download cache."""
    volumes={}
    for path,needed in requirements:
        ancestor=Path(path).resolve()
        while not ancestor.exists():ancestor=ancestor.parent
        key=ancestor.anchor.casefold()
        volumes.setdefault(key,{'path':str(ancestor),'required_gib':0,'free_gib':shutil.disk_usage(ancestor).free/GIB})
        volumes[key]['required_gib']+=needed
    return list(volumes.values())
