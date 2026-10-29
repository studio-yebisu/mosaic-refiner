"""Pinned restoration components, acquired only by an explicit setup operation."""
import os
from pathlib import Path
import shutil
import subprocess

from .checkpoint import digest_file
from .model_setup import Model, SetupError, default_cache, install

from .runtime_settings import RUNTIME
COMFY_ROOT = RUNTIME.root
COMFY_PYTHON = RUNTIME.python
RESTORATION_LORA = 'mosaic_refiner/DecensoreH3-v1.0.3.safetensors'
RESTORATION_LORA_SHA = '8c693023caa51063738cd961de1f4623ad274b003bcf6eedf677a7adbf456586'
RESTORATION_LORA_SIZE = 155110296
BASIC_PROFILE = 'lada-basicvsr-ema-v1.2-bgr-fp16-overlap-v2'
BASIC_CHUNK_FRAMES = 90
BASIC_HALO_FRAMES = 8


class BasicModel(Model):
    @property
    def url(self):
        return super().url + '?download=true'


BASIC = BasicModel('lada-basicvsr-generic-v1-2', 'ladaapp/lada',
    'bcf461d46d9a98981fc64b815df5178f42215cdf',
    'lada_mosaic_restoration_model_generic_v1.2.pth', 'basicvsr-v1.2.pth', 78441770,
    'd404152576ce64fb5b2f315c03062709dac4f5f8548934866cd01c823c8104ee',
    'AGPL-3.0', 'https://github.com/ladaapp/lada/blob/20cb34a20a83c72c87a991d2c949032c70085b16/LICENSE.md', '2026-10-09')


def model_directory():
    return Path(os.environ.get('LOCALAPPDATA', Path.home() / '.cache')) / 'MosaicRefiner/restoration'


def basic_path():
    installed = BASIC.destination(model_directory())
    development = Path(__file__).resolve().parents[1] / 'models/basicvsr-v1.2.pth'
    return installed if installed.is_file() else development


def lora_path():
    return COMFY_ROOT / 'models/loras' / RESTORATION_LORA


def _verified(path, size, expected):
    return path.is_file() and path.stat().st_size == size and digest_file(path) == expected


def ready(lora_profile='unlocked_v2'):
    from .h3_lora_profiles import selected_lora
    selected=selected_lora(lora_profile)
    # Model presence alone does not resolve the pending LoRA conditions.
    return False


def identity(processing_options=None):
    if not _verified(basic_path(), BASIC.size_bytes, BASIC.sha256):
        raise SetupError('モザイク補正用BasicVSR++が未導入またはhash不一致です。「補正モデルを導入」を実行してください')
    from .processing_options import resolve_options
    from .h3_lora_profiles import selected_lora
    from .h3_profile import require_public_options
    options = resolve_options(processing_options)
    require_public_options(options)
    lora = selected_lora(options.lora_profile, verify=True)
    return {'method': 'basicvsr-ema-to-official-h3-required-lora-v1',
            'basicvsr_profile': BASIC_PROFILE, 'chunk_frames': BASIC_CHUNK_FRAMES,
            'halo_frames': BASIC_HALO_FRAMES, 'temporal_blend': 'linear-overlap',
            'basicvsr_sha256': BASIC.sha256, 'basicvsr_weights': 'generator_ema',
            'basicvsr_source_revision': '20cb34a20a83c72c87a991d2c949032c70085b16',
            'lora': lora['lora'], 'lora_sha256': lora['lora_sha256'],
            'lora_profile': options.lora_profile, 'lora_strength': options.lora_strength,
            'lora_enabled': options.lora_strength > 0, 'quality_adopted': False}


def setup(cancelled=lambda: False, progress=lambda **kw: None, lora_profile='unlocked_v2'):
    """Explicit setup; unresolved LoRA conditions stop before file or network writes."""
    from .h3_lora_profiles import selected_lora
    selected_lora(lora_profile, verify=True)
    directory = model_directory()
    if _verified(basic_path(), BASIC.size_bytes, BASIC.sha256):
        target = BASIC.destination(directory)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with basic_path().open('rb') as source, target.open('xb') as output:
                shutil.copyfileobj(source, output)
    else:
        install(directory, default_cache(), models=[BASIC], cancelled=cancelled, progress=progress)
    return identity({'lora_profile': lora_profile})
