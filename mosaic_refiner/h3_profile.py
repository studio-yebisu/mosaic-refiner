"""Reviewed public H3 file identities; importing never downloads a model."""
from .model_setup import catalog, SetupError
from pathlib import Path

PROFILE = 'official-h3-ref2va-int4-required-lora-v1'
ROLES = {'dit': 'official-ref2va-int4', 'clip': 'encoder',
         'vae': 'official-video-vae', 'audio_vae': 'audio-vae'}


def assets():
    models = {model.id: model for model in catalog()}
    if set(models) != set(ROLES.values()):
        raise SetupError('公開H3構成の台帳が一致しません。未知モデルへ代用しません')
    return {role: models[name] for role, name in ROLES.items()}


def loader_names():
    categories = {'dit': 'diffusion_models/', 'clip': 'text_encoders/', 'vae': 'vae/', 'audio_vae': 'vae/'}
    # ComfyUI's enum uses the host OS separator for relative model names.
    return {role: str(Path(model.target.removeprefix(categories[role]))) for role, model in assets().items()}


def require_public_options(options):
    if options.version != 3 or options.lora_profile not in ('unlocked_v2', 'restoration') or not 0 < options.lora_strength <= 2:
        raise SetupError('現行要件は追加LoRA必須です。旧記録を保持し、Unlocked V2またはDecensoreH3を選んで新しい処理を準備してください')


def require_public_request(request):
    if request.get('adapter_contract_version') != 4 or request.get('model_expected_sha256') != assets()['dit'].sha256:
        raise SetupError('入力bundleのモデル構成が公開用と異なります。旧記録を保全し、新しい入力を準備してください')
