"""Pinned local H3 LoRA choices; no downloads or base-weight changes."""

from .model_setup import SetupError
from .restoration_models import (COMFY_ROOT, RESTORATION_LORA, RESTORATION_LORA_SHA,
                                 RESTORATION_LORA_SIZE)

UNLOCKED_LORA = "minimax_h3/Minimax_H3_Unlocked_V2.safetensors"
UNLOCKED_SHA = "b05686a14b96bf84102274b1c355558a2f41369753b472ddc0724c6f5147503b"
UNLOCKED_SIZE = 298257967
H3_TERMS_URL = 'https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/42ed227ee7df40d41602854ae760620d6eb651fe/LICENSE'
H3_TERMS_SHA256 = '59b99642b95ea21630e311198ddbfffbfe05aadba0c2f5d884cbdf4efcc90f44'
PROFILES = {
    'none': {'label': '公式H3由来・追加LoRAなし', 'lora': None,
             'lora_sha256': None, 'size_bytes': 0},
    "restoration": {"label": "DecensoreH3", "lora": RESTORATION_LORA,
                    "lora_sha256": RESTORATION_LORA_SHA, "size_bytes": RESTORATION_LORA_SIZE,
                    'id': 'decensore-h3-v1', 'filename': 'DecensoreH3_000002500.safetensors',
                    'model_id': 2990026, 'version_id': 3390424, 'file_id': 3280455,
                    'creator': 'herpderpmerp',
                    'creator_url': 'https://civitai.com/user/herpderpmerp',
                    'clearance_issue': '作者の許可本文と、H3条件を継承したローカル動画補正への許可範囲を確認待ちです',
                    'terms_url': 'https://civitai.com/models/2990026?modelVersionId=3390424',
                    'source_checked_at': '2026-10-27'},
    "unlocked_v2": {"label": "Minimax H3 Unlocked V2", "lora": UNLOCKED_LORA,
                    "lora_sha256": UNLOCKED_SHA, "size_bytes": UNLOCKED_SIZE,
                    'id': 'minimax-h3-unlocked-v2', 'filename': 'Minimax_H3_Unlocked_V2.safetensors',
                    'model_id': 2925727, 'version_id': 3354822, 'file_id': 3242253,
                    'creator': 'The_Midnight_Lab',
                    'creator_url': 'https://civitai.com/user/The_Midnight_Lab',
                    'clearance_issue': '作者が説明する安全措置の解除目的と、H3 Exhibit A §5との整合を確認待ちです',
                    'terms_url': 'https://civitai.com/models/2925727?modelVersionId=3354822',
                    'source_checked_at': '2026-10-27'},
}


def selected_lora(profile, verify=False):
    if profile not in PROFILES:
        raise ValueError("H3 LoRAの選択が不正です")
    result = {"profile": profile, **PROFILES[profile]}
    if verify:
        if profile == 'none':
            raise SetupError('追加LoRAは必須です。Unlocked V2またはDecensoreH3を選択してください')
        # Selection is a product requirement, not evidence of a license grant.
        raise SetupError('追加LoRAの許諾確認待ち: ' + result['clearance_issue'] + '。自動取得・推論は停止中です。docs/required-lora.mdを確認してください')
    return result


def available_profiles():
    return {name: {**PROFILES[name], 'available': False, 'required': True,
                   'license_status': 'unresolved', 'execution_enabled': False}
            for name in ('unlocked_v2', 'restoration')}
