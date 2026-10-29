"""Pinned required LoRAs and explicit per-user Civitai authentication; no import I/O."""
from .h3_lora_profiles import H3_TERMS_SHA256, H3_TERMS_URL, PROFILES
from .model_auth import AuthError, Credentials, check, validate_token
from .model_setup import Model, SetupError


def model(profile='unlocked_v2'):
    if profile not in ('unlocked_v2', 'restoration'):
        raise SetupError('必須LoRAはUnlocked V2 / DecensoreH3から選択してください')
    selected = PROFILES[profile]
    return Model(selected['id'], '', '', selected['filename'], 'loras/' + selected['lora'],
                 selected['size_bytes'], selected['lora_sha256'],
                 selected['creator'] + ' / Civitai作者の利用条件 + H3条件',
                 selected['terms_url'], selected['source_checked_at'], provider='civitai',
                 model_id=selected['model_id'], version_id=selected['version_id'], file_id=selected['file_id'])


def conditions_text(profile='unlocked_v2'):
    """Read-only source attribution. A displayed condition is not a license grant."""
    model(profile)  # Reject historical/unknown profiles consistently with acquisition.
    selected = PROFILES[profile]
    return '\n'.join((
        f"{selected['label']} / 作者: {selected['creator']}（credit要）",
        f"固定版: model {selected['model_id']} / version {selected['version_id']} / file {selected['file_id']}",
        '作者: ' + selected['creator_url'],
        '作者の配布ページ・条件表示: ' + selected['terms_url'],
        '基盤条件: ' + H3_TERMS_URL,
        '照合したH3条件のSHA-256: ' + H3_TERMS_SHA256,
        '利用前の確認事項: ' + selected['clearance_issue'],
        'Civitai上の配布・生成の契約と、取得後のローカル利用条件は別です。本人のAPI認証や確認チェックで許諾済みにはなりません。',
    ))


def require_user_key(credentials):
    """Pure presence/format check. Never load a vault, contact an API, or return a key."""
    if not isinstance(credentials, Credentials) or not credentials.tokens.get('civitai'):
        raise AuthError('必須LoRAの取得には本人のCivitai APIキーが必要です。セットアップ画面の「本人のAPI認証」で入力するか、本人の対話端末でauth login civitai --saveを実行してください')
    validate_token('civitai', credentials.tokens['civitai'])


def authenticated_credentials(credentials=None, opener=None):
    """Called only by an explicit acquisition/resume; no anonymous fallback."""
    credentials = Credentials.saved() if credentials is None else credentials
    require_user_key(credentials)
    check('civitai', credentials, opener)
    return credentials


def preflight_selection(profile='unlocked_v2', credentials=None, opener=None):
    """Authenticate and check the fixed file before acquiring any large dependency."""
    selected = model(profile)
    credentials = authenticated_credentials(credentials, opener)
    from .model_setup import download_url
    download_url(selected, credentials, opener)
    return credentials
