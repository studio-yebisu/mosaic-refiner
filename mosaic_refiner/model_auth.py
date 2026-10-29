"""Per-user model access. No secret is written outside the OS credential vault."""
import ctypes
from ctypes import wintypes
from dataclasses import dataclass, field
import json
import os
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

PROVIDERS = {
    "huggingface": ("huggingface.co", "https://huggingface.co/api/whoami-v2",
                    "https://huggingface.co/settings/tokens"),
    "civitai": ("civitai.com", "https://civitai.com/api/v1/me",
                "https://civitai.com/user/account"),
}


class AuthError(ValueError):
    pass


def validate_token(provider, token):
    if provider not in PROVIDERS:
        raise AuthError("対応する配布元はhuggingface / civitaiです")
    if not isinstance(token, str) or not re.fullmatch(r"[A-Za-z0-9._-]{1,2000}", token):
        raise AuthError("認証情報の形式が不正です。公式設定ページから入力し直してください")
    return token


def official_url(provider, url):
    try:
        parsed = urlsplit(url)
        return (provider in PROVIDERS and parsed.scheme == "https"
                and parsed.hostname == PROVIDERS[provider][0]
                and parsed.port in (None, 443) and not parsed.username and not parsed.password)
    except (ValueError,TypeError,AttributeError):
        return False


class SafeRedirect(HTTPRedirectHandler):
    """Never forward credentials to a CDN, another origin, or plaintext HTTP."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        old, new = urlsplit(req.full_url), urlsplit(newurl)
        bearer=req.get_header("Authorization","").removeprefix("Bearer ")
        if len(bearer)>=12 and bearer in newurl:
            raise AuthError("認証情報を含む取得URLへの転送を拒否しました")
        if new.scheme != "https" or new.username or new.password:
            raise AuthError("安全でない取得先への転送を拒否しました")
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None:
            try:changed=(old.scheme, old.hostname, old.port or 443) != (new.scheme, new.hostname, new.port or 443)
            except ValueError:raise AuthError("取得先の形式が不正です") from None
            if changed:
                redirected.remove_header("Authorization")
                redirected.remove_header("Cookie")
        return redirected


def secure_open(request, timeout=30):
    return build_opener(SafeRedirect()).open(request, timeout=timeout)


class WindowsVault:
    """Generic credentials of the signed-in Windows user; no JSON token file."""
    def __init__(self, prefix="MosaicRefiner/model-auth/"):
        self.prefix = prefix

    def _api(self):
        if os.name != "nt":
            raise AuthError("認証情報の保存はWindows資格情報マネージャーでのみ利用できます")
        class Credential(ctypes.Structure):
            _fields_ = [
                ("Flags", wintypes.DWORD), ("Type", wintypes.DWORD),
                ("TargetName", wintypes.LPWSTR), ("Comment", wintypes.LPWSTR),
                ("LastWritten", wintypes.FILETIME), ("CredentialBlobSize", wintypes.DWORD),
                ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)), ("Persist", wintypes.DWORD),
                ("AttributeCount", wintypes.DWORD), ("Attributes", ctypes.c_void_p),
                ("TargetAlias", wintypes.LPWSTR), ("UserName", wintypes.LPWSTR),
            ]
        api = ctypes.WinDLL("advapi32", use_last_error=True)
        api.CredReadW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                 ctypes.POINTER(ctypes.POINTER(Credential))]
        api.CredReadW.restype = wintypes.BOOL
        api.CredWriteW.argtypes = [ctypes.POINTER(Credential), wintypes.DWORD]
        api.CredWriteW.restype = wintypes.BOOL
        api.CredDeleteW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD]
        api.CredDeleteW.restype = wintypes.BOOL
        api.CredFree.argtypes = [ctypes.c_void_p]
        api.CredFree.restype = None
        return api, Credential

    def load(self, provider):
        if provider not in PROVIDERS:
            raise AuthError("配布元が不正です")
        api, Credential = self._api()
        pointer = ctypes.POINTER(Credential)()
        if not api.CredReadW(self.prefix + provider, 1, 0, ctypes.byref(pointer)):
            if ctypes.get_last_error() == 1168:
                return None
            raise AuthError("OSから認証情報を読み取れません。資格情報マネージャーを確認してください")
        try:
            return validate_token(provider, ctypes.string_at(
                pointer.contents.CredentialBlob, pointer.contents.CredentialBlobSize).decode("utf-8"))
        finally:
            api.CredFree(pointer)

    def save(self, provider, token):
        token = validate_token(provider, token)
        api, Credential = self._api()
        data = token.encode("utf-8")
        blob = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
        record = Credential(Type=1, TargetName=self.prefix + provider,
                            CredentialBlobSize=len(data), CredentialBlob=blob, Persist=2,
                            UserName="MosaicRefiner")
        if not api.CredWriteW(ctypes.byref(record), 0):
            raise AuthError("OSへ認証情報を保存できません")

    def delete(self, provider):
        if provider not in PROVIDERS:
            raise AuthError("配布元が不正です")
        api, _ = self._api()
        if not api.CredDeleteW(self.prefix + provider, 1, 0) and ctypes.get_last_error() != 1168:
            raise AuthError("OSの認証情報を削除できません")


@dataclass(repr=False)
class Credentials:
    tokens: dict = field(default_factory=dict, repr=False)

    def __repr__(self):
        return "Credentials(providers=" + repr(sorted(self.tokens)) + ")"

    @classmethod
    def saved(cls, vault=None):
        vault = vault or WindowsVault()
        if os.name != "nt" and isinstance(vault, WindowsVault):
            return cls()
        return cls({p: token for p in PROVIDERS if (token := vault.load(p))})

    def set(self, provider, token):
        self.tokens[provider] = validate_token(provider, token)

    def headers(self, provider, url):
        token = self.tokens.get(provider)
        if not token:
            return {}
        if not official_url(provider, url):
            raise AuthError("認証情報は対応する公式HTTPS配布元にのみ送信できます")
        return {"Authorization": "Bearer " + token}


def read_json(provider, url, credentials, opener=None):
    if not official_url(provider, url):
        raise AuthError("公式API以外からモデル情報を取得しません")
    request = Request(url, headers={"Accept": "application/json","User-Agent":"mosaic-refiner/0.1",
                                   **credentials.headers(provider, url)})
    try:
        with (opener or secure_open)(request, timeout=30) as response:
            effective_url = response.geturl() if callable(getattr(response, 'geturl', None)) else url
            if not official_url(provider, effective_url):
                raise AuthError('配布元APIの応答が公式originと一致しません')
            data = response.read(2 * 1024 * 1024 + 1)
            if len(data) > 2 * 1024 * 1024:
                raise AuthError("配布元のモデル情報が大きすぎます")
            return json.loads(data)
    except HTTPError as exc:
        if exc.code in (401, 403):
            raise AuthError(f"配布元がAPI接続を拒否しました（HTTP {exc.code}）。公式ページでキー・同意・承認・接続制限を確認してください") from None
        raise AuthError(f"配布元APIを取得できません（HTTP {exc.code}）") from None
    except (OSError, URLError, ValueError) as exc:
        if isinstance(exc, AuthError):
            raise
        raise AuthError("配布元APIとの通信または応答の確認に失敗しました") from None


def check(provider, credentials, opener=None):
    if provider not in PROVIDERS or provider not in credentials.tokens:
        raise AuthError("この配布元の認証情報を入力してください")
    result = read_json(provider, PROVIDERS[provider][1], credentials, opener)
    if not isinstance(result, dict) or not (result.get("id") or result.get("name") or result.get("username")):
        raise AuthError("配布元の認証応答を確認できません")
    # Never return personal profile fields or credentials.
    return {"provider": provider, "authenticated": True, "model_access_verified": False}
