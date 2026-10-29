"""Explicit, resumable speed/detail choices. No quality ranking or GPU effects."""
from dataclasses import asdict, dataclass
import math


PRESETS = {
    "speed": {"render_scale_percent": 75, "max_pixels": 147456, "crop_margin": 16, "sampling": "light"},
    "balanced": {"render_scale_percent": 100, "max_pixels": 262144, "crop_margin": 32, "sampling": "standard"},
    "detail": {"render_scale_percent": 125, "max_pixels": 589824, "crop_margin": 64, "sampling": "standard"},
}
LABELS = {"speed": "速度優先", "balanced": "標準", "detail": "精細さ優先"}
SIGMAS = {
    "standard": ("1,0.9677419355,0.9230769231,0.8571428571,0.75,0.5454545455,0",
                 "0.8571428571,0.6666666667,0.4615384615,0"),
    "light": ("1,0.9473684211,0.8571428571,0.6666666667,0",
              "0.8571428571,0.4615384615,0"),
}


def _shifted_sigmas(steps, start=1.0):
    return ','.join(f'{6*t/(1+5*t):.10g}' for t in (start*(steps-i)/steps for i in range(steps+1)))


PUBLIC_SIGMAS = {'standard': (_shifted_sigmas(20), _shifted_sigmas(10, .5)),
                'light': (_shifted_sigmas(12), _shifted_sigmas(6, .5))}


def sampling_sigmas(options):
    # The selected base remains non-Turbo. A restoration LoRA is not a Turbo distillation.
    return (SIGMAS if options.version == 1 else PUBLIC_SIGMAS)[options.sampling]


@dataclass(frozen=True)
class ProcessingOptions:
    mode: str = "balanced"
    render_scale_percent: int = 100
    max_pixels: int = 262144
    crop_margin: int = 32
    sampling: str = "standard"
    version: int = 3
    lora_profile: str = "unlocked_v2"
    lora_strength: float = 1.0

    def to_dict(self):
        return asdict(self)

    def summary(self):
        low, high = sampling_sigmas(self)
        values = self.to_dict()
        customized = any(values[key] != value for key, value in PRESETS[self.mode].items())
        return {**values, "label": LABELS[self.mode] + ("（詳細変更）" if customized else ""),
                "low_steps": len(low.split(",")) - 1, "high_steps": len(high.split(",")) - 1,
                "guided_steps_per_tile": 1 if self.version == 1 else 0, "h3_regeneration_required": True,
                "quality_reviewed": False}


def resolve_options(raw=None):
    if isinstance(raw, ProcessingOptions):
        raw = raw.to_dict()
    if isinstance(raw, str):
        raw = {"mode": raw}
    if raw is None:
        raw = {}
    if not isinstance(raw, dict) or set(raw) - set(ProcessingOptions.__dataclass_fields__):
        raise ValueError("処理モードの設定が不正です")
    mode = raw.get("mode", "balanced")
    if not isinstance(mode, str) or mode not in PRESETS:
        raise ValueError("処理モードは速度優先・標準・精細さ優先から選択してください")
    values = {"mode": mode, **PRESETS[mode], "version": 3, **raw}
    for key, allowed in (("render_scale_percent", (50, 75, 100, 125)),
                         ("max_pixels", (65536, 147456, 262144, 589824)),
                         ("crop_margin", (0, 16, 32, 64)), ("version", (1,2,3))):
        if type(values[key]) is not int or values[key] not in allowed:
            raise ValueError(f"処理設定 {key} の選択値が不正です")
    if not isinstance(values["sampling"], str) or values["sampling"] not in SIGMAS:
        raise ValueError("H3の計算量は速度優先・標準から選択してください")
    from .h3_lora_profiles import PROFILES
    values.setdefault("lora_profile", {1: "restoration", 2: "none", 3: "unlocked_v2"}[values['version']])
    values.setdefault("lora_strength", 0.0 if values['lora_profile'] == 'none' else 1.0)
    if not isinstance(values["lora_profile"], str) or values["lora_profile"] not in PROFILES:
        raise ValueError("H3 LoRAは導入済みの明示profileから選択してください")
    strength = values["lora_strength"]
    if type(strength) not in (int, float) or not math.isfinite(strength) or not 0 <= strength <= 2:
        raise ValueError("LoRA強度は0〜2の有限の数値で指定してください（0で無効）")
    values["lora_strength"] = float(strength)
    if values['lora_profile'] == 'none' and values['lora_strength'] != 0:
        raise ValueError('追加LoRAなしの構成は強度0を指定してください')
    if values['version'] == 3 and (values['lora_profile'] == 'none' or values['lora_strength'] <= 0):
        raise ValueError('追加LoRAは必須です。Unlocked V2またはDecensoreH3と、0より大きい強度を指定してください')
    return ProcessingOptions(**values)


def catalog(lora_profile="unlocked_v2"):
    return {mode: resolve_options({"mode": mode, "lora_profile": lora_profile}).summary() for mode in PRESETS}
