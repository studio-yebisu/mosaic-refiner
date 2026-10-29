"""Independent API adapter for the installed canonical H3 nodes.

No model download, server restart, global interrupt, or environment mutation.
"""
from pathlib import Path
import json
import shutil
import subprocess
import time
import uuid
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from .checkpoint import atomic_json, digest_file
from .h3_adapter import CANONICAL_ENDPOINT, MODEL_CANDIDATE, MODEL_EXPECTED_SHA256
from .media import MediaError, command, probe
from .runtime_observation import Observation, identity
from .processing_options import resolve_options, sampling_sigmas
from .restoration_models import COMFY_PYTHON, BASIC_PROFILE, identity as restoration_identity
from .h3_profile import PROFILE, assets as public_assets, loader_names, require_public_options, require_public_request
from .h3_lora_profiles import selected_lora

from .runtime_settings import RUNTIME
COMFY_ROOT = RUNTIME.root


def api(path, data=None):
    request = Request(CANONICAL_ENDPOINT + path,
                      data=json.dumps(data).encode() if data is not None else None,
                      headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=30) as response:
            return json.load(response)
    except HTTPError as exc:
        try:
            rejected = json.load(exc)
            failures = [f"{key}/{error.get('extra_info',{}).get('input_name',error.get('type','input'))}: {error.get('message','invalid')}"
                        for key, node in rejected.get('node_errors',{}).items() for error in node.get('errors',[])]
        except (ValueError, AttributeError):
            failures = []
        raise MediaError(f"選択したComfyUI APIが入力を拒否しました ({exc.code}): " + '; '.join(failures)[:1500]) from exc


def build_graph(bundle: Path, staged: str, prefix: str, prompt: str) -> dict:
    """Prepare the required-LoRA graph; execution still requires license clearance."""
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if manifest["state"] != "complete":
        raise ValueError("未完成のH3入力bundleです")
    request = manifest["request"]
    options = resolve_options(request.get("processing_options"))
    require_public_options(options)
    require_public_request(request)
    lora = selected_lora(options.lora_profile)
    low_sigmas, high_sigmas = sampling_sigmas(options)
    names = loader_names()
    canvas, window = request["canvas"], request["window"]
    width, height = canvas["width"], canvas["height"]
    frames, seed = window["padded_model_frames"], request["window_seed"]
    restored = (bundle / "restoration.json").is_file()
    if restored:
        preparation = json.loads((bundle / "restoration.json").read_text(encoding="utf-8"))
        if preparation.get("state") != "complete":
            raise MediaError("モザイク補正参照が未完成です")
        for name, artifact in preparation["artifacts"].items():
            if digest_file(bundle / name) != artifact["sha256"]:
                raise MediaError("モザイク補正参照のhashが一致しません")
    graph = {}

    def node(key, kind, **inputs):
        graph[key] = {"class_type": kind, "inputs": inputs}
        return [key, 0]

    for stage in ("low", "high"):
        mask_key = "highmaskfile" if stage == "high" else "lowmask"
        reference = f"restored-{stage}-reference.nut" if restored else f"{stage}-reference.nut"
        node(stage, "VHS_LoadVideoPath", video=str(bundle / reference),
             force_rate=0, custom_width=0, custom_height=0, frame_load_cap=frames,
             skip_first_frames=0, select_every_nth=1, format="None")
        if request.get("frame_regions") is not None:
            node(stage + "maskvideo", "VHS_LoadVideoPath", video=str(bundle / f"{stage}-mask.nut"),
                 force_rate=0, custom_width=0, custom_height=0, frame_load_cap=frames,
                 skip_first_frames=0, select_every_nth=1, format="None")
            node(mask_key, "ImageToMask", image=[stage + "maskvideo", 0], channel="red")
        else:
            node(mask_key, "LoadImageMask", image=f"{staged}/{stage}-mask.png", channel="red")
    node("silence", "LoadAudio", audio=f"{staged}/silence.wav")
    node("dit", "UNETLoader", unet_name=MODEL_CANDIDATE, weight_dtype="default")
    node("clip", "CLIPLoader", clip_name=names['clip'],
         type="minimax", device="default")
    node("vae", "VAELoader", vae_name=names['vae'])
    node("avae", "VAELoader", vae_name=names['audio_vae'])
    node('mainlora', 'LoraLoaderBypassModelOnly', model=['dit', 0],
         lora_name=str(Path(lora['lora'])), strength_model=options.lora_strength)
    node('mainshift', 'MiniMaxH3SigmaShift', model=['mainlora', 0], shift_video=6.0, shift_audio=3.0)
    node('main', 'H3TrialChunkedMLP', model=['mainshift', 0], chunk_size=4096, eager_mlp=True)
    for stage, w, h in (("l", width // 2, height // 2), ("h", width, height)):
        node(stage + "ref", "MiniMaxH3ReferenceToVideo", clip=["clip", 0], vae=["vae", 0],
             prompt=prompt, width=w, height=h, length=frames, ref_image_size="match",
             **{"ref_videos.ref_video_0": ["low" if stage == "l" else "high", 0]})
    node("spatial", "H3V2VGranularFractionalDenoise", latent=["lref", 1], model=["main", 0],
         vae=["vae", 0], audio_vae=["avae", 0], source_frames=["low", 0],
         source_audio=["silence", 0], source_fps=24.0, mode="spatial", global_strength=1.0,
         inside_strength=1.0, outside_strength=0.0, audio_strength=0.0, source_fit="start",
         crop="disabled", mask_temporal_reduce="max", invert_mask=False, source_mask=["lowmask", 0])
    node("lowoffload", "UnloadAllModels", value=["spatial", 0])
    node("lguider", "BasicGuider", model=["spatial", 1], conditioning=["lref", 0])
    node("sampler", "KSamplerSelect", sampler_name="euler")
    node("lsigmas", "ManualSigmas", sigmas=low_sigmas)
    node("lnoise", "RandomNoise", noise_seed=seed)
    node("low6", "SamplerCustomAdvanced", noise=["lnoise", 0], guider=["lguider", 0],
         sampler=["sampler", 0], sigmas=["lsigmas", 0], latent_image=["lowoffload", 0])
    node('upscale', 'MosaicH3LatentUpscale2x', latent=['low6', 1])
    # Refresh the mask on the high latent grid after deterministic spatial x2.
    node("highmask", "MiniMaxH3SetAVNoiseMask", latent=["upscale", 0], video_mask=["highmaskfile", 0])
    node("hoffload", "UnloadAllModels", value=["highmask", 0])
    node("hguider", "BasicGuider", model=["main", 0], conditioning=["href", 0])
    node("hsigmas", "ManualSigmas", sigmas=high_sigmas)
    node("hnoise", "RandomNoise", noise_seed=seed + 1)
    node("high3", "SamplerCustomAdvanced", noise=["hnoise", 0], guider=["hguider", 0],
         sampler=["sampler", 0], sigmas=["hsigmas", 0], latent_image=["hoffload", 0])
    node("decodeoffload", "UnloadAllModels", value=["high3", 0])
    node("decode", "VAEDecodeTiled", samples=["decodeoffload", 0], vae=["vae", 0],
         tile_size=512, overlap=64, temporal_size=64, temporal_overlap=8)
    node("video", "CreateVideo", images=["decode", 0], fps=24.0, bit_depth=8)
    node("save", "SaveVideo", video=["video", 0], filename_prefix=prefix, format="mp4",
         **{"format.codec": "h264", "format.codec.encoding": "re-encode",
            "format.codec.encoding.crf": 0.0}, codec="auto")
    return graph


class H3Runtime:
    ai_backend = True
    def __init__(self, processing_options=None):
        self.verified = False
        self.checked_nodes = set()
        self.processing_options = resolve_options(processing_options)

    def fingerprint(self):
        require_public_options(self.processing_options)
        selected_lora(self.processing_options.lora_profile, verify=True)
        RUNTIME.validate()
        # Verify weights once per invocation, not once per window.
        if not self.verified:
            for model in public_assets().values():
                path = model.destination(COMFY_ROOT / 'models')
                if not path.is_file() or path.stat().st_size != model.size_bytes or digest_file(path) != model.sha256:
                    raise MediaError('公開H3構成のモデルが未導入またはhash不一致です: '+model.id)
            self.verified = True
        code = [COMFY_ROOT / "comfy_extras/nodes_minimax_h3.py",
                COMFY_ROOT / "comfy_extras/nodes_lora_debug.py",
                COMFY_ROOT / "custom_nodes/mosaic_h3_public.py",
                COMFY_ROOT / "custom_nodes/ComfyUI-H3-Motion-Context-MultiRef/h3_v2v_fractional.py",
                COMFY_ROOT / 'custom_nodes/h3_trial_chunked_mlp.py']
        return {'profile': PROFILE, "connection": RUNTIME.to_dict(), "model": MODEL_CANDIDATE, "model_sha256": MODEL_EXPECTED_SHA256,
                "runtime_code": {str(p.relative_to(COMFY_ROOT)): digest_file(p) for p in code},
                'models': {role: {'id': model.id, 'sha256': model.sha256} for role, model in public_assets().items()},
                "restoration": restoration_identity(self.processing_options),
                "style_acting_lora": "none",
                "sampling": f"basicvsr-{self.processing_options.summary()['low_steps']}-spatial-x2-{self.processing_options.summary()['high_steps']}",
                "processing_options": self.processing_options.to_dict()}

    def prepare_restoration(self, bundle, cancelled, progress):
        """Run only our worker in the existing canonical Python, before H3."""
        receipt = bundle / "restoration.json"
        if receipt.is_file():
            saved = json.loads(receipt.read_text(encoding="utf-8"))
            if (saved.get("state") == "complete" and saved.get("profile") == BASIC_PROFILE and all(
                    digest_file(bundle / name) == data["sha256"] for name, data in saved["artifacts"].items())):
                return saved
        progress(phase="BasicVSR++でモザイク領域を先に補正")
        process = subprocess.Popen([str(COMFY_PYTHON), "-B", "-m", "mosaic_refiner.basicvsr_worker", str(bundle)],
            cwd=Path(__file__).resolve().parents[1], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        lines = []
        try:
            for line in process.stdout:
                lines.append(line.rstrip())
                lines = lines[-30:]
                if line.startswith("PREPASS "):
                    frame, total = map(int, line.split()[1].split("/"))
                    progress(phase="BasicVSR++でモザイク領域を先に補正", prepass_frame=frame, prepass_total=total)
                if cancelled():
                    raise KeyboardInterrupt
            if process.wait():
                raise MediaError("モザイク補正の前処理に失敗しました: " + "\n".join(lines)[-2500:])
            return json.loads(receipt.read_text(encoding="utf-8"))
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=15)
            process.stdout.close()

    def recover(self, bundle, directory, cancelled, progress):
        receipt = directory / "submission.json"
        if not receipt.is_file():
            return None
        prompt_id = json.loads(receipt.read_text(encoding="utf-8"))["prompt_id"]
        while True:
            history = api("/history/" + prompt_id).get(prompt_id)
            if history:
                atomic_json(directory / "history.json", history)
                if history.get("status", {}).get("status_str") in ("error", "interrupted"):
                    return None
                if history.get("status", {}).get("completed"):
                    try:
                        return self.collect(bundle, directory, history, prompt_id, None)
                    except FileNotFoundError:
                        # History survives external output cleanup. Preserve
                        # this attempt and let the processor make a new one.
                        return None
            queue = api("/queue")
            active = [item[1] for name in ("queue_running", "queue_pending") for item in queue.get(name, [])]
            if prompt_id not in active:
                # Process restart/history loss: this exact prompt is no longer
                # running or queued. Preserve old attempt; permit a new one.
                return None
            progress(phase="前回のH3窓を回収しています", prompt_id=prompt_id)
            time.sleep(1)

    def collect(self, bundle, directory, history, prompt_id, elapsed):
        entries = history.get("outputs", {}).get("save", {}).get("images", [])
        if not entries:
            raise MediaError("H3の保存動画をhistoryから確認できません")
        entry = entries[0]
        root = (COMFY_ROOT / "output").resolve()
        generated = (root / entry.get("subfolder", "") / entry["filename"]).resolve()
        if root not in generated.parents:
            raise MediaError("H3出力パスが正本outputの外です")
        result = directory / "generated.mp4"
        shutil.copyfile(generated, result)
        info = probe(result)
        request = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))["request"]
        canvas = request["canvas"]
        frames = request["window"]["padded_model_frames"]
        if (info.frame_count, info.width, info.height) != (frames, canvas["width"], canvas["height"]):
            raise MediaError("H3生成frame数・寸法が入力契約と一致しません")
        atomic_json(directory / "result.json", {"state": "complete", "prompt_id": prompt_id,
                    "elapsed_seconds": elapsed, "sha256": digest_file(result),
                    "info": info.to_dict(), "profile": self.fingerprint()})
        return result

    def generate(self, bundle: Path, directory: Path, cancelled, progress=lambda **kw: None) -> Path:
        request=json.loads((bundle / 'manifest.json').read_text(encoding='utf-8'))['request']
        require_public_request(request)
        request_options = resolve_options(request.get('processing_options'))
        if request_options != self.processing_options:
            raise MediaError("入力bundleとH3 runtimeの処理モードが異なります")
        self.fingerprint()
        process_identity = identity()
        # Check the live server, not just newly installed distribution metadata.
        # A process retaining the old DLLs must be restarted before inference.
        from .setup_packages import require_secure_torch
        require_secure_torch(api('/system_stats').get('system',{}).get('pytorch_version',''))
        while True:
            if cancelled():
                raise KeyboardInterrupt
            queue = api("/queue")
            if not queue.get("queue_running") and not queue.get("queue_pending"):
                break
            progress(phase="GPUの空きを待っています")
            time.sleep(1)
        self.prepare_restoration(bundle, cancelled, progress)
        if cancelled():
            raise KeyboardInterrupt
        # A different producer may have submitted while our prepass ran.
        while api("/queue").get("queue_running") or api("/queue").get("queue_pending"):
            if cancelled():
                raise KeyboardInterrupt
            progress(phase="GPUの空きを待っています")
            time.sleep(1)
        token = uuid.uuid4().hex
        staged = f"mosaic-refiner/{token}"
        staging = COMFY_ROOT / "input" / staged
        staging.mkdir(parents=True, exist_ok=False)
        directory.mkdir(parents=True, exist_ok=True)
        manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
        frames = manifest["request"]["window"]["padded_model_frames"]
        observation = None
        prompt_id = None
        terminal = False
        try:
            if manifest["request"].get("frame_regions") is None:
                for stage in ("low", "high"):
                    command(["ffmpeg", "-v", "error", "-nostdin", "-i", str(bundle / f"{stage}-write-mask.pgm"),
                             "-frames:v", "1", str(staging / f"{stage}-mask.png")])
            command(["ffmpeg", "-v", "error", "-nostdin", "-f", "lavfi", "-i", "anullsrc=r=32000:cl=stereo",
                     "-t", str(frames / 24 + 1), "-c:a", "pcm_f32le", str(staging / "silence.wav")])
            graph = build_graph(bundle, staged, f"mosaic-refiner/{token}",
                                manifest["request"]["prompt"])
            atomic_json(directory / "workflow.json", graph)
            for kind in {node["class_type"] for node in graph.values()}:
                if kind in self.checked_nodes:
                    continue
                if kind not in api("/object_info/" + kind):
                    raise MediaError(f"選択したComfyUIに必要nodeがありません: {kind}")
                self.checked_nodes.add(kind)
            if cancelled():
                raise KeyboardInterrupt
            started = time.monotonic()
            observation = Observation(token, process_identity)
            response = api("/prompt", {"prompt": graph, "client_id": token})
            prompt_id = response["prompt_id"]
            atomic_json(directory / "submission.json", {"prompt_id": prompt_id, "client_id": token})
            # Cancellation never calls global /interrupt. Wait for this window,
            # preserve its artifact, and stop before the next submission.
            missing = 0
            while True:
                history = api("/history/" + prompt_id).get(prompt_id)
                if history:
                    atomic_json(directory / "history.json", history)
                    if history.get("status", {}).get("status_str") in ("error", "interrupted"):
                        terminal = True
                        raise MediaError("H3生成に失敗しました。jobのhistory.jsonを確認してください")
                    if history.get("status", {}).get("completed"):
                        terminal = True
                        break
                else:
                    queue = api("/queue")
                    active = [item[1] for name in ("queue_running", "queue_pending") for item in queue.get(name, [])]
                    missing = 0 if prompt_id in active else missing + 1
                    if missing >= 3:
                        raise MediaError("選択したComfyUIのpromptがqueue/historyから消えました。jobを保存しました。再開で照合してください")
                events = [e for e in observation.events if e["type"] == "executing" and e["data"].get("prompt_id") == prompt_id]
                node = events[-1]["data"].get("node") if events else None
                settings = self.processing_options.summary()
                stages = {"low6":f"低段: {settings['low_steps']}step","upscale":"空間拡大: x2（LoRAなし）",
                          "high3":f"高段: {settings['high_steps']}step",
                          "decode":"映像を復号","save":"生成区間を保存"}
                progress(phase="取消し待ち（実行中の窓を保存）" if cancelled() else
                         f"H3 {settings['low_steps']} → 空間 x2 → {settings['high_steps']}",
                         node_stage=stages.get(node,"参照の符号化・モデル準備"),
                         prompt_id=prompt_id, elapsed_seconds=round(time.monotonic() - started, 1))
                time.sleep(1)
            return self.collect(bundle, directory, history, prompt_id, time.monotonic() - started)
        finally:
            if observation:
                atomic_json(directory / "observation.json", observation.finish(prompt_id))
            # Only our three staged files; no recursive shared-folder cleanup.
            # A transport failure may leave our prompt active: keep its inputs.
            if prompt_id is None or terminal:
                for name in ("low-mask.png", "high-mask.png", "silence.wav"):
                    (staging / name).unlink(missing_ok=True)
                staging.rmdir()
