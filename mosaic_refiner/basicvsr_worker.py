"""One owned restoration prepass using the canonical Python's existing Torch.

No new environment, package installation, content rendering or server mutation.
This entry point is isolated from the CPU GUI process so Torch is unloaded before H3.
"""
import argparse
from contextlib import ExitStack
from pathlib import Path
import time

from .checkpoint import atomic_json, digest_file, load_json
from .media import ChunkEncoder, MediaInfo, read_exact, rgb_reader, probe
from .restoration_models import BASIC, basic_path, BASIC_PROFILE, BASIC_CHUNK_FRAMES, BASIC_HALO_FRAMES
from .temporal_blend import TemporalBlend, temporal_tiles


def read_static_mask(path):
    import cv2
    import numpy as np
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)


def restore_bundle(bundle: Path, chunk_frames=BASIC_CHUNK_FRAMES, halo_frames=BASIC_HALO_FRAMES):
    import cv2
    import numpy as np
    import torch
    from .vendor.lada_basicvsr.network import BasicVSRPlusPlusNet

    started = time.monotonic()
    manifest = load_json(bundle / 'manifest.json')
    request = manifest['request']
    canvas = request['canvas']
    width, height = canvas['width'], canvas['height']
    count = request['window']['padded_model_frames']
    if min(width, height) < 256 or width % 4 or height % 4:
        raise ValueError('BasicVSR++には両辺256px以上・4の倍数のcanvasが必要です')
    if digest_file(basic_path()) != BASIC.sha256:
        raise ValueError('BasicVSR++の固定重みhashが一致しません')
    targets = [bundle / 'restored-high-reference.nut', bundle / 'restored-low-reference.nut']
    if any(path.exists() for path in targets):
        raise ValueError('既存の補正参照を上書きしません')
    state = torch.load(basic_path(), map_location='cpu', weights_only=True)
    # The official inference configuration uses the EMA generator at evaluation.
    weights = {name[len('generator_ema.'):]: value for name, value in state.items()
               if name.startswith('generator_ema.')}
    if not weights:
        raise ValueError('固定checkpointのEMA生成器がありません')
    model = BasicVSRPlusPlusNet(mid_channels=64, num_blocks=15)
    model.load_state_dict(weights, strict=True)
    del state, weights
    if not torch.cuda.is_available():
        raise ValueError('この推論prepassには正本環境のCUDAが必要です')
    device = torch.device('cuda')
    model = model.eval().to(device=device, dtype=torch.float16)
    torch.cuda.reset_peak_memory_stats()
    source = []
    masks = []
    size = width * height * 3
    with ExitStack() as stack:
        reader = stack.enter_context(rgb_reader(bundle / 'high-reference.nut'))
        dynamic = (bundle / 'high-mask.nut').is_file()
        mask_reader = stack.enter_context(rgb_reader(bundle / 'high-mask.nut')) if dynamic else None
        static_mask = read_static_mask(bundle / 'high-write-mask.pgm') if not dynamic else None
        if not dynamic and (static_mask is None or static_mask.shape != (height, width)):
            raise ValueError('固定maskの寸法が入力契約と一致しません')
        for _ in range(count):
            source.append(np.frombuffer(read_exact(reader, size), np.uint8).reshape(height, width, 3).copy())
            masks.append(np.frombuffer(read_exact(mask_reader, size), np.uint8).reshape(height, width, 3)[:, :, 0].copy()
                         if dynamic else static_mask)
    encoders = []
    changed_sum = channel_count = 0
    model_seconds = 0.0
    blender = TemporalBlend()
    try:
        for path, w, h in ((targets[0], width, height), (targets[1], width // 2, height // 2)):
            encoders.append(ChunkEncoder(path.with_suffix('.partial.nut'),
                MediaInfo(w, h, count, '24', '1/24', 0, 'rgb24', 'unknown', 'unknown')))
        with torch.inference_mode():
            for first, end, context_first, context_end, weights, flush_end in temporal_tiles(count, chunk_frames, halo_frames):
                # Model was trained in BGR. FFmpeg adapters and H3 use RGB.
                batch = np.stack(source[context_first:context_end])[..., ::-1].copy()
                tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).unsqueeze(0).to(device=device, dtype=torch.float16) / 255
                infer_start = time.monotonic()
                result = model(tensor)[0]
                torch.cuda.synchronize()
                model_seconds += time.monotonic() - infer_start
                if not torch.isfinite(result).all():
                    raise ValueError('BasicVSR++の出力に非有限値があります')
                restored = result.clamp(0, 1).mul(255).permute(0, 2, 3, 1).cpu().numpy()[..., ::-1]
                blender.add(context_first, restored, weights)
                for frame_id, pixels in blender.flush(flush_end):
                    original = source[frame_id]
                    # Full-strength throughout each detected shape. Feathering is
                    # done only once, by the authoritative final source compositor.
                    active = masks[frame_id] > 0
                    high = original.copy()
                    high[active] = pixels[active]
                    difference = np.abs(high[active].astype(np.int16) - original[active].astype(np.int16))
                    changed_sum += int(difference.sum())
                    channel_count += int(difference.size)
                    encoders[0].write(high.tobytes())
                    encoders[1].write(high[1::2, 1::2].tobytes())
                del tensor, result, restored
                print(f'PREPASS {end}/{count}', flush=True)
        for encoder in encoders:
            encoder.close()
        encoders.clear()
        for path in targets:
            partial = path.with_suffix('.partial.nut')
            info = probe(partial)
            if info.frame_count != count:
                raise ValueError('補正参照のframe数が入力契約と一致しません')
            partial.rename(path)
        report = {'state': 'complete', 'profile': BASIC_PROFILE,
                  'weights_sha256': BASIC.sha256, 'generator': 'generator_ema',
                  'chunk_frames': chunk_frames, 'halo_frames': halo_frames,
                  'temporal_blend': 'linear-overlap', 'max_blend_buffer_frames': blender.max_buffered_frames,
                  'frame_count': count, 'canvas': [width, height],
                  'seconds': time.monotonic() - started, 'model_seconds': model_seconds,
                  'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                  'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                  'torch_version': torch.__version__, 'gpu': torch.cuda.get_device_name(),
                  'masked_mae': changed_sum / max(1, channel_count),
                  'media_rendered_or_played': False, 'quality_adopted': False,
                  'artifacts': {p.name: {'sha256': digest_file(p), 'bytes': p.stat().st_size} for p in targets}}
        atomic_json(bundle / 'restoration.json', report)
        return report
    finally:
        for encoder in encoders:
            encoder.close(abort=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('bundle', type=Path)
    args = parser.parse_args()
    restore_bundle(args.bundle)


if __name__ == '__main__':
    main()
