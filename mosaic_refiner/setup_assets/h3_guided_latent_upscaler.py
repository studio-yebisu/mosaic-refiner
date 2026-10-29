"""Canonical x2 restoration of the clean low-pass H3 AV latent.

The pinned dntpi runner is unchanged. The graph supplies an independent
W4A8-safe Bypass model with the restoration LoRA. No project-specific inputs
or output/report paths live in this adapter.
"""
from pathlib import Path
import importlib.util
import json
import struct
import sys

import torch

PACKAGE = '_h3_guided_latent_upscaler_vendor'
UPSTREAM_COMMIT = '36cb612ec3df30094eeb4fec66f1528925ebba73'


def vendor():
    if PACKAGE not in sys.modules:
        base = Path(__file__).parent / 'ComfyUI-H3-Video-Upsampler'
        spec = importlib.util.spec_from_file_location(
            PACKAGE, base / '__init__.py', submodule_search_locations=[str(base)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[PACKAGE] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(PACKAGE, None)
            raise
    return (sys.modules[PACKAGE + '.tiled_upsampler'],
            sys.modules[PACKAGE + '.upscale_guide'])


class H3GuidedLatentUpscaler:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {'required': {
            'model': ('MODEL',), 'conditioning': ('CONDITIONING',),
            'latent': ('LATENT',), 'vae': ('VAE',), 'sampler': ('SAMPLER',),
            'upscale_model': (folder_paths.get_filename_list('latent_upscale_models'),),
            'guide_lora': (folder_paths.get_filename_list('loras'),),
            'width': ('INT', {'default': 768, 'min': 64, 'max': 4096, 'step': 64}),
            'height': ('INT', {'default': 1344, 'min': 64, 'max': 4096, 'step': 64}),
            'denoise': ('FLOAT', {'default': 0.1, 'min': 0.001, 'max': 1.0}),
            'tile_tokens': ('INT', {'default': 70000, 'min': 8192, 'max': 1048576}),
            'noise_seed': ('INT', {'default': 8250043, 'min': 0,
                                   'max': 18446744073709551615})}}

    RETURN_TYPES = ('LATENT',)
    FUNCTION = 'run'
    CATEGORY = 'H3/canonical'

    def run(self, model, conditioning, latent, vae, sampler, upscale_model,
            guide_lora, width, height, denoise, tile_tokens, noise_seed):
        import folder_paths
        import nodes

        tu, ug = vendor()
        lora_path = folder_paths.get_full_path_or_raise('loras', guide_lora)
        with open(lora_path, 'rb') as handle:
            header_size = struct.unpack('<Q', handle.read(8))[0]
            if header_size > 16 * 1024 * 1024:
                raise ValueError('Invalid guide LoRA header size')
            header = json.loads(handle.read(header_size))
        factor = ug.factor_from_meta(header.get('__metadata__'))
        if factor != 2:
            raise ValueError('Canonical restoration requires an x2 guide LoRA')
        samples = latent['samples']
        if not (getattr(samples, 'is_nested', False) and len(samples.tensors) == 2):
            raise ValueError('Expected the clean H3 video/audio latent')
        video, audio = samples.tensors
        if video.ndim != 5 or video.shape[0] != 1 or video.shape[1] != 24:
            raise ValueError('Expected one H3 video [1,24,T,H,W]')
        width, height = int(width), int(height)
        if (width, height) != (video.shape[4] * 16 * factor,
                               video.shape[3] * 16 * factor):
            raise ValueError('Target dimensions must be exactly twice the low pass')
        megapixels = width * height / 1048576.0
        if tu.target_size(video.shape[4] * 16, video.shape[3] * 16,
                          megapixels, align=32 * factor) != (width, height):
            raise ValueError('Target dimensions do not fit the x2 guide grid')
        ug.install()

        def decode(chunk):
            return nodes.VAEDecodeTiled().decode(
                vae, {'samples': chunk}, tile_size=512, overlap=64,
                temporal_size=64, temporal_overlap=8)[0]

        guide = tu.Guide(factor=factor, cond=conditioning, decode=decode,
                         encode=vae.encode, pixels=ug.guide_pixels)
        plan = tu.chunks(video.shape[2])
        tile_counts = [len(tu.tile_grid(height // 16, width // 16, k1 - k0,
                                        factor, int(tile_tokens))[0])
                       for k0, f0, k1, f1 in plan]
        details = {
            'source_latent': list(video.shape), 'output': [width, height],
            'time_chunks': len(plan), 'chunk_plan': [list(chunk) for chunk in plan],
            'spatial_tiles_per_chunk': tile_counts,
            'sampling_steps_per_tile': 1, 'redraw_forwards': sum(tile_counts),
            'tile_tokens': int(tile_tokens), 'guide_factor': factor,
            'guide_lora': guide_lora, 'denoise': float(denoise),
            'sigmas': tu.recipe_sigmas(model, denoise).tolist(),
            'seed': int(noise_seed), 'upstream_commit': UPSTREAM_COMMIT}
        previous_seed = tu.SEED
        tu.SEED = int(noise_seed)
        try:
            result = tu.upsample(model, conditioning, latent, sampler, upscale_model,
                                 megapixels, float(denoise), guide=guide,
                                 budget=int(tile_tokens))
        finally:
            tu.SEED = previous_seed
        output_video, output_audio = result['samples'].tensors
        if not torch.equal(output_audio, audio):
            raise RuntimeError('Guided upscaler changed the audio latent')
        if tuple(output_video.shape) != (1, 24, video.shape[2], height // 16, width // 16):
            raise RuntimeError('Guided upscaler changed the target dimensions or duration')
        details.update(audio_latent_equal=True, output_latent=list(output_video.shape))
        return {'ui': {'guided_upscale': [details]}, 'result': (result,)}


NODE_CLASS_MAPPINGS = {'H3GuidedLatentUpscaler': H3GuidedLatentUpscaler}
NODE_DISPLAY_NAME_MAPPINGS = {'H3GuidedLatentUpscaler': 'H3 Guided x2 Latent Upscaler'}
