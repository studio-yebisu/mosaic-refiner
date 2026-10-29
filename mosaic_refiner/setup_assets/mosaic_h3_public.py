"""Independent deterministic spatial x2. No checkpoint or LoRA is used."""


class MosaicH3LatentUpscale2x:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'latent': ('LATENT',)}}

    RETURN_TYPES = ('LATENT',)
    FUNCTION = 'upscale'
    CATEGORY = 'MosaicRefiner'

    def upscale(self, latent):
        import torch.nn.functional as F
        samples = latent['samples']
        video, audio = samples.unbind()
        if video.ndim != 5 or video.shape[0] < 1 or video.shape[2] < 1:
            raise ValueError('H3のB,C,T,H,W映像latentが必要です')
        batch, channels, frames, height, width = video.shape
        flat = video.movedim(2, 1).reshape(batch * frames, channels, height, width)
        scaled = F.interpolate(flat.float(), scale_factor=2, mode='bicubic', align_corners=False).to(video.dtype)
        scaled = scaled.reshape(batch, frames, channels, height * 2, width * 2).movedim(1, 2)
        result = dict(latent)
        result['samples'] = type(samples)((scaled, audio))
        # The caller supplies the original-coordinate mask on the new grid.
        result.pop('noise_mask', None)
        return (result,)


NODE_CLASS_MAPPINGS = {'MosaicH3LatentUpscale2x': MosaicH3LatentUpscale2x}
NODE_DISPLAY_NAME_MAPPINGS = {'MosaicH3LatentUpscale2x': 'H3 spatial latent x2 (no model / LoRA)'}
