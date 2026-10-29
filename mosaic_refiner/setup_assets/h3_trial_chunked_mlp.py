"""Opt-in H3 MLP token chunking for long local W4A8 trials.

No weights or attention are changed. ModelPatcher restores bound forwards when
this model clone is unpatched; unrelated workflows do not opt into this node.
"""
import torch
import logging


def eager_mlp_forward(mlp):
    def forward(x):
        gate, up = mlp.fc1(x).chunk(2, dim=-1)
        return mlp.fc2(torch.nn.functional.silu(gate) * up)
    return forward


def chunked_forward(original, chunk_size):
    def forward(x):
        if x.ndim != 2 or x.shape[0] <= chunk_size:
            return original(x)
        result = torch.empty_like(x)
        for start in range(0, x.shape[0], chunk_size):
            end = min(start + chunk_size, x.shape[0])
            result[start:end] = original(x[start:end])
        return result
    return forward


class H3TrialChunkedMLP:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'model': ('MODEL',), 'chunk_size': ('INT', {'default': 4096, 'min': 256, 'max': 16384, 'step': 256})},
                'optional': {'eager_mlp': ('BOOLEAN', {'default': False})}}

    RETURN_TYPES = ('MODEL',)
    FUNCTION = 'patch'
    CATEGORY = 'model/patch/local trial'

    def patch(self, model, chunk_size, eager_mlp=False):
        from comfy.ldm.minimax.model import MiniMaxH3Model, MLP
        core = model.get_model_object('diffusion_model')
        if not isinstance(core, MiniMaxH3Model):
            raise ValueError('H3TrialChunkedMLP requires MiniMax H3')
        patched = model.clone()
        if eager_mlp:
            count = 0
            for name, module in core.named_modules():
                if isinstance(module, MLP):
                    patched.add_object_patch(f'diffusion_model.{name}.forward', chunked_forward(eager_mlp_forward(module), chunk_size))
                    count += 1
            logging.info('H3 trial: %d MLPs use explicit fc1/fc2 forwards so bypass LoRA hooks are preserved', count)
        else:
            for index, block in enumerate(core.blocks):
                patched.add_object_patch(f'diffusion_model.blocks.{index}.mlp.forward', chunked_forward(block.mlp.forward, chunk_size))
        return (patched,)


NODE_CLASS_MAPPINGS = {'H3TrialChunkedMLP': H3TrialChunkedMLP}
NODE_DISPLAY_NAME_MAPPINGS = {'H3TrialChunkedMLP': 'H3 MLP Token Chunking (local trial)'}
