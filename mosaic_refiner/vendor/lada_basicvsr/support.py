"""Small inference-only Torch support; no MMEngine/MMCV dependency or shim packages."""
import torch.nn as nn

BaseModule = nn.Module


def constant_init(module, val, bias=0):
    nn.init.constant_(module.weight, val)
    if module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module, a=0, nonlinearity='relu'):
    nn.init.kaiming_normal_(module.weight, a=a, mode='fan_out', nonlinearity=nonlinearity)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


def default_init_weights(module, scale=1):
    # Strict state-dictionary loading replaces every initialized parameter.
    for child in module.modules():
        if isinstance(child, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(child.weight, a=0, mode='fan_in')
            child.weight.data.mul_(scale)
            if child.bias is not None:
                nn.init.zeros_(child.bias)


def make_layer(block, count, **kwargs):
    return nn.Sequential(*(block(**kwargs) for _ in range(count)))
