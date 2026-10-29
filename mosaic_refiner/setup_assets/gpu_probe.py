"""Small synthetic CUDA/Sol-Attn/Triton checks; no model or media access."""
import json
import os
import sys
from pathlib import Path

if len(sys.argv)>1:
    os.environ['TRITON_CACHE_DIR']=str(Path(sys.argv[1])/'triton')

import torch
import triton
import triton.language as tl
import comfy_kitchen as ck
import comfy_aimdo
import cv2
import av
import torchaudio
import torchvision


@triton.jit
def add_one(source, target, count:tl.constexpr, block:tl.constexpr):
    indices=tl.program_id(0)*block+tl.arange(0,block)
    values=tl.load(source+indices,indices<count,other=0)
    tl.store(target+indices,values+1,indices<count)


def main():
    if not torch.cuda.is_available():raise RuntimeError('CUDA版TorchまたはNVIDIAドライバーを確認してください')
    device=torch.device('cuda:0')
    props=torch.cuda.get_device_properties(device)
    if props.total_memory/1024**3<11.9 or torch.cuda.get_device_capability(device)<(8,0):
        raise RuntimeError('実際に選択されたCUDA GPUが12GB / Ampere以降の条件を満たしません')
    if not ck.sol_attn_is_available(device):raise RuntimeError('このGPU用のSol-Attnカーネルがありません')
    q=torch.zeros((1,64,1,128),device=device,dtype=torch.bfloat16)
    result=ck.sol_attn(q,q,q)
    x=torch.zeros(64,device=device,dtype=torch.float32);y=torch.empty_like(x)
    add_one[(1,)](x,y,x.numel(),64)
    torch.cuda.synchronize()
    if result.shape!=q.shape or not torch.isfinite(result).all().item() or not torch.equal(y,torch.ones_like(y)):
        raise RuntimeError('GPUカーネルの合成数値検査に失敗しました')
    print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),
                     'capability':torch.cuda.get_device_capability(0),'vram_gib':props.total_memory/1024**3,
                     'sol_attn':True,'triton':triton.__version__,
                     'triton_kernel':True,'synthetic_kernel_only':True,'model_inference_tested':False}))


if __name__=='__main__':main()
