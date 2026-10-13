# ComfyUI-DN_PatchVAEAttention

A ComfyUI custom node that patches the attention mechanism inside a VAE's encoder and decoder to use a specific implementation, regardless of what ComfyUI auto-selected at startup.

## Requirements

No additional packages required beyond ComfyUI's own dependencies. xformers support requires xformers, and the `flash` / `sage` / `ck` backends require `flash-attn` / `sageattention` / `comfy-kitchen` respectively.

## Backends

| backend | notes |
| --- | --- |
| `pytorch` | ComfyUI's `pytorch_attention` for VAE |
| `xformers` | falls back to `pytorch` when xformers is unavailable for the VAE |
| `split` | ComfyUI's `normal_attention` (self-tiling, O(N^2) memory) |
| `flash` | `flash_attn.flash_attn_func`, fp16/bf16 |
| `sage` | `sageattention.sageattn`, fp16/bf16, int8 Q/K with smoothed K |
| `ck` | `comfy_kitchen.int8_attention` - the same INT8 kernel ComfyUI uses for `--use-ck-attention`, but wired to the VAE |

ComfyUI's `--use-ck-attention` flag only reaches the diffusion models; stock ComfyUI leaves VAE attention on `split`/`pytorch`, which is what the `ck` backend here fills in.

## Installation

Clone into your ComfyUI custom nodes folder:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/0xDELUXA/ComfyUI-DN_PatchVAEAttention
```

Then restart ComfyUI.

## Usage

Find the node under **DN > VAEAttention > Patch VAE Attention DN**.

Wire it between your VAE loader and decoder - the patched VAE output is what you connect downstream:

```
Load VAE --> Patch VAE Attention DN --> VAE Decode
```

Or from a checkpoint:

```
Load Checkpoint (VAE) --> Patch VAE Attention DN --> VAE Decode
```

Set `enabled` to `False` to bypass the patch and pass the VAE through unchanged.

## Notes

- ComfyUI automatically selects a VAE attention backend at startup based on your hardware and installed packages; this node lets you override that per-workflow without restarting
- ComfyUI will still print a message like `Using split attention in VAE` or `Using xformers attention in VAE` at startup - this is logged before any node runs and can be ignored; the node is working if `PatchVAEAttentionDN: VAE attention successfully patched to X` appears in the log shortly after
- If xformers is unavailable, the node automatically falls back to PyTorch attention and logs a warning
- Every backend falls back to slice attention (with a logged warning) rather than crashing if the kernel is missing or fails on the given shape
- **`flash`, `sage` and `ck` approximate the VAE attention.** A VAE attention block is a *single* head whose width equals its channel count (512 for the SDXL VAE, 384 for the Qwen-Image / Wan VAE). Those three kernels cap `head_dim` at 256, so the channel dimension is split into multiple heads, and each head only sums its own channels - so the softmax weights differ from the true single-head attention. `pytorch`, `xformers` and `split` are exact. Check the image before relying on a fused backend.
- `ck` is additionally **lossy by design**: it quantizes Q/K/V to INT8. It is the fastest option to measure, not the most faithful one.
- Set `PRINT_SHAPE=1` in the environment to print each distinct VAE attention shape per backend, e.g. `("VAE_CK", 1, 2, 16384, 16384, 256, 256, torch.bfloat16)`
- Supported VAE families: SD/SDXL/Flux (`ldm.modules.diffusionmodules.model.AttnBlock`), Wan 2.1 / Qwen-Image (`ldm.wan.vae.AttentionBlock`), Cosmos, MMAudio and MageFlow - all five use the same `optimized_attention(q, k, v)` hook, so the patch reaches every one of them. TAESD / pixel-space VAEs have no attention and log a warning instead
