import logging
import math

import torch

import comfy.model_management as mm
from comfy.ldm.modules.diffusionmodules.model import AttnBlock


# Attention implementations

def _pytorch_attention(q, k, v, **kwargs):
    import comfy.ldm.modules.diffusionmodules.model as vae_model
    return vae_model.pytorch_attention(q, k, v)


def _xformers_attention(q, k, v, **kwargs):
    import comfy.ldm.modules.diffusionmodules.model as vae_model
    if not mm.xformers_enabled_vae():
        logging.warning("PatchVAEAttentionDN: xformers not available for VAE, falling back to pytorch")
        return vae_model.pytorch_attention(q, k, v)
    return vae_model.xformers_attention(q, k, v)


def _slice_attention(q, k, v, **kwargs):
    import comfy.ldm.modules.diffusionmodules.model as vae_model
    return vae_model.normal_attention(q, k, v)


def _fallback_attention(q, k, v, reason):
    """ComfyUI's normal_attention (self-tiling slice attention, O(N^2) memory).

    q/k/v keep the original (B, C, N) or (B, C, H, W) layout - normal_attention
    does its own reshape/permute internally.
    """
    logging.warning("PatchVAEAttentionDN: %s; falling back to slice attention", reason)
    return _slice_attention(q, k, v)


def _split_head_dims(C, allowed):
    """Largest allowed head_dim that divides C; returns (head_dim, nheads)."""
    if C in allowed:
        return C, 1
    for hd in allowed:
        if C % hd == 0:
            return hd, C // hd
    return None, None


def flash_attention(q, k, v, **kwargs):
    """Flash attention for VAE AttnBlocks.

    - Runs in fp16/bf16 (flash-attn does not support fp32); the result is cast
      back to the input dtype.
    - The VAE attention is single-head with channel count C. flash-attn requires
      head_dim to be a multiple of 8 (and typically <= 256, hard shared-memory
      limit on many GPUs - e.g. 512 channels on a Radeon 780M fails with
      OutOfResources). When C does not fit, the single head is split into
      multiple heads of the largest supported head_dim; softmax_scale keeps the
      same temperature as the original single-head attention (1/sqrt(C)).
    - Any failure (OOM, OutOfResources, dtype, missing package, ...) falls back
      to ComfyUI's slice attention instead of crashing the process.
    """
    try:
        from flash_attn import flash_attn_func
    except ImportError:
        return _fallback_attention(q, k, v, "flash-attn is not installed")

    orig_shape = q.shape
    B = orig_shape[0]
    C = orig_shape[1]
    N = orig_shape[2] if len(orig_shape) == 3 else orig_shape[2] * orig_shape[3]

    orig_dtype = q.dtype
    if orig_dtype in (torch.float16, torch.bfloat16):
        work_dtype = orig_dtype
    else:
        work_dtype = torch.float16  # fp32 VAE: convert locally, convert back

    # head_dim must be a multiple of 8 and fit the flash-attn kernels (<= 256).
    head_dim, nheads = _split_head_dims(C, tuple(range(256, 7, -8)))
    if head_dim is None:
        return _fallback_attention(
            q, k, v, "channel count {} is unsupported by flash-attn".format(C))

    softmax_scale = 1.0 / math.sqrt(C)

    try:
        def _to_flash(t):
            # (B, C, N) or (B, C, H, W) -> (B, N, nheads, head_dim), contiguous
            t = t.reshape(B, C, N).transpose(1, 2)  # (B, N, C) view
            if t.dtype != work_dtype:
                t = t.to(work_dtype)
            t = t.contiguous()
            if nheads > 1:
                t = t.view(B, N, nheads, head_dim)
            else:
                t = t.unsqueeze(2)  # (B, N, 1, C)
            return t

        out = flash_attn_func(
            _to_flash(q), _to_flash(k), _to_flash(v),
            dropout_p=0.0,
            causal=False,
            softmax_scale=softmax_scale,
        )
        out = out.view(B, N, C).transpose(1, 2).reshape(orig_shape)
        if out.dtype != orig_dtype:
            out = out.to(orig_dtype)
        return out
    except Exception as e:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        return _fallback_attention(
            q, k, v, "flash attention failed ({}: {})".format(type(e).__name__, e))


def sage_attention(q, k, v, **kwargs):
    """SageAttention (thu-ml/SageAttention) for VAE AttnBlocks.

    - Runs in fp16/bf16 (fp32 is converted locally and converted back).
    - VAE attention is single-head with channel count C; SageAttention kernels
      accept head_dim in [64, 96, 128] (1.x) or up to 256 (2.x). The single
      head is split into multiple heads of the largest supported head_dim and
      sm_scale/scale keeps the same softmax temperature as single-head
      attention (1/sqrt(C)). API differences between 1.x and 2.x are detected
      via the function signature.
    - Any failure (missing package, unsupported shape, OOM, ...) falls back to
      ComfyUI's slice attention instead of crashing the process.
    """
    try:
        import inspect
        from sageattention import sageattn
    except ImportError:
        return _fallback_attention(q, k, v, "sageattention is not installed")

    orig_shape = q.shape
    B = orig_shape[0]
    C = orig_shape[1]
    N = orig_shape[2] if len(orig_shape) == 3 else orig_shape[2] * orig_shape[3]

    orig_dtype = q.dtype
    if orig_dtype in (torch.float16, torch.bfloat16):
        work_dtype = orig_dtype
    else:
        work_dtype = torch.float16  # fp32 VAE: convert locally, convert back

    try:
        # SageAttention 2.x uses `scale`, 1.x uses `sm_scale`; 2.x supports
        # head_dim up to 256, 1.x only [64, 96, 128]. Pick head splits that
        # work for the installed version.
        params = inspect.signature(sageattn).parameters
        if "scale" in params:
            allowed_head_dims = (256, 128, 96, 64)
            scale_kwarg = "scale"
        else:
            allowed_head_dims = (128, 96, 64)
            scale_kwarg = "sm_scale"

        head_dim, nheads = _split_head_dims(C, allowed_head_dims)
        if head_dim is None:
            return _fallback_attention(
                q, k, v, "channel count {} is unsupported by sageattention".format(C))

        def _to_sage(t):
            # (B, C, N) or (B, C, H, W) -> (B, N, nheads, head_dim), contiguous
            t = t.reshape(B, C, N).transpose(1, 2)  # (B, N, C) view
            if t.dtype != work_dtype:
                t = t.to(work_dtype)
            t = t.contiguous()
            if nheads > 1:
                t = t.view(B, N, nheads, head_dim)
            else:
                t = t.unsqueeze(2)  # (B, N, 1, C)
            return t

        call_kwargs = {
            "tensor_layout": "NHD",
            "is_causal": False,
            "smooth_k": True,
            scale_kwarg: 1.0 / math.sqrt(C),
        }
        out = sageattn(_to_sage(q), _to_sage(k), _to_sage(v), **call_kwargs)
        out = out.view(B, N, C).transpose(1, 2).reshape(orig_shape)
        if out.dtype != orig_dtype:
            out = out.to(orig_dtype)
        return out
    except Exception as e:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        return _fallback_attention(
            q, k, v, "sage attention failed ({}: {})".format(type(e).__name__, e))


_BACKENDS = {
    "pytorch":      _pytorch_attention,
    "xformers":     _xformers_attention,
    "split":        _slice_attention,
    "flash":        flash_attention,
    "sageattention": sage_attention,
}


# Patch helpers

def _patch_vae(vae, attn_fn):
    model = vae.first_stage_model
    patched = 0
    for module in model.modules():
        if isinstance(module, AttnBlock):
            module.optimized_attention = attn_fn
            patched += 1
    return vae


# ComfyUI node

class PatchVAEAttentionDN:
    """
    Patches the attention implementation used inside a VAE's encoder/decoder.
    ComfyUI automatically selects a VAE attention backend at startup,
    use this node to override that selection per-workflow.

    Set enabled to False to bypass the patch and pass the VAE through unchanged.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "attention": (list(_BACKENDS.keys()),),
                "enabled": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "patch"
    DESCRIPTION = (
        "Patches the attention implementation used inside a VAE's encoder and decoder. "
        "Overrides ComfyUI's auto-selected backend per-workflow without restarting. "
        "Wire the output VAE into your decoder."
    )
    EXPERIMENTAL = True
    CATEGORY = "DN/VAEAttention"

    def patch(self, vae, attention: str, enabled: bool):
        if not enabled:
            return (vae,)

        attn_fn = _BACKENDS[attention]
        _patch_vae(vae, attn_fn)
        logging.info(f"PatchVAEAttentionDN: VAE attention successfully patched to {attention}")
        return (vae,)
