import logging
import math
import os

import torch

import comfy.model_management as mm


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
    """Largest allowed head_dim that divides C; returns (head_dim, nheads).

    Splitting the VAE's single C-wide head into C//head_dim heads is an
    approximation, not a relabelling: every fused kernel here caps head_dim, so
    no head sees all C channels in its dot product and the softmax weights
    differ from the true single-head result. `scale=1/sqrt(C)` is still the right
    temperature - that is what the caller passes - because the split does not
    change the value the original head would have used.
    """
    if C in allowed:
        return C, 1
    for hd in allowed:
        if C % hd == 0:
            return hd, C // hd
    return None, None


# comfy_kitchen's INT8 attention kernels are instantiated for head_dim 64 / 128 /
# 256 (see _SAGE_HEAD_DIMS in comfy_kitchen/backends/hip/__init__.py). int8_attention
# pads anything smaller up to the next tile itself and rejects head_dim > 256.
_CK_HEAD_DIMS = (256, 128, 64)


# 已打印过的 shape（去重，同一 shape 只打印一次）
_LOGGED_SHAPES = set()

_DTYPE_NAMES = {
    torch.float16: "torch.float16",
    torch.bfloat16: "torch.bfloat16",
    torch.float32: "torch.float32",
}


def _log_vae_shape(backend, B, nheads, N, head_dim, dtype):
    """PRINT_SHAPE=1 时，打印 VAE attention 实际使用的 shape。
      ("VAE_FLASH", 1, 2, 2, 16384, 16384, 256, torch.float16)
    """
    if os.environ.get("PRINT_SHAPE") != "1":
        return
    sig = (backend, B, nheads, N, head_dim, str(dtype))
    if sig in _LOGGED_SHAPES:
        return
    _LOGGED_SHAPES.add(sig)
    dtype_name = _DTYPE_NAMES.get(dtype, str(dtype))
    print(
        '[PatchVAEAttentionDN] VAE attention shape: '
        '("VAE_{}", {}, {}, {}, {}, {}, {}, {})'.format(
            backend, B, nheads, nheads, N, N, head_dim, dtype_name),
        flush=True,
    )


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

        _log_vae_shape("FLASH", B, nheads, N, head_dim, work_dtype)
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
        params = inspect.signature(sageattn).parameters
        if "scale" in params:
            allowed_head_dims = (256, 128, 64)
            scale_kwarg = "scale"
        else:
            allowed_head_dims = (128, 64)
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

        _log_vae_shape("SAGE", B, nheads, N, head_dim, work_dtype)

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


def ck_attention(q, k, v, **kwargs):
    """comfy-kitchen INT8 attention for VAE attention blocks.

    This is the same kernel ComfyUI uses when it is started with
    `--use-ck-attention` (`comfy.ldm.modules.attention.attention_comfy_kitchen_int8`
    -> `comfy_kitchen.int8_attention`). ComfyUI only wires it into the diffusion
    models; the VAE keeps using split/pytorch attention, so this backend has no
    equivalent in stock ComfyUI.

    - Quantizes Q/K/V to INT8 and runs the softmax in FP32. It is therefore
      **lossy**; the other backends here keep fp16/bf16 accumulation. Use it to
      measure the speed, and check the image before committing to it.
    - `int8_attention` takes [B, H, N, D] (head-major) and rejects head_dim > 256.
      A VAE attention block is single-head with channel count C (512 for the SDXL
      VAE, 384 for the Qwen-Image / Wan VAE), so the channel dim is split into
      C // head_dim heads of the largest supported head_dim and `scale=1/sqrt(C)`
      keeps the original single-head temperature. Note that the split is an
      **approximation**, not a relabelling: each head only sums its own channels,
      so the softmax weights differ from the true C-wide ones. flash/sage make the
      same trade here and for the same reason (no kernel here accepts head_dim
      512), so the three fused backends stay comparable with each other.
    - fp32 VAE activations (the default, since ComfyUI does not enable
      `--fp16-vae`) are converted to fp16 locally and converted back, matching
      what flash/sage do here. Keeping one working dtype across the three fused
      backends is what makes their timings comparable.
    - Any failure (missing comfy-kitchen, no matrix cores, RDNA2's head_dim <= 128
      limit, unsupported channel count, OOM, ...) falls back to ComfyUI's slice
      attention instead of crashing the process.
    """
    try:
        import comfy_kitchen
    except ImportError:
        return _fallback_attention(q, k, v, "comfy_kitchen is not installed")

    # getattr rather than a bare call: comfy-kitchen ships the INT8 attention op
    # and its availability probe together, but an older wheel has neither, and a
    # bare AttributeError would come out of the except below as "comfy-kitchen
    # attention failed (AttributeError: ...)" instead of saying what is wrong.
    int8_attention = getattr(comfy_kitchen, "int8_attention", None)
    is_available = getattr(comfy_kitchen, "int8_attention_is_available", None)
    if int8_attention is None or is_available is None:
        return _fallback_attention(
            q, k, v, "the installed comfy-kitchen has no INT8 attention (upgrade comfy-kitchen)")

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
        if not is_available(q.device):
            return _fallback_attention(
                q, k, v, "comfy-kitchen INT8 attention is not available on {}".format(q.device))

        head_dim, nheads = _split_head_dims(C, _CK_HEAD_DIMS)
        if head_dim is None:
            return _fallback_attention(
                q, k, v, "channel count {} is unsupported by comfy-kitchen".format(C))

        def _to_ck(t):
            # (B, C, N) or (B, C, H, W) -> (B, nheads, N, head_dim), contiguous in D.
            # int8_attention only requires stride(-1) == 1 (the HIP kernels take
            # the batch/head/seq strides as arguments), so the final transpose is a
            # free view rather than a second copy of q/k/v.
            t = t.reshape(B, C, N).transpose(1, 2)  # (B, N, C) view
            if t.dtype != work_dtype:
                t = t.to(work_dtype)
            t = t.contiguous()
            if nheads > 1:
                t = t.view(B, N, nheads, head_dim)
            else:
                t = t.unsqueeze(2)  # (B, N, 1, C)
            return t.transpose(1, 2)  # (B, nheads, N, head_dim)

        _log_vae_shape("CK", B, nheads, N, head_dim, work_dtype)

        out = int8_attention(
            _to_ck(q), _to_ck(k), _to_ck(v), scale=1.0 / math.sqrt(C))
        # On gfx1103 comfy-kitchen allocates the output NHD-packed, so this
        # transpose/reshape is a view there and a copy elsewhere; either way the
        # layout is restored to what the surrounding AttnBlock expects.
        out = out.transpose(1, 2).reshape(B, N, C)
        out = out.transpose(1, 2).reshape(orig_shape)
        if out.dtype != orig_dtype:
            out = out.to(orig_dtype)
        return out
    except Exception as e:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        return _fallback_attention(
            q, k, v, "comfy-kitchen attention failed ({}: {})".format(type(e).__name__, e))


_BACKENDS = {
    "pytorch":  _pytorch_attention,
    "xformers": _xformers_attention,
    "split":    _slice_attention,
    "flash":    flash_attention,
    "sage":     sage_attention,
    "ck":       ck_attention,
}


# Patch helpers

# ComfyUI does not have one VAE attention class; it has five, and they all take
# the same hook: an `optimized_attention` attribute invoked as
# `self.optimized_attention(q, k, v)` with q/k/v shaped (B, C, N) or (B, C, H, W).
#
#   * ldm.modules.diffusionmodules.model.AttnBlock   - SD / SDXL / Flux VAE
#   * ldm.wan.vae.AttentionBlock                       - Wan 2.1 / Qwen-Image VAE
#   * ldm.cosmos.cosmos_tokenizer.layers3d.CausalAttnBlock
#   * ldm.mmaudio.vae.vae_modules.AttnBlock1D          - (B, C, N)
#   * ldm.mage_flow.vae.AttnBlock
#
# The attribute is therefore the contract, and patching on isinstance against the
# first class silently patched **zero** modules on the Qwen-Image VAE (anima,
# krea2) - the second entry - while still logging "successfully patched". That is
# the whole reason this walks the model duck-typed instead of by class.
def _looks_like_vae_attn_hook(fn):
    """Callable that takes (q, k, v) and nothing else required.

    `optimized_attention` is also the name of a *function* ComfyUI passes to
    transformer blocks (`clip_model.py`, `ldm.modules.attention`), but never as a
    module attribute, so duck-typing alone is nearly enough. The signature check
    is here for third-party models: patching a hook that also expects `heads` or
    `mask` would silently feed it the wrong positional arguments.
    """
    import inspect
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return True  # builtins / C callables: trust the attribute name
    required = [p for p in params.values()
                if p.default is inspect.Parameter.empty
                and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    return len(required) <= 3


def _patch_vae(vae, attn_fn):
    """Install attn_fn on every VAE attention hook. Returns the number patched."""
    model = vae.first_stage_model
    patched = 0
    for module in model.modules():
        current = getattr(module, "optimized_attention", None)
        if callable(current) and _looks_like_vae_attn_hook(current):
            module.optimized_attention = attn_fn
            patched += 1
    return patched


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
        patched = _patch_vae(vae, attn_fn)
        if patched == 0:
            # Say so instead of logging success. A VAE with no VAE-attention hook
            # (TAESD, pixel-space VAE, ...) is legitimate; a VAE that *should*
            # have one and got 0 is the bug this message exists to surface.
            logging.warning(
                "PatchVAEAttentionDN: no VAE attention block found - nothing patched. "
                "This VAE likely has no attention (TAESD / pixel-space / tiny "
                "autoencoders); if it should have one, the model layout changed.")
        else:
            logging.info(
                f"PatchVAEAttentionDN: VAE attention successfully patched to "
                f"{attention} ({patched} block{'s' if patched != 1 else ''})")
        return (vae,)
