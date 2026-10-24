"""Request-owned, immutable prefix KV for the pinned NCM MHA block.

Attention/block operations adapted from HazyResearch/ncm models/common.py,
Copyright (c) 2025 hermann kumbong, MIT; see THIRD_PARTY_NOTICES.md.
"""
import torch
from torch.nn import functional as F


def validate_blocks(net):
    for block in net.transformer.blocks:
        if any(getattr(block, name, False) for name in (
            'parallel_block', 'normformer', 'use_layerscale', 'use_rezero'
        )):
            raise ValueError('prefix cache requires the ordinary residual block')
        attn = block.attn
        if attn.n_kv_head != attn.n_head:
            raise ValueError('prefix cache currently supports MHA only')
        if attn.caching or attn.cached_k is not None or attn.cached_v is not None:
            raise ValueError('module-global KV caching must be disabled')
    if next(net.parameters()).dtype != torch.float32:
        raise ValueError('prefix cache currently supports FP32 only')


def rotate(x, cos, sin):
    a, b = x.chunk(2, dim=-1)
    return x * cos.to(x.dtype) + torch.cat([-b, a], dim=-1) * sin.to(x.dtype)


def attention(attn, x, bias, rope, prefix_kv=None):
    batch, length, width = x.shape
    q, k, v = attn.c_attn(x).split([attn.n_embd, attn._kv_dim, attn._kv_dim], dim=2)
    head_dim = width // attn.n_head
    q = q.view(batch, length, attn.n_head, head_dim).transpose(1, 2)
    k = k.view(batch, length, attn.n_head, head_dim).transpose(1, 2)
    v = v.view(batch, length, attn.n_head, head_dim).transpose(1, 2)
    if attn.use_qk_norm:
        q, k = attn.q_norm(q), attn.k_norm(k)
    q, k = rotate(q, *rope), rotate(k, *rope)
    if prefix_kv is None:
        # Contiguous clones avoid retaining the combined QKV projection storage.
        cache = (k.detach().contiguous().clone(), v.detach().contiguous().clone())
    else:
        cache = prefix_kv
        k, v = torch.cat([cache[0], k], dim=2), torch.cat([cache[1], v], dim=2)
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias,
                                       dropout_p=0., is_causal=False)
    y = y.transpose(1, 2).contiguous().view(batch, length, width)
    return attn.resid_dropout(attn.c_proj(y)), cache


def block_forward(block, x, bias, rope, prefix_kv=None):
    a, cache = attention(block.attn, block.ln_1(x), bias, rope, prefix_kv)
    x = x + a
    return x + block.mlp(block.ln_2(x)), cache


def prefill(net, prefix, pad_bias):
    length = prefix.shape[1]
    rope = (net.rope_cos_gen_0[:length], net.rope_sin_gen_0[:length])
    hidden, caches = prefix, []
    for block in net.transformer.blocks:
        hidden, cache = block_forward(block, hidden, pad_bias, rope)
        caches.append(cache)
    return net.transformer.ln_f(hidden), tuple(caches)


def target_logits(net, x, caches, pad_bias, head_index):
    p, length = net.prefix_len, x.shape[1]
    if len(caches) != len(net.transformer.blocks):
        raise ValueError('incomplete prefix KV')
    rope = (getattr(net, f'rope_cos_gen_{head_index}')[p:p+length],
            getattr(net, f'rope_sin_gen_{head_index}')[p:p+length])
    bias = F.pad(pad_bias, (0, length))
    for block, cache in zip(net.transformer.blocks, caches):
        if cache[0].shape[2] != p or cache[1].shape[2] != p:
            raise ValueError('prefix cache length changed')
        x, _ = block_forward(block, x, bias, rope, cache)
    return net.lm_heads[head_index](net.transformer.ln_f(x))
