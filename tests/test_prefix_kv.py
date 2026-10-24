"""CPU reference checks; NCM_ROOT points to the pinned external checkout."""
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from ncm_serving.engine import NCMLevelExecutor
from ncm_serving.scheduler import TwoGroupScheduler, RequestStatus
from ncm_serving import prefix_kv as kv

ROOT = os.environ.get('NCM_ROOT')
if ROOT:
    sys.path.insert(0, str(Path(ROOT).resolve()))
    from models.next_concept import NextConceptTransformer
    from models.vqvae import VQVAE


def toy():
    torch.manual_seed(31)
    vqcfg = OmegaConf.create({
        'clss': 'multiscale_residual_vector_quantizer', 'decay': .99,
        'epsilon': 1e-5, 'commitment_cost': .25, 'learned_l1_sampling': False,
        'learned_all_sampling': False, 'quant_resi': {'enabled': False},
        'levels': {'use_manual_levels': True, 'manual_levels': [1, 3, 7]},
    })
    vq = VQVAE(in_vocab_size=32, context_length=7, embed_dim=8,
               vqvae_vocab_size=[16]*3, n_layers=2, vector_quantizer_config=vqcfg,
               compression_factor=1, dropout=0., pre_quant_groupnorm=None,
               use_qk_norm=False).eval()
    cfg = OmegaConf.create({
        'n_layer': 2, 'n_head': 2, 'bias': True, 'dropout': 0., 'n_embd': 16,
        'context_length': 16, 'prefix_len': 5, 'raw_token_prefix': True,
        'prefix_pad_token_id': 0, 'vocab_size': [16,16], 'predict_level0': True,
        'query_level0': False, 'attn_pattern': 'block_diagonal',
        'use_level_encoding': True, 'use_positional_encoding': True,
        'use_rope': True, 'rope_base': 10000, 'use_qk_norm': True,
        'use_flex_attention': False, 'untied_head_bias': False,
    })
    return types.SimpleNamespace(net=NextConceptTransformer(vq, cfg).eval(),
        tokenizer=types.SimpleNamespace(vqvae=vq), pad=0, eos=1,
        cfg=types.SimpleNamespace(generation=types.SimpleNamespace(
            temperature=1., top_k=1, top_p=0.)))


@torch.no_grad()
def reference(rt, prefix, cfg, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        g = rt.cfg.generation
        codes = rt.net.generate(None, prefix, temperature=g.temperature,
                               top_k=g.top_k, top_p=g.top_p,
                               cfg_scale=cfg, cfg_null_eos=rt.eos)
        _, z = rt.tokenizer.vqvae.vq.idx_to_zhats(codes, return_zlevels=True)
        return {'codes': codes, 'latent': z[-1],
                'token_ids': rt.tokenizer.vqvae.decoder(z[-1]).argmax(-1)}


@unittest.skipUnless(ROOT, 'set NCM_ROOT to run actual NCM CPU reference checks')
class PrefixCacheTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_parity_positions_and_cache_immutability(self):
        for split in (1,2):
            for cfg in (0.,1.):
                for random in (False,True):
                    with self.subTest(split=split,cfg=cfg,random=random):
                        rt = toy()
                        if random:
                            rt.cfg.generation.top_k, rt.cfg.generation.top_p = 5,.8
                        p = torch.tensor([[2,0,0,0,0],[5,6,7,8,9],[4,3,2,0,0]])
                        expected = reference(rt,p,cfg,55)
                        executor = NCMLevelExecutor(rt,split,prefix_cache=True)
                        lengths = [[] for _ in rt.net.transformer.blocks]
                        hooks = [b.attn.c_attn.register_forward_pre_hook(
                            lambda m,a,i=i: lengths[i].append(a[0].shape[1]))
                            for i,b in enumerate(rt.net.transformer.blocks)]
                        before = torch.get_rng_state().clone()
                        state = executor.submit('r',p,seed=55,cfg=cfg)
                        executor.short_group(state)
                        cache = [(k.clone(),v.clone()) for k,v in state.prefix_kv]
                        original_cache = state.prefix_kv
                        moved = executor.handoff(state)
                        self.assertIsNone(state.prefix_kv)
                        for old,new in zip(original_cache,moved.prefix_kv):
                            for a,b in zip(old,new):
                                self.assertNotEqual(a.data_ptr(),b.data_ptr())
                                self.assertTrue(torch.equal(a,b))
                        self.assertEqual(moved.prefix_kv[0][0].shape[2],5)
                        retained = moved.prefix_kv
                        actual = executor.long_group(moved)
                        for old,new in zip(cache,retained):
                            for a,b in zip(old,new): self.assertTrue(torch.equal(a,b))
                        for hook in hooks: hook.remove()
                        for key in ('codes','token_ids'):
                            self.assertTrue(torch.equal(expected[key],actual[key]),key)
                        torch.testing.assert_close(expected['latent'],actual['latent'],atol=1e-4,rtol=1e-3)
                        self.assertEqual(lengths,[[5,3,7]]*2)
                        self.assertTrue(torch.equal(before,torch.get_rng_state()))
                        self.assertIsNone(moved.prefix_kv)
                        self.assertFalse(executor.states)
                        self.assertTrue(all(b.attn.cached_k is None for b in rt.net.transformer.blocks))

    def test_interleaving_cancel_and_exception_release(self):
        rt = toy()
        rt.cfg.generation.top_k,rt.cfg.generation.top_p = 5,.8
        p,q = torch.tensor([[2,3,0,0,0]]),torch.tensor([[2,3,4,5,0]])
        ea,eb = reference(rt,p,0.,1),reference(rt,q,1.,2)
        ex = NCMLevelExecutor(rt,1,prefix_cache=True)
        a,b = ex.submit('a',p,seed=1),ex.submit('b',q,seed=2,cfg=1.)
        ex.short_group(a)
        saved = [(k.clone(),v.clone()) for k,v in a.prefix_kv]
        ex.short_group(b)
        for (k,v),(sk,sv) in zip(a.prefix_kv,saved):
            self.assertTrue(torch.equal(k,sk) and torch.equal(v,sv))
        self.assertTrue(torch.equal(ex.long_group(ex.handoff(b))['codes'],eb['codes']))
        self.assertTrue(torch.equal(ex.long_group(ex.handoff(a))['codes'],ea['codes']))
        for i in range(100):
            s=ex.submit(str(i),p,seed=i)
            ex.short_group(s)
            ex.cancel(s)
            self.assertIsNone(s.prefix_kv)
        self.assertFalse(ex.states)
        for stage in ('short','handoff','long'):
            s=ex.submit(stage,p,seed=1)
            if stage != 'short': ex.short_group(s)
            if stage == 'long': s=ex.handoff(s)
            obj,method = (s,'snapshot') if stage=='handoff' else (rt.net,'_sample_level_raw')
            with patch.object(obj,method,side_effect=RuntimeError('injected')):
                with self.assertRaisesRegex(RuntimeError,'injected'):
                    getattr(ex,stage if stage=='handoff' else stage+'_group')(s)
            self.assertIsNone(s.prefix_kv)
            self.assertFalse(ex.states)
        scheduler=TwoGroupScheduler(ex)
        rec=scheduler.submit('fail',p,seed=1)
        scheduler.step()
        with patch.object(rt.net,'_sample_level_raw',side_effect=RuntimeError('injected')):
            with self.assertRaisesRegex(RuntimeError,'injected'): scheduler.step()
        self.assertIs(rec.status,RequestStatus.FAILED)
        self.assertIsNone(rec.state)
        self.assertFalse(ex.states)

    def test_unsupported_and_foreign_state(self):
        rt=toy()
        rt.net.transformer.blocks[0].parallel_block=True
        with self.assertRaisesRegex(ValueError,'block'):
            NCMLevelExecutor(rt,1,prefix_cache=True)
        rt.net.transformer.blocks[0].parallel_block=False
        ex=NCMLevelExecutor(rt,1,prefix_cache=True)
        with torch.autocast('cpu',dtype=torch.bfloat16):
            with self.assertRaisesRegex(ValueError,'autocast'):
                ex.submit('autocast',torch.tensor([[2,3,0,0,0]]),seed=1)
        for p in (torch.empty(0,5,dtype=torch.long),torch.zeros(1,5,dtype=torch.long)):
            with self.assertRaises(ValueError): ex.submit('bad',p,seed=1)
        state=ex.submit('a',torch.tensor([[2,3,0,0,0]]),seed=1)
        ex.short_group(state)
        old=state
        state=ex.handoff(state)
        with self.assertRaisesRegex(ValueError,'stale'): ex.long_group(old)
        state.model_id='wrong'
        with self.assertRaisesRegex(ValueError,'mismatch'): ex.long_group(state)

    @torch.no_grad()
    def test_dense_float64_mask_and_rope_oracle(self):
        rt=toy()
        attn=rt.net.transformer.blocks[0].attn
        # Oracle uses explicit dense softmax and complex-number rotation,
        # independently of both production SDPA and split-half rotation.
        def rotate64(x,positions):
            half=x.shape[-1]//2
            angle=positions.double()[:,None] / (10000. ** (torch.arange(half).double()/half))
            z=torch.complex(x[...,:half].double(),x[...,half:].double())
            z=z*torch.polar(torch.ones_like(angle),angle)
            return torch.cat((z.real,z.imag),-1)
        def project(x,positions):
            q,k,v=attn.c_attn(x).chunk(3,-1)
            q,k,v=[a.reshape(3,-1,2,8).transpose(1,2) for a in (q,k,v)]
            return rotate64(attn.q_norm(q),positions),rotate64(attn.k_norm(k),positions),v.double()
        p=torch.randn(3,5,16)
        x=torch.randn(3,7,16)
        bias=torch.zeros(3,1,1,5)
        bias[0,:,:,1:]=-torch.inf
        bias[2,:,:,3:]=-torch.inf
        pc,ps=rt.net.rope_cos_gen_0[:5],rt.net.rope_sin_gen_0[:5]
        _,cache=kv.attention(attn,p,bias,(pc,ps))
        _,pk,pv=project(p,torch.arange(5))
        torch.testing.assert_close(cache[0].double(),pk,atol=1e-5,rtol=1e-4)
        torch.testing.assert_close(cache[1].double(),pv,atol=1e-5,rtol=1e-4)
        rope=(rt.net.rope_cos_gen_1[5:12],rt.net.rope_sin_gen_1[5:12])
        full_bias=torch.nn.functional.pad(bias,(0,7))
        actual,_=kv.attention(attn,x,full_bias,rope,cache)
        q,k,v=project(x,torch.arange(7))
        probabilities=(q @ torch.cat((pk,k),2).transpose(-1,-2)/8**.5+full_bias.double()).softmax(-1)
        self.assertEqual(probabilities[0,:,:,:5][:,:,1:].count_nonzero(),0)
        self.assertTrue((probabilities[:,:,0,-1]>0).all()) # future target visible
        dense=(probabilities@torch.cat((pv,v),2)).transpose(1,2).reshape(3,7,16)
        expected=torch.nn.functional.linear(dense,attn.c_proj.weight.double(),attn.c_proj.bias.double())
        torch.testing.assert_close(actual.double(),expected,atol=1e-5,rtol=1e-4)
        # Demonstrate sensitivity to omitted padding, wrong target positions,
        # and repeated rotation of prefix K, the three tempting regressions.
        bad_mask,_=kv.attention(attn,x,torch.zeros_like(full_bias),rope,cache)
        shifted=(torch.roll(rope[0],1,0),torch.roll(rope[1],1,0))
        bad_pos,_=kv.attention(attn,x,full_bias,shifted,cache)
        bad_key,_=kv.attention(attn,x,full_bias,rope,(kv.rotate(cache[0],pc,ps),cache[1]))
        for bad in (bad_mask,bad_pos,bad_key):
            self.assertFalse(torch.allclose(bad.double(),expected,atol=1e-5,rtol=1e-4))


if __name__ == '__main__': unittest.main()
