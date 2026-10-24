"""F01 B01/B03: isolated, source-frozen three-path GPU workers.

Run each worker through scripts/gpu_budget.py. A passing full-quality report
is required. Profile instrumentation is confined to --mode profile.
"""
import argparse
import contextlib
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from validate_prefix_kv import load_data, native, exact, run, dump, SEED
from ncm_serving.engine import NCMLevelExecutor
from ncm_serving.runtime import load_wmt14_runtime
from ncm_serving.service import NCMServingSystem
from ncm_serving import prefix_kv as kv


class PhaseProfile:
    """CUDA event spans and CPU call wall time, without per-phase synchronizes.

    Event spans can contain host enqueue gaps. Nested spans are not additive;
    these are timeline measurements, not a claim of exclusive kernel time.
    """
    def __init__(self,rt,server):
        self.rt,self.server=rt,server
        self.level=0
        self.events=[]
        self.rows=[]
        self.patches=[]
        self.hooks=[]
        self.cache_bytes=0
        self.batch=0

    @contextlib.contextmanager
    def span(self,label):
        level=self.level
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record();start=time.perf_counter()
        try:yield
        finally:
            cpu=(time.perf_counter()-start)*1000
            b.record()
            self.events.append((label,level,a,b,cpu))

    def wrap(self,obj,name,label,after=None):
        original=getattr(obj,name)
        def wrapped(*args,**kwargs):
            with self.span(label):result=original(*args,**kwargs)
            if after:after(result)
            return result
        setattr(obj,name,wrapped)
        self.patches.append((obj,name,original))

    def install(self):
        def next_level(_):self.level+=1
        self.wrap(self.rt.net,'_sample_level_raw','sample',next_level)
        vq=self.rt.tokenizer.vqvae.vq
        self.wrap(vq,'get_next_autoregressive_input','latent_update')
        self.wrap(vq,'idx_to_zhats','latent_reconstruct')
        self.wrap(self.rt.tokenizer.vqvae.decoder,'forward','decoder')
        self.wrap(torch.Tensor,'argmax','token_selection')
        for i,block in enumerate(self.rt.net.transformer.blocks):
            self.wrap(block.mlp,'forward',f'mlp/{i}')
        if self.server and self.server.executor.prefix_cache:
            self.wrap(kv,'attention','attention')
            self.wrap(kv,'block_forward','block')
            self.wrap(kv,'prefill','prefix_prefill')
            def cache_size(state):
                self.cache_bytes=max(self.cache_bytes,sum(t.numel()*t.element_size() for pair in state.prefix_kv for t in pair))
            self.wrap(self.server.executor,'short_group','short_group',cache_size)
        else:
            for i,block in enumerate(self.rt.net.transformer.blocks):
                self.wrap(block.attn,'forward',f'attention/{i}')
                self.wrap(block,'forward',f'block/{i}')
        if self.server:
            self.wrap(self.server.executor,'handoff','handoff')
            self.wrap(self.server,'submit_prefixes','submit')
        return self

    def finish_batch(self):
        torch.cuda.synchronize()
        self.rows.extend({'batch_index':self.batch,'phase':label,'level':level,'cuda_span_ms':a.elapsed_time(b),'cpu_call_ms':cpu}
                         for label,level,a,b,cpu in self.events)
        self.events.clear()
        self.batch+=1

    def close(self):
        for obj,name,original in reversed(self.patches):setattr(obj,name,original)


@torch.no_grad()
def compute(rt,server,p,cfg):
    if server:
        result=server.generate_prefixes(p,seed=SEED,cfg=cfg,request_id='batch')
        ids=result['token_ids'].cpu()
        server.scheduler.requests.pop('batch')
        del result
    else:
        g=rt.cfg.generation
        # Match the request-owned RNG initialization used by both serving paths.
        with torch.random.fork_rng(devices=[0]):
            torch.random.default_generator.manual_seed(SEED)
            torch.cuda.default_generators[0].manual_seed(SEED)
            codes=rt.net.generate(None,p.cuda(),temperature=float(g.temperature),
                top_k=int(g.top_k),top_p=float(g.top_p),cfg_scale=cfg,cfg_null_eos=rt.eos)
            _,z=rt.tokenizer.vqvae.vq.idx_to_zhats(codes,return_zlevels=True)
            ids=rt.tokenizer.vqvae.decoder(z[-1]).argmax(-1).cpu()
    return ids


def tokens(ids,rt):
    total=0
    for row in ids.tolist():
        for t in row:
            if t==rt.eos:break
            if t!=rt.pad:total+=1
    return total


def quantile(values,q):
    ordered=sorted(values);at=(len(ordered)-1)*q
    lo=int(at);hi=min(lo+1,len(ordered)-1)
    return ordered[lo]+(ordered[hi]-ordered[lo])*(at-lo)


def worker(args,result):
    quality=json.loads(args.quality.read_text(encoding='utf8'))
    if quality['status']!='PASS' or len(quality['quality'])!=2:
        raise ValueError('full WMT14 quality gate has not passed')
    start=time.perf_counter()
    rt=load_wmt14_runtime(args.ncm_root,args.checkpoint_dir,args.tokenizer_dir)
    prefixes,_=load_data(rt,args.data_dir)
    server=None if args.variant=='U0' else NCMServingSystem(rt,split=5,prefix_cache=args.variant=='K1',record_trace=False)
    result['load_and_fingerprint_seconds']=time.perf_counter()-start
    result['environment']={'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
        'fp32':True,'tf32':False,'grad_enabled_in_compute':False,'threads':torch.get_num_threads(),
        'attention':'PyTorch SDPA default dispatch; MHA FP32; additive mask; dropout=0'}
    result['quality_report_sha256']=hashlib.sha256(args.quality.read_bytes()).hexdigest()
    if args.mode=='preflight':
        e=NCMLevelExecutor(rt,5,record_trace=False)
        c=NCMLevelExecutor(rt,5,prefix_cache=True,record_trace=False)
        result['cells']=[]
        for b in (1,4,8):
            for cfg in (0,1):
                try:
                    a=native(rt,prefixes[:b],cfg)
                    for ex,name in ((e,'E0'),(c,'K1')):
                        v,_=run(ex,prefixes[:b],cfg)
                        exact(v,a,f'preflight {name} B{b} CFG{cfg}')
                    for name in ('U0','E0','K1'):
                        s=None if name=='U0' else NCMServingSystem(rt,split=5,prefix_cache=name=='K1',record_trace=False)
                        instrument=PhaseProfile(rt,s).install()
                        try:
                            ids=compute(rt,s,prefixes[:b],cfg)
                            instrument.finish_batch()
                            assert torch.equal(ids,a['token_ids']),f'profile changed output {name}'
                            assert len([r for r in instrument.rows if r['phase']=='sample'])==11
                        finally:instrument.close()
                    result['cells'].append({'batch':b,'cfg':cfg,'status':'PASS','instrumentation_exact':True})
                except torch.cuda.OutOfMemoryError:
                    if b!=8:raise
                    result['cells'].append({'batch':b,'cfg':cfg,'status':'OOM'})
                    torch.cuda.empty_cache()
        result['status']='PASS'
        return
    warm=[]
    for i in range(50):
        torch.cuda.synchronize();started=time.perf_counter()
        compute(rt,server,prefixes[:args.batch],args.cfg)
        torch.cuda.synchronize();warm.append(time.perf_counter()-started)
        if len(warm)>=10:
            a,b=statistics.median(warm[-10:-5]),statistics.median(warm[-5:])
            if abs(a-b)/a<=.05:break
    else:raise RuntimeError('warmup did not stabilize in 50 batches')
    result['warmup_seconds']=warm
    torch.cuda.reset_peak_memory_stats()
    result['memory_start_allocated']=torch.cuda.memory_allocated()
    result['memory_start_reserved']=torch.cuda.memory_reserved()
    profile=PhaseProfile(rt,server).install() if args.mode=='profile' else None
    samples=16*args.batch if profile else 1024
    elapsed=[];actual_tokens=0;digest=hashlib.sha256()
    try:
        for begin in range(0,samples,args.batch):
            if profile:profile.level=0
            torch.cuda.synchronize()
            started=time.perf_counter()
            ids=compute(rt,server,prefixes[begin:begin+args.batch],args.cfg)
            torch.cuda.synchronize()
            elapsed.append(time.perf_counter()-started)
            # Text, output hashing, and actual-token counting are outside the
            # declared CPU-prefix -> CPU-final-IDs boundary.
            actual_tokens+=tokens(ids,rt)
            digest.update(ids.contiguous().numpy().tobytes())
            if profile:profile.finish_batch()
        if profile:
            result['phases']=profile.rows
            result['cache_bytes']=profile.cache_bytes
    finally:
        if profile:profile.close()
    result.update(status='PASS',completed_samples=samples,completed_batches=len(elapsed),
        seconds=sum(elapsed),batch_seconds=elapsed,requests_per_second=samples/sum(elapsed),
        actual_tokens=actual_tokens,tokens_per_second=actual_tokens/sum(elapsed),
        batch_p50_ms=quantile(elapsed,.5)*1000,batch_p95_ms=quantile(elapsed,.95)*1000,
        ids_sha256=digest.hexdigest(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),allocated_end_bytes=torch.cuda.memory_allocated())
    if server:
        assert not server.executor.states and not server.scheduler.requests and not server.executor.trace

    if args.mode=='e2e':
        # Separate instrumented pass AFTER the closed, hook-free E2E window.
        # Share only model loading/warmup, not measured iterations or timers.
        profile=PhaseProfile(rt,server).install()
        torch.cuda.reset_peak_memory_stats()
        try:
            for begin in range(0,16*args.batch,args.batch):
                profile.level=0
                compute(rt,server,prefixes[begin:begin+args.batch],args.cfg)
                profile.finish_batch()
            result['profile']={'batches':16,'phases':profile.rows,'cache_bytes':profile.cache_bytes,
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                'timing':'separate post-E2E pass; nested CUDA event spans and CPU call time; spans not additive'}
        finally:profile.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('ncm-root','checkpoint-dir','tokenizer-dir','data-dir','quality','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--mode',choices=['preflight','e2e','profile'],required=True)
    p.add_argument('--variant',choices=['U0','E0','K1'],required=True)
    p.add_argument('--batch',type=int,choices=[1,4,8],default=4)
    p.add_argument('--cfg',type=int,choices=[0,1],default=0)
    p.add_argument('--window',type=int,choices=range(5),default=0)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    result={'status':'RUNNING','mode':args.mode,'variant':args.variant,'batch':args.batch,'cfg':args.cfg,'window':args.window}
    try:worker(args,result)
    except torch.cuda.OutOfMemoryError as exc:
        result.update(status='OOM',error=str(exc))
        raise
    except BaseException as exc:
        result.update(status='FAILED',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:dump(args.output,result)


if __name__=='__main__':main()
