"""F01 correctness only: full WMT14 parity and cache structure/lifetime.

Run through gpu_budget.py; no timings from this file are performance evidence.
"""
import argparse
import hashlib
import json
import pickle
import traceback
from pathlib import Path
from unittest.mock import patch

import torch

from ncm_serving.engine import NCMLevelExecutor
from ncm_serving.runtime import load_wmt14_runtime
from ncm_serving import prefix_kv as kv

SEED=20261023


def dump(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def output_hash(rows):
    return hashlib.sha256(json.dumps(rows,separators=(',',':')).encode()).hexdigest()


def load_data(rt, directory):
    with (directory/'sources.pickle').open('rb') as f: items=pickle.load(f)
    with (directory/'reference_translations.pickle').open('rb') as f: refs=pickle.load(f)
    encoded=[rt.tokenizer.pre_tokenizer.hf_tokenizer(x)['input_ids'] if isinstance(x,str) else list(x) for x in items]
    p=rt.net.prefix_len
    return torch.tensor([(x[:p]+[rt.pad]*p)[:p] for x in encoded],dtype=torch.long),refs


@torch.no_grad()
def run(ex,p,cfg,rng=None):
    s=ex.submit('validation',p,seed=SEED,cfg=cfg)
    if rng is not None: s.cpu_rng,s.device_rng=[x.clone() for x in rng]
    ex.short_group(s)
    s=ex.handoff(s)
    out=ex.long_group(s)
    return {k:v.cpu() for k,v in out.items()},(s.cpu_rng,s.device_rng)


@torch.no_grad()
def native(rt,p,cfg):
    g=rt.cfg.generation
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    c=rt.net.generate(None,p.cuda(),temperature=float(g.temperature),top_k=int(g.top_k),
                      top_p=float(g.top_p),cfg_scale=cfg,cfg_null_eos=rt.eos)
    _,z=rt.tokenizer.vqvae.vq.idx_to_zhats(c,return_zlevels=True)
    return {'codes':c.cpu(),'latent':z[-1].cpu(),
            'token_ids':rt.tokenizer.vqvae.decoder(z[-1]).argmax(-1).cpu()}


def exact(a,b,context):
    for key in ('codes','token_ids'):
        if not torch.equal(a[key],b[key]):
            loc=(a[key]!=b[key]).nonzero()[0].tolist()
            raise AssertionError(f'{context}: {key} differs at batch row/column {loc}')
    torch.testing.assert_close(a['latent'],b['latent'],atol=1e-4,rtol=1e-3)


@torch.no_grad()
def intermediates(eager,cached,p,cfg):
    """Capture actual full-block outputs and normalized/rotated prefix K/V."""
    from models.common import apply_rotary_emb
    ref,actual={},{}
    counters=[0]*len(eager.net.transformer.blocks)
    handles=[]
    def pre(i,module,args,kwargs):
        x=args[0]; b,t,c=x.shape; a=module; h=c//a.n_head
        _,k,v=a.c_attn(x).split([a.n_embd,a._kv_dim,a._kv_dim],dim=2)
        k=k.view(b,t,a.n_head,h).transpose(1,2)
        v=v.view(b,t,a.n_head,h).transpose(1,2)
        if a.use_qk_norm: k=a.k_norm(k)
        cos,sin=kwargs['rope_cos_sin']
        k=apply_rotary_emb(k,cos[:t],sin[:t])
        level=counters[i]
        ref[(level,i,'K')]=k[:,:,:eager.net.prefix_len].cpu()
        ref[(level,i,'V')]=v[:,:,:eager.net.prefix_len].cpu()
    def post(i,module,args,output):
        level=counters[i]
        ref[(level,i,'hidden')]=(output if level==0 else output[:,eager.net.prefix_len:]).cpu()
        counters[i]+=1
    for i,b in enumerate(eager.net.transformer.blocks):
        handles.append(b.attn.register_forward_pre_hook(lambda m,a,k,i=i:pre(i,m,a,k),with_kwargs=True))
        handles.append(b.register_forward_hook(lambda m,a,o,i=i:post(i,m,a,o)))
    sample=eager.net._sample_level_raw
    ref_logits=[]
    def sample_ref(logits,*args):
        ref_logits.append(logits.cpu())
        return sample(logits,*args)
    try:
        with patch.object(eager.net,'_sample_level_raw',sample_ref): baseline,_=run(eager,p,cfg)
    finally:
        for h in handles: h.remove()
    counters=[0]*len(counters)
    indices={id(b):i for i,b in enumerate(eager.net.transformer.blocks)}
    original=kv.block_forward
    def capture(block,x,bias,rope,prefix_kv=None):
        out,cache=original(block,x,bias,rope,prefix_kv)
        i=indices[id(block)];level=counters[i];counters[i]+=1
        actual[(level,i,'hidden')]=out.cpu()
        actual[(level,i,'K')]=cache[0].cpu()
        actual[(level,i,'V')]=cache[1].cpu()
        return out,cache
    act_logits=[]
    def sample_act(logits,*args):
        act_logits.append(logits.cpu())
        return sample(logits,*args)
    with patch.object(kv,'block_forward',capture),patch.object(eager.net,'_sample_level_raw',sample_act):
        result,_=run(cached,p,cfg)
    exact(result,baseline,f'intermediates cfg{cfg}')
    rows=[]
    pairs=[(str(k),ref[k],actual[k]) for k in ref]
    pairs.extend((f'logits-{i}',a,b) for i,(a,b) in enumerate(zip(ref_logits,act_logits)))
    for name,a,b in pairs:
        error=(a-b).abs()
        row={'name':name,'max_abs':error.max().item(),
             'max_tolerance_ratio':(error/(1e-4+1e-3*a.abs())).max().item()}
        rows.append(row)
        torch.testing.assert_close(a,b,atol=1e-4,rtol=1e-3,msg=str(row))
    return rows


def validate(args,report):
    import sacrebleu
    rt=load_wmt14_runtime(args.ncm_root,args.checkpoint_dir,args.tokenizer_dir)
    p,refs=load_data(rt,args.data_dir)
    assert len(p)==len(refs)==3003
    report['environment']={'torch':torch.__version__,'cuda':torch.version.cuda,
        'device':torch.cuda.get_device_name(),'dtype':'float32','tf32':False,
        'seed':SEED,'generation':str(rt.cfg.generation)}
    report['inputs']={x.name:hashlib.sha256(x.read_bytes()).hexdigest()
                      for x in args.data_dir.glob('*.pickle')}
    e=NCMLevelExecutor(rt,5,record_trace=False)
    c=NCMLevelExecutor(rt,5,prefix_cache=True,record_trace=False)
    report['model_id']=c.model_id
    report['boundaries']=[]
    report['intermediates']={}
    for cfg in (0,1):
        report['intermediates'][str(cfg)]=intermediates(e,c,p[:4],cfg)
        expected=native(rt,p[:4],cfg)
        for split in range(1,11):
            c.split=split
            actual,_=run(c,p[:4],cfg)
            exact(actual,expected,f'cfg{cfg} split{split}')
            report['boundaries'].append({'cfg':cfg,'split':split,'samples':4,'status':'PASS'})
    c.split=5
    lengths={kind:[[] for _ in rt.net.transformer.blocks] for kind in ('qkv','mlp')}
    hooks=[]
    for i,b in enumerate(rt.net.transformer.blocks):
        for kind,mod in (('qkv',b.attn.c_attn),('mlp',b.mlp)):
            hooks.append(mod.register_forward_pre_hook(lambda m,a,i=i,kind=kind:lengths[kind][i].append(a[0].shape[1])))
    try: run(c,p[:4],1)
    finally:
        for h in hooks:h.remove()
    expected_lengths=[96,4,8,14,20,26,32,40,48,56,64]
    assert all(x==expected_lengths for v in lengths.values() for x in v)
    report['positions']=lengths
    allocations=[]
    for i in range(20):
        s=c.submit('memory',p[:4],seed=SEED,cfg=1)
        c.short_group(s)
        s=c.handoff(s)
        c.cancel(s)
        assert s.prefix_kv is None and not c.states
        del s
        torch.cuda.synchronize()
        allocations.append(torch.cuda.memory_allocated())
    assert max(allocations[5:])-min(allocations[5:])<1024*1024
    report['memory']={'allocated_after_cycles':allocations,'reserved_bytes':torch.cuda.memory_reserved()}
    report['quality']={}
    dump(args.output,report)
    for cfg in (0,1):
        historic=json.loads((args.historical_dir/f'quality-native-capture-cfg{cfg}.json').read_text(encoding='utf8'))
        prior=json.loads((args.historical_dir/f'quality-harness-cfg{cfg}.json').read_text(encoding='utf8'))
        codes,ids,hyps=[],[],[]
        erng=crng=None
        for start in range(0,len(p),4):
            baseline,erng=run(e,p[start:start+4],cfg,erng)
            actual,crng=run(c,p[start:start+4],cfg,crng)
            try:
                exact(actual,baseline,f'cfg{cfg} sample{start}')
                assert baseline['codes'].tolist()==historic['codes'][start:start+4], 'new eager differs from historical codes'
                assert baseline['token_ids'].tolist()==historic['token_ids'][start:start+4], 'new eager differs from historical IDs'
            except AssertionError:
                report['first_failure']={'cfg':cfg,'batch_start':start,'prefix':p[start:start+4].tolist(),
                    'eager':{k:v.tolist() for k,v in baseline.items() if k!='latent'},
                    'cached':{k:v.tolist() for k,v in actual.items() if k!='latent'},
                    'matched_samples_before_failure':len(ids)}
                raise
            codes.extend(actual['codes'].tolist());ids.extend(actual['token_ids'].tolist())
            hyps.extend(rt.decode(actual['token_ids']))
            if start%100==0: print(f'quality cfg={cfg}: {min(start+4,len(p))}/{len(p)} exact',flush=True)
        assert hyps==[row['hypothesis'] for row in prior['rows']]
        metrics={}
        for name,metric in [('BLEU',sacrebleu.metrics.BLEU()),('chrF',sacrebleu.metrics.CHRF())]:
            score=metric.corpus_score(hyps,[refs])
            metrics[name]={'score':score.score,'signature':str(metric.get_signature())}
        report['quality'][str(cfg)]={'status':'PASS','count':len(ids),'codes_sha256':output_hash(codes),
            'token_ids_sha256':output_hash(ids),'metrics':metrics,'historical_and_eager_exact':True}
        dump(args.output.with_name(f'quality-cfg{cfg}.json'),{'codes':codes,'token_ids':ids,'hypotheses':hyps})
        dump(args.output,report)
    report['status']='PASS'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('ncm-root','checkpoint-dir','tokenizer-dir','data-dir','historical-dir','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    report={'status':'RUNNING','kind':'correctness-only'}
    try: validate(args,report)
    except BaseException as exc:
        report.update(status='FAILED',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        raise
    finally:dump(args.output,report)


if __name__=='__main__':main()
