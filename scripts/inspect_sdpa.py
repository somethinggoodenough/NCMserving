"""Record actual SDPA dispatcher operators without making kernel-time claims.

This post-matrix diagnostic uses the same compute path and checks profiler
transparency. CPU profiler activities identify dispatch; CUDA timing remains
the separate B01/B03 experiment. Run through gpu_budget.py.
"""
import argparse
import platform
import sys
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from benchmarks.prefix_kv import compute
from validate_prefix_kv import dump, load_data
from ncm_serving.runtime import load_wmt14_runtime
from ncm_serving.service import NCMServingSystem


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('ncm-root','checkpoint-dir','tokenizer-dir','data-dir','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    report={'status':'RUNNING','kind':'dispatch diagnostic; not performance timing',
            'python':sys.version,'platform':platform.platform(),'cells':[]}
    try:
        rt=load_wmt14_runtime(args.ncm_root,args.checkpoint_dir,args.tokenizer_dir)
        prefixes,_=load_data(rt,args.data_dir)
        servers={'U0':None,'E0':NCMServingSystem(rt,record_trace=False),
                 'K1':NCMServingSystem(rt,prefix_cache=True,record_trace=False)}
        for batch in (1,4,8):
            for cfg in (0,1):
                for variant,server in servers.items():
                    expected=compute(rt,server,prefixes[:batch],cfg)
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                                record_shapes=True) as prof:
                        actual=compute(rt,server,prefixes[:batch],cfg)
                    assert torch.equal(actual,expected),'profiler changed output'
                    operators=[{'name':event.key,'count':event.count,'input_shapes':event.input_shapes}
                               for event in prof.key_averages(group_by_input_shape=True)
                               if 'scaled_dot_product' in event.key or 'efficient_attention' in event.key]
                    assert operators,'no attention dispatcher events captured'
                    report['cells'].append({'variant':variant,'batch':batch,'cfg':cfg,
                                            'ids_exact':True,'operators':operators})
        report['status']='PASS'
    except BaseException as exc:
        report.update(status='FAILED',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:dump(args.output,report)


if __name__=='__main__':main()
