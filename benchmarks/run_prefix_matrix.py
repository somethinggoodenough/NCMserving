"""Sequential F01 matrix controller: five paired, rotating three-path windows."""
import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from gpu_budget import fingerprint, dump


def file_hash(path):
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):digest.update(data)
    return digest.hexdigest()


def quantile(values,q):
    s=sorted(values);p=(len(s)-1)*q;i=int(p)
    return s[i]+(s[min(i+1,len(s)-1)]-s[i])*(p-i)


def summarize(folder):
    cells=[]
    for b in (1,4,8):
        for cfg in (0,1):
            paths={v:[folder/f'e2e-{v}-b{b}-c{cfg}-w{w}.json' for w in range(5)] for v in ('U0','E0','K1')}
            if not all(p.exists() for ps in paths.values() for p in ps):
                cells.append({'batch':b,'cfg':cfg,'status':'INCOMPLETE'})
                continue
            rows={v:[json.loads(p.read_text()) for p in ps] for v,ps in paths.items()}
            if any(r['status']!='PASS' for rs in rows.values() for r in rs):
                cells.append({'batch':b,'cfg':cfg,'status':'FAILED_OR_OOM'})
                continue
            for w in range(5):
                assert len({rows[v][w]['ids_sha256'] for v in rows})==1,'measured output mismatch'
                assert all(rows[v][w]['completed_samples']==1024 for v in rows)
            stats={}
            for v,rs in rows.items():
                seconds=[r['seconds'] for r in rs]
                stats[v]={'requests_per_second_median':statistics.median([r['requests_per_second'] for r in rs]),
                    'requests_per_second_iqr':quantile([r['requests_per_second'] for r in rs],.75)-quantile([r['requests_per_second'] for r in rs],.25),
                    'window_seconds':seconds,'window_seconds_cv':statistics.stdev(seconds)/statistics.mean(seconds),
                    'batch_p50_ms_median':statistics.median([r['batch_p50_ms'] for r in rs]),
                    'batch_p95_ms_median':statistics.median([r['batch_p95_ms'] for r in rs]),
                    'tokens_per_second_median':statistics.median([r['tokens_per_second'] for r in rs]),
                    'peak_allocated_bytes':max(r['peak_allocated_bytes'] for r in rs),
                    'peak_reserved_bytes':max(r['peak_reserved_bytes'] for r in rs)}
            improvement={v:[1-rows['K1'][w]['seconds']/rows[v][w]['seconds'] for w in range(5)] for v in ('U0','E0')}
            stable=all(s['window_seconds_cv']<=.1 for s in stats.values())
            gain=statistics.median(improvement['E0'])>=.05 and min(improvement['E0'])>0
            cells.append({'batch':b,'cfg':cfg,'status':('STABLE_GAIN' if gain else 'NO_GAIN') if stable else 'NO_STABLE_MEASUREMENT',
                'paths':stats,'paired_latency_improvement':improvement,'ids_exact_all_windows':True})
    return {'cells':cells,'kind':'F01 single-device five-window benchmark'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('ncm-root','checkpoint-dir','tokenizer-dir','data-dir','quality','output-dir','ledger'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    workspace=ROOT.parents[1]
    workspace_bytes=sum(x.stat().st_size for x in workspace.rglob('*') if x.is_file())
    if workspace_bytes>20*1024**3:raise RuntimeError('workspace disk cap exceeded')
    files=[*args.ncm_root.rglob('*.py'),*args.ncm_root.rglob('*.yaml'),
           *args.checkpoint_dir.glob('*.pt'),*args.tokenizer_dir.glob('*'),*args.data_dir.glob('*.pickle')]
    resources={str(x.resolve()):file_hash(x) for x in files if x.is_file()}
    resource_stats={name:(Path(name).stat().st_size,Path(name).stat().st_mtime_ns) for name in resources}
    manifest={'source':fingerprint(),'resources':resources,'quality_sha256':file_hash(args.quality),
        'workspace_bytes_before':workspace_bytes,
        'schedule':'window -> batch -> CFG -> rotating U0/E0/K1; 1024 samples + separate 16-batch profile pass',
        'source_diff':subprocess.run(['git','diff','HEAD'],cwd=ROOT,capture_output=True,text=True,check=True).stdout,
        'source_commit':subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip(),
        'driver':subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()}
    manifest_path=args.output_dir/'manifest.json'
    if manifest_path.exists():raise FileExistsError('matrix output directory must be fresh')
    dump(manifest_path,manifest)
    frozen=fingerprint()
    base=[sys.executable,'scripts/gpu_budget.py','--ledger',str(args.ledger),'--','benchmarks/prefix_kv.py']
    for name in ('ncm_root','checkpoint_dir','tokenizer_dir','data_dir','quality'):
        base.extend(['--'+name.replace('_','-'),str(getattr(args,name))])
    def invoke(mode,variant,b,cfg,w):
        output=args.output_dir/f'{mode}-{variant}-b{b}-c{cfg}-w{w}.json'
        if fingerprint()!=frozen:raise RuntimeError('matrix source changed')
        if any((Path(n).stat().st_size,Path(n).stat().st_mtime_ns)!=s for n,s in resource_stats.items()):
            raise RuntimeError('external resource changed')
        size=sum(x.stat().st_size for x in args.output_dir.parent.rglob('*') if x.is_file())
        if size>256*1024**2:raise RuntimeError('F01 incremental disk budget exceeded')
        command=base+['--mode',mode,'--variant',variant,'--batch',str(b),'--cfg',str(cfg),'--window',str(w),'--output',str(output)]
        print(f'START {output.name}',flush=True)
        completed=subprocess.run(command,cwd=ROOT)
        if completed.returncode:
            if b==8 and output.exists() and json.loads(output.read_text())['status']=='OOM':return False
            raise RuntimeError(f'worker failed: {output}')
        print(f'DONE {output.name}',flush=True)
        return True
    invoke('preflight','U0',4,0,0)
    preflight=json.loads((args.output_dir/'preflight-U0-b4-c0-w0.json').read_text())
    capacity={(r['batch'],r['cfg']) for r in preflight['cells'] if r['status']=='OOM'}
    try:
        for w in range(5):
            order=['U0','E0','K1'];order=order[w%3:]+order[:w%3]
            for b in (1,4,8):
                for cfg in (0,1):
                    if (b,cfg) in capacity:continue
                    for v in order:
                        if not invoke('e2e',v,b,cfg,w):
                            capacity.add((b,cfg));break
                        dump(args.output_dir/'summary.json',summarize(args.output_dir))
    finally:
        result=summarize(args.output_dir)
        result['capacity_oom_cells']=list(capacity)
        dump(args.output_dir/'summary.json',result)
    if fingerprint()!=frozen:raise RuntimeError('matrix source changed')
    if any(file_hash(Path(name))!=digest for name,digest in resources.items()):
        raise RuntimeError('external resource contents changed')


if __name__=='__main__':main()
