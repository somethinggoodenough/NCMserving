"""Audit F01 raw evidence and produce the small versioned result summary."""
import argparse
import array
from collections import Counter, defaultdict
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from benchmarks.run_prefix_matrix import summarize


def read(path):return json.loads(path.read_text(encoding='utf8'))
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(rows):
    values=array.array('i',(v for row in rows for v in row))
    assert values.itemsize==4
    if sys.byteorder!='little':values.byteswap()
    return hashlib.sha256(values.tobytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    raw=args.raw_dir;matrix=raw/'matrix-01'
    quality=read(raw/'quality-01.json');ledger=read(raw/'gpu-budget.json')
    manifest=read(matrix/'manifest.json');preflight=read(matrix/'preflight-U0-b4-c0-w0.json')
    assert quality['status']=='PASS' and len(quality['boundaries'])==20
    assert preflight['status']=='PASS' and len(preflight['cells'])==6
    assert all(c['status']=='PASS' and c['instrumentation_exact'] for c in preflight['cells'])
    assert all(r['status']=='PASS' and r['source_before']==r['source_after'] for r in ledger['runs'])
    assert math.isclose(sum(r['wall_seconds'] for r in ledger['runs']),ledger['used_seconds'])
    assert ledger['used_seconds']<=ledger['cap_seconds']==10800
    performance_runs=[r for r in ledger['runs'] if 'e2e' in r['command']]
    assert len(performance_runs)==90
    for r in performance_runs:
        assert all(r['source_before'].get(k)==v for k,v in manifest['source'].items()),'matrix source mismatch'
    result=summarize(matrix)
    assert len(result['cells'])==6 and all(c['status'] in ('NO_GAIN','STABLE_GAIN') for c in result['cells'])
    for cell in result['cells']:
        cell['paired_latency_improvement_median']={k:statistics.median(v) for k,v in cell['paired_latency_improvement'].items()}
    phases=defaultdict(list);hashes=defaultdict(set);warmups=[];loading=[];files={}
    windows=list(matrix.glob('e2e-*.json'))
    assert len(windows)==90
    for path in windows:
        r=read(path);files[path.name]=digest(path)
        assert r['status']=='PASS' and r['quality_report_sha256']==manifest['quality_sha256']
        assert r['completed_samples']==1024 and r['completed_batches']==1024//r['batch']
        assert len(r['batch_seconds'])==r['completed_batches'] and min(r['batch_seconds'])>0
        assert math.isclose(sum(r['batch_seconds']),r['seconds'])
        assert math.isclose(r['requests_per_second'],1024/r['seconds'])
        assert math.isclose(r['tokens_per_second'],r['actual_tokens']/r['seconds'])
        hashes[r['cfg']].add(r['ids_sha256'])
        warm=r['warmup_seconds'];warmups.append(len(warm));loading.append(r['load_and_fingerprint_seconds'])
        assert 10<=len(warm)<=50
        assert abs(statistics.median(warm[-10:-5])-statistics.median(warm[-5:]))/statistics.median(warm[-10:-5])<=.05
        rows=r['profile']['phases'];counts=Counter(x['phase'].split('/')[0] for x in rows)
        assert r['profile']['batches']==16
        for name,count in [('sample',176),('block',2112),('attention',2112),('mlp',2112),('decoder',16),('token_selection',16)]:
            assert counts[name]==count,(path,name,counts[name])
        cache_bytes=2*12*r['batch']*(1+r['cfg'])*96*768*4 if r['variant']=='K1' else 0
        assert r['profile']['cache_bytes']==cache_bytes
        assert counts['prefix_prefill']==(16 if r['variant']=='K1' else 0)
        totals=defaultdict(lambda:[0.,0.])
        for x in rows:
            assert math.isfinite(x['cuda_span_ms']) and x['cuda_span_ms']>=0
            assert math.isfinite(x['cpu_call_ms']) and x['cpu_call_ms']>=0
            key=(x['phase'].split('/')[0],x['level'])
            totals[key][0]+=x['cuda_span_ms']/16
            totals[key][1]+=x['cpu_call_ms']/16
        for (name,level),values in totals.items():
            phases[(r['variant'],r['batch'],r['cfg'],name,level)].append(values)
    assert all(len(v)==1 for v in hashes.values()),'batch shape/window/variant output mismatch'
    result['phase_medians']=[{'variant':v,'batch':b,'cfg':c,'phase':name,'level':level,
        'cuda_span_ms_per_batch':statistics.median(x[0] for x in rows),
        'cpu_call_ms_per_batch':statistics.median(x[1] for x in rows)}
        for (v,b,c,name,level),rows in sorted(phases.items())]
    assert all(len(v)==5 for v in phases.values())
    result.update(status='PASS',implementation_commit=manifest['source_commit'],
        formal_windows=90,formal_completed_samples=90*1024,profile_windows=90,profile_batches=90*16,
        maximum_window_cv=max(s['window_seconds_cv'] for c in result['cells'] for s in c['paths'].values()),
        warmup_batches_range=[min(warmups),max(warmups)],loading_seconds_range=[min(loading),max(loading)],
        quality=quality['quality'],canonical_output_hashes={},gpu_budget_seconds=ledger['used_seconds'],
        gpu_budget_cap_seconds=ledger['cap_seconds'],gpu_workers=len(ledger['runs']),
        raw_file_hashes=files,source_and_count_audit='PASS',
        resource_and_upstream_hashes={Path(k).relative_to(ROOT.parents[1]).as_posix():v
                                      for k,v in manifest['resources'].items()},
        phase_measurement_note='Nested CUDA event spans and CPU call wall time; not exclusive kernel times; do not sum nested spans',
        e2e_boundary='CPU pretokenized prefix -> CPU final IDs; includes prefill/handoff/request cleanup; excludes loading/text/scoring/output hashing')
    for cfg in (0,1):
        q=read(raw/f'quality-cfg{cfg}.json')
        assert len(q['codes'])==len(q['token_ids'])==len(q['hypotheses'])==3003
        result['canonical_output_hashes'][str(cfg)]={k:canonical_hash(q[k]) for k in ('codes','token_ids')}
    dispatch=read(raw/'sdpa-dispatch.json')
    assert dispatch['status']=='PASS' and len(dispatch['cells'])==18
    result['dispatch_operators']=sorted({op['name'] for cell in dispatch['cells'] for op in cell['operators']})
    result['dispatch_diagnostic_sha256']=digest(raw/'sdpa-dispatch.json')
    result['audit_evidence_sha256']={str(p.relative_to(raw).as_posix()):digest(p) for p in (
        raw/'quality-01.json',raw/'gpu-budget.json',matrix/'manifest.json',
        matrix/'preflight-U0-b4-c0-w0.json',raw/'cpu-tests.log')}
    result['python']=dispatch['python'];result['platform']=dispatch['platform']
    result['raw_mib']=sum(p.stat().st_size for p in raw.rglob('*') if p.is_file())/2**20
    assert result['raw_mib']<256
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print('PASS: 90 E2E + 90 profile windows, source/count/output/budget audit')
    print('GPU seconds:',ledger['used_seconds'])
    print('Dispatch:',result['dispatch_operators'])


if __name__=='__main__':main()
