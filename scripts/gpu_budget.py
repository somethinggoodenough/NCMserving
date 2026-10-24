"""Run one GPU worker under the approved cumulative F01 wall-time budget.

Usage: python scripts/gpu_budget.py --ledger PATH -- <python worker arguments>
The ledger charges the whole subprocess, including imports and loading. An
exclusive lock prevents overlapping invocations. Interrupted runs fail closed.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAP = 10800.0


def dump(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n', encoding='utf8')
    temporary.replace(path)


def fingerprint(external_root=None):
    paths = [p for folder in ('src', 'scripts', 'benchmarks', 'tests')
             for p in (ROOT/folder).rglob('*.py')]
    result = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(paths)}
    if external_root is not None:
        for pattern in ('*.py','*.yaml'):
            for p in sorted(external_root.rglob(pattern)):
                result['upstream/'+p.relative_to(external_root).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
    return result


def stop_tree(worker):
    # Windows venv python.exe is a redirector that launches another python.exe.
    # Killing only the redirector can leave GPU work alive beyond the cap.
    if os.name=='nt':
        subprocess.run(['taskkill','/PID',str(worker.pid),'/T','/F'],
                       stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=5)
    if worker.poll() is None: worker.kill()
    worker.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    lock = args.ledger.with_suffix('.lock')
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        ledger = json.loads(args.ledger.read_text()) if args.ledger.exists() else {
            'cap_seconds': CAP, 'accounting': 'whole child process wall time; one GPU', 'runs': []}
        if ledger['cap_seconds'] != CAP or any(r['status']=='RUNNING' for r in ledger['runs']):
            raise RuntimeError('budget cap mismatch or interrupted run needs conservative reconciliation')
        remaining = CAP - sum(r['wall_seconds'] for r in ledger['runs'])
        if remaining <= 10:
            raise RuntimeError('GPU budget exhausted')
        command = args.command[1:] if args.command[:1] == ['--'] else args.command
        if not command:
            raise ValueError('worker arguments required after --')
        external_root = (Path(command[command.index('--ncm-root')+1]).resolve()
                         if '--ncm-root' in command else None)
        command = [sys.executable, *command]
        record = {'command': command, 'started_utc': datetime.now(timezone.utc).isoformat(),
                  'status': 'RUNNING', 'remaining_before_seconds': remaining,
                  'source_before': fingerprint(external_root)}
        ledger['runs'].append(record)
        dump(args.ledger, ledger)
        start = time.monotonic()
        worker = None
        try:
            log = args.ledger.parent/f'worker-{len(ledger["runs"]):03}.log'
            record['log'] = str(log)
            with log.open('w', encoding='utf8') as stream:
                worker = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                # Poll for changes during execution, not just at worker exit.
                while worker.poll() is None:
                    if time.monotonic()-start >= remaining-10:
                        record['status']='BUDGET_EXHAUSTED'
                        stop_tree(worker)
                        break
                    if fingerprint(external_root) != record['source_before']:
                        record['status']='SOURCE_CHANGED'
                        stop_tree(worker)
                        break
                    time.sleep(.5)
                code = worker.wait()
            if record['status']=='RUNNING': record['status']='PASS' if code==0 else 'FAILED'
            record['returncode']=code
        finally:
            if worker is not None and worker.poll() is None:
                stop_tree(worker)
            record['wall_seconds']=time.monotonic()-start
            if record['status']=='RUNNING': record['status']='INTERRUPTED'
            record['source_after']=fingerprint(external_root)
            ledger['used_seconds']=sum(r['wall_seconds'] for r in ledger['runs'])
            dump(args.ledger, ledger)
            print(json.dumps({k:record[k] for k in ('status','wall_seconds')}), flush=True)
        if record['status']!='PASS': raise SystemExit(1)
    finally:
        os.close(fd)
        lock.unlink()


if __name__=='__main__': main()
