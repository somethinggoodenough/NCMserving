"""CPU checks for benchmark denominators and output-token accounting."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from benchmarks.prefix_kv import tokens, quantile
from benchmarks.run_prefix_matrix import summarize
from scripts.gpu_budget import dump


class BenchmarkChecks(unittest.TestCase):
    def test_counts_and_quantiles(self):
        ids=torch.tensor([[5,6,1,7],[0,8,0,1],[9,9,9,9]])
        self.assertEqual(tokens(ids,SimpleNamespace(eos=1,pad=0)),7)
        self.assertEqual(quantile([4,1,3,2],.5),2.5)
        self.assertAlmostEqual(quantile([4,1,3,2],.95),3.85)

    def test_paired_gain_requires_five_valid_windows_and_identical_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            self.assertEqual(summarize(folder)['cells'][0]['status'],'INCOMPLETE')
            for w in range(5):
                for v,seconds in [('U0',100),('E0',110),('K1',80)]:
                    dump(folder/f'e2e-{v}-b1-c0-w{w}.json',{
                        'status':'PASS','ids_sha256':'same','completed_samples':1024,
                        'seconds':seconds,'requests_per_second':1024/seconds,
                        'batch_p50_ms':1,'batch_p95_ms':2,'tokens_per_second':30,
                        'peak_allocated_bytes':1,'peak_reserved_bytes':2})
            cell=summarize(folder)['cells'][0]
            self.assertEqual(cell['status'],'STABLE_GAIN')
            self.assertAlmostEqual(cell['paired_latency_improvement']['E0'][0],1-80/110)
            path=folder/'e2e-K1-b1-c0-w0.json'
            import json
            row=json.loads(path.read_text());row['ids_sha256']='wrong';dump(path,row)
            with self.assertRaisesRegex(AssertionError,'output mismatch'):summarize(folder)


if __name__=='__main__':unittest.main()
