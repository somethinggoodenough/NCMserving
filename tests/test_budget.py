"""The GPU deadline must stop the Windows venv redirector AND its child."""
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from scripts.gpu_budget import stop_tree


@unittest.skipUnless(os.name=='nt','Windows process-tree regression')
class BudgetTests(unittest.TestCase):
    def test_redirected_worker_tree_is_stopped(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'heartbeat'
            code='import time,sys; from pathlib import Path\np=Path(sys.argv[1])\nfor i in range(100):\n p.write_text(str(i)); time.sleep(.05)\n'
            worker=subprocess.Popen([sys.executable,'-c',code,str(path)])
            try:
                deadline=time.monotonic()+5
                while not path.exists() and time.monotonic()<deadline:time.sleep(.05)
                self.assertTrue(path.exists())
                stop_tree(worker)
                before=path.read_text()
                time.sleep(.2)
                self.assertEqual(before,path.read_text())
                self.assertIsNotNone(worker.poll())
            finally:
                if worker.poll() is None:stop_tree(worker)
