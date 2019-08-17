"""
:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2018-05-28
:Copyright: 2018, Karr Lab
:License: MIT
"""

import unittest
import re
from capturer import CaptureOutput


class TestSimulationRun(unittest.TestCase):

    @unittest.skip("faster tests")
    def test_simulation_run(self):
        with CaptureOutput(relay=False) as capturer:
            import examples.simulation_run

            events = re.search(r'Simulated (\d+) events', capturer.get_text())
            num_events = int(events.group(1))
            self.assertTrue(0 < num_events)
            results = re.search("Saved checkpoints and run results in '(.*?)'", capturer.get_text())
            self.assertTrue(results is not None)
