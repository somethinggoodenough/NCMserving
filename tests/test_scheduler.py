from dataclasses import dataclass
import unittest

from ncm_serving.scheduler import RequestStatus, TwoGroupScheduler


@dataclass
class FakeState:
    request_id: str
    value: int
    phase: str = "submitted"


class FakeExecutor:
    def __init__(self):
        self.active = {}
        self.calls = []

    def submit(self, request_id, prefix, *, seed, cfg):
        state = FakeState(request_id, prefix + seed + int(cfg))
        self.active[request_id] = state
        self.calls.append(("submit", request_id))
        return state

    def short_group(self, state):
        assert self.active[state.request_id] is state
        assert state.phase == "submitted"
        state.phase = "short"
        self.calls.append(("short", state.request_id))

    def handoff(self, state):
        assert state.phase == "short"
        snapshot = FakeState(state.request_id, state.value, "transferred")
        self.active[state.request_id] = snapshot
        self.calls.append(("handoff", state.request_id))
        return snapshot

    def long_group(self, state):
        assert self.active[state.request_id] is state
        assert state.phase == "transferred"
        del self.active[state.request_id]
        self.calls.append(("long", state.request_id))
        return {"value": state.value * 2}

    def cancel(self, state):
        assert self.active[state.request_id] is state
        del self.active[state.request_id]
        self.calls.append(("cancel", state.request_id))


class SchedulerTests(unittest.TestCase):
    def test_lifecycle_and_long_queue_priority(self):
        executor = FakeExecutor()
        scheduler = TwoGroupScheduler(executor)
        first = scheduler.submit("a", 10, seed=2, cfg=0)
        second = scheduler.submit("b", 20, seed=3, cfg=1)

        scheduler.step()
        self.assertIs(first.status, RequestStatus.QUEUED_LONG)
        self.assertIs(second.status, RequestStatus.QUEUED_SHORT)

        completed = scheduler.step()
        self.assertIs(completed, first)
        self.assertIs(first.status, RequestStatus.COMPLETED)
        self.assertEqual(first.result, {"value": 24})
        self.assertEqual(scheduler.run_until_complete("b"), {"value": 48})
        self.assertEqual(scheduler.pending(), 0)
        self.assertFalse(executor.active)

    def test_cancel_releases_executor_state_and_queue_entry(self):
        executor = FakeExecutor()
        scheduler = TwoGroupScheduler(executor)
        scheduler.submit("a", 1, seed=1, cfg=0)
        scheduler.cancel("a")
        self.assertIs(scheduler.requests["a"].status, RequestStatus.CANCELLED)
        self.assertEqual(scheduler.pending(), 0)
        self.assertFalse(executor.active)
        with self.assertRaisesRegex(ValueError, "cannot cancel"):
            scheduler.cancel("a")

    def test_duplicate_request_is_rejected_before_executor_mutation(self):
        executor = FakeExecutor()
        scheduler = TwoGroupScheduler(executor)
        scheduler.submit("a", 1, seed=1)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            scheduler.submit("a", 2, seed=2)
        self.assertEqual(list(executor.active), ["a"])

    def test_idle_step_and_unknown_request(self):
        scheduler = TwoGroupScheduler(FakeExecutor())
        self.assertIsNone(scheduler.step())
        with self.assertRaises(KeyError):
            scheduler.run_until_complete("missing")


if __name__ == "__main__":
    unittest.main()
