"""Request lifecycle and two-group queue scheduling."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from time import monotonic_ns
from typing import Any


class RequestStatus(str, Enum):
    QUEUED_SHORT = "queued_short"
    QUEUED_LONG = "queued_long"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class RequestRecord:
    request_id: str
    state: Any
    status: RequestStatus
    submitted_ns: int = field(default_factory=monotonic_ns)
    short_finished_ns: int | None = None
    completed_ns: int | None = None
    result: Any = None
    error: str | None = None


class TwoGroupScheduler:
    """Schedule logical A/B groups while keeping request state isolated.

    One call to ``step`` runs one complete group. Long-group work is selected
    first so transferred state is released before admitting more A work. This
    scheduler is intentionally single-threaded and single-device.
    """

    def __init__(self, executor: Any):
        self.executor = executor
        self.requests: dict[str, RequestRecord] = {}
        self.short_queue: deque[str] = deque()
        self.long_queue: deque[str] = deque()

    def submit(self, request_id: str, prefix: Any, *, seed: int, cfg: float = 0.0) -> RequestRecord:
        if request_id in self.requests:
            raise ValueError("duplicate request_id")
        state = self.executor.submit(request_id, prefix, seed=seed, cfg=cfg)
        record = RequestRecord(request_id, state, RequestStatus.QUEUED_SHORT)
        self.requests[request_id] = record
        self.short_queue.append(request_id)
        return record

    def step(self) -> RequestRecord | None:
        if self.long_queue:
            request_id = self.long_queue.popleft()
            record = self.requests[request_id]
            try:
                record.result = self.executor.long_group(record.state)
                record.state = None
                record.status = RequestStatus.COMPLETED
                record.completed_ns = monotonic_ns()
            except BaseException as exc:
                if hasattr(self.executor, 'abort'):
                    self.executor.abort(record.state)
                record.state = None
                record.status = RequestStatus.FAILED
                record.error = f"{type(exc).__name__}: {exc}"
                raise
            return record

        if self.short_queue:
            request_id = self.short_queue.popleft()
            record = self.requests[request_id]
            try:
                self.executor.short_group(record.state)
                record.state = self.executor.handoff(record.state)
                record.status = RequestStatus.QUEUED_LONG
                record.short_finished_ns = monotonic_ns()
                self.long_queue.append(request_id)
            except BaseException as exc:
                if hasattr(self.executor, 'abort'):
                    self.executor.abort(record.state)
                record.state = None
                record.status = RequestStatus.FAILED
                record.error = f"{type(exc).__name__}: {exc}"
                raise
            return record
        return None

    def run_until_complete(self, request_id: str) -> Any:
        record = self.requests[request_id]
        while record.status not in (
            RequestStatus.COMPLETED,
            RequestStatus.CANCELLED,
            RequestStatus.FAILED,
        ):
            if self.step() is None:
                raise RuntimeError("scheduler became idle with an unfinished request")
        if record.status is RequestStatus.COMPLETED:
            return record.result
        raise RuntimeError(f"request ended with status {record.status.value}: {record.error or ''}")

    def cancel(self, request_id: str) -> None:
        record = self.requests[request_id]
        if record.status in (
            RequestStatus.COMPLETED,
            RequestStatus.CANCELLED,
            RequestStatus.FAILED,
        ):
            raise ValueError(f"cannot cancel request in state {record.status.value}")
        self.short_queue = deque(item for item in self.short_queue if item != request_id)
        self.long_queue = deque(item for item in self.long_queue if item != request_id)
        self.executor.cancel(record.state)
        record.state = None
        record.status = RequestStatus.CANCELLED
        record.completed_ns = monotonic_ns()

    def pending(self) -> int:
        return len(self.short_queue) + len(self.long_queue)
