"""Crash injection for recovery tests.

`STEPWISE_FAULT=<point>` makes the worker process die with SIGKILL semantics (no cleanup, no
finally blocks, no lease release) the first time execution reaches that point. Points:

    after_llm_call        model responded, response not yet journaled
    before_tool_commit    transactional tool ran, its transaction not yet committed
    before_external_call  `started` step journaled, external call not yet made
    after_external_call   external call succeeded, result not yet journaled
"""

import logging
import os

logger = logging.getLogger(__name__)

CRASH_EXIT_CODE = 137

FAULT_POINTS = frozenset({"after_llm_call", "before_tool_commit", "before_external_call", "after_external_call"})


class FaultInjector:
    def __init__(self, point: str | None) -> None:
        if point is not None and point not in FAULT_POINTS:
            raise ValueError(f"unknown fault point {point!r}; expected one of {sorted(FAULT_POINTS)}")
        self._point = point

    def hit(self, point: str) -> None:
        if point == self._point:
            logger.warning("fault injection: crashing at %s", point)
            os._exit(CRASH_EXIT_CODE)


NO_FAULTS = FaultInjector(None)
