"""Supply saved samples to Inspect score in retained batch-one order."""

import asyncio
from collections.abc import AsyncGenerator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from inspect_ai.log import EvalLog, EvalSample


def ordered_samples(
    log: EvalLog,
) -> Callable[[int], AbstractAsyncContextManager[EvalSample]]:
    """Use score_async's public sample-provider API without rewriting samples.

    Pass copy=False: score updates these in-memory samples, then writes a separate
    scored artifact. The original answer log stays immutable on disk.
    """
    condition = asyncio.Condition()
    next_index = 0
    samples = log.samples
    if samples is None:
        raise ValueError("Missing saved answer samples")

    @asynccontextmanager
    async def read(index: int) -> AsyncGenerator[EvalSample, None]:
        nonlocal next_index
        async with condition:
            await condition.wait_for(lambda: index == next_index)
            try:
                yield samples[index]
            finally:
                next_index += 1
                condition.notify_all()

    return read
