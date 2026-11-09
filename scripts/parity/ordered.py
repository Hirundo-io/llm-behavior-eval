"""Supply saved samples to Inspect score in retained batch-one order."""

import asyncio
from collections.abc import AsyncGenerator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from inspect_ai.log import EvalLog, EvalSample


def ordered_samples(
    log: EvalLog,
    ids: list[str | int] | None = None,
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
    if ids is not None:
        by_id = {sample.id: sample for sample in samples}
        if (
            len(by_id) != len(samples)
            or set(by_id) != set(ids)
            or len(ids) != len(samples)
        ):
            raise ValueError("Pinned source IDs do not uniquely match saved samples")
        # Log readers may sort samples by ID. Select references in the actual
        # source order; leave the samples and their original saved log untouched.
        samples = [by_id[ident] for ident in ids]

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
