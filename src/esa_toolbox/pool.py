"""A process pool that behaves the same on Linux, macOS and Windows (spawn), with one GDAL thread per worker."""

from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Iterable


def _init() -> None:
    os.environ["GDAL_NUM_THREADS"] = "1"
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def run(func: Callable, jobs: Iterable[tuple], workers: int,
        on_done: Callable[[int, int], None] | None = None) -> list:
    """Call func(*job) for every job, in `workers` processes, and return results in the order of the jobs."""
    jobs = list(jobs)
    if workers <= 1 or len(jobs) <= 1:
        out = []
        for i, j in enumerate(jobs):
            out.append(func(*j))
            if on_done:
                on_done(i + 1, len(jobs))
        return out
    results: list = [None] * len(jobs)
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=ctx, initializer=_init) as ex:
        futures = {ex.submit(func, *j): i for i, j in enumerate(jobs)}
        done = 0
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()
            done += 1
            if on_done:
                on_done(done, len(jobs))
    return results
