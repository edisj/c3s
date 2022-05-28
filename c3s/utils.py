import numpy as np
import time
from tqdm.auto import tqdm
from typing import List

class timeit(object):
    """Measures the time elapsed during the execution of a block of code.

    :class:`timeit` is meant to be used as a context manager where you wrap a section of code with a
    `with timeit() as time_this_block:` where the elapsed time will be stored in the
    `time_this_block.elapsed` attribute.

    """
    def __enter__(self):
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        self.elapsed = end_time - self._start_time
        return False


def slice_tasks_for_parallel_workers(n_tasks: int, n_workers: int) -> List[slice]:
    """Makes approximately even sized slices for some set of parallel workers
    to use as indices for their local block of work.

    Note
    ----
    If `mpi4py` is not installed and the

    """

    # begin by making evenly sized block with the remainder truncated
    blocksizes = np.ones(n_workers, dtype=np.int64) * n_tasks // n_workers
    # sprinkle the remainder over the workers
    remainders = (np.arange(n_workers, dtype=np.int64) < n_tasks % n_workers)
    blocksizes += remainders
    # the boundaries of the slices
    ids = np.cumsum(np.concatenate(([0], blocksizes)))
    # use block boundaries as indices for slices
    slices = [slice(start, stop, 1) for start, stop in zip(ids[:-1], ids[1:])]
    # fixes the last block's stop index in cases where it equals n_tasks
    #last = slices[-1]
    #last_stop = min(last.stop, n_tasks-1)
    #slices[-1] = slice(last.start, last_stop, last.step)

    return slices

class ProgressBar(tqdm):
    """tqdm progress bar with the default settings I want."""

    def __init__(self, *args, **kwargs):
        """"""
        super(ProgressBar, self).__init__(leave=False, dynamic_ncols=True,
                                          bar_format='{desc}:{percentage:3.0f}%|{bar}[{elapsed}]',
                                          *args, **kwargs)
