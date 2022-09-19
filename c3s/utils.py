import numpy as np
import time
from tqdm.auto import tqdm


class timeit:
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

def split_tasks_for_workers(N_tasks, N_workers, rank):
    """Makes approximately even sized slices for some set of parallel workers
    to use as indices for their local block of work."""

    if N_workers is None or N_workers == 1:
        start, stop, blocksize = 0, N_tasks, N_tasks
        return start, stop, blocksize

    # begin by making evenly sized block with the remainder truncated
    blocksizes = np.ones(N_workers, dtype=np.int64) * N_tasks // N_workers
    # sprinkle the remainder over the workers
    remainders = (np.arange(N_workers, dtype=np.int64) < N_tasks % N_workers)
    blocksizes += remainders
    # the boundaries of the slices
    ids = np.cumsum(np.concatenate(([0], blocksizes)))
    # use block boundaries as indices for slices
    slices = [slice(start, stop, 1) for start, stop in zip(ids[:-1], ids[1:])]
    start = slices[rank].start
    stop = slices[rank].stop
    blocksize = stop - start
    print(start, stop, blocksize)

    return start, stop, blocksize

def latexify_string(string):
    """helper function to prepare strings for raw string format

    this function still needs some work...

    """

    for symbol in ('_', '^'):
        if symbol in string:
            split_string = string.split(symbol)
            assert len(split_string) == 2
            # need lots of {} for f-string formatting
            string = split_string[0] + symbol + f'{{{split_string[1]}}}'
    if '*' in string:
        string = string.replace('*', '^*')
    # dress with $
    string = '$' + string + '$'

    return string

class ProgressBar(tqdm):
    """tqdm progress bar with the default settings I want."""

    def __init__(self, *args, **kwargs):
        """"""
        super(ProgressBar, self).__init__(leave=False, dynamic_ncols=False, *args, **kwargs)
