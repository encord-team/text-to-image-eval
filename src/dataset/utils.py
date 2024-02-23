import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor as Executor
from concurrent.futures import as_completed
from pathlib import Path
from typing import Any, TypeVar

import requests
from tqdm.auto import tqdm

T = TypeVar("T")
G = TypeVar("G")

_default_max_workers = min(10, (os.cpu_count() or 1) + 4)


def collect_async(
    fn: Callable[[T], G],
    job_args: list[T],
    max_workers=_default_max_workers,
    **kwargs,
) -> list[G]:
    """
    Distribute work across multiple workers. Good for, e.g., downloading data.
    Will return results in dictionary.
    :param fn: The function to be applied
    :param job_args: Arguments to `fn`.
    :param max_workers: Number of workers to distribute work over.
    :param kwargs: Arguments passed on to tqdm.
    :return: List [fn(*job_args)]
    """
    if len(job_args) == 0:
        tmp: list[G] = []
        return tmp
    if not isinstance(job_args[0], tuple):
        _job_args: list[tuple[Any]] = [(j,) for j in job_args]
    else:
        _job_args = job_args  # type: ignore

    results: list[G] = []
    with tqdm(total=len(job_args), **kwargs) as pbar:
        with Executor(max_workers=max_workers) as exe:
            jobs = [exe.submit(fn, *args) for args in _job_args]
            for job in as_completed(jobs):
                result = job.result()
                if result is not None:
                    results.append(result)
                pbar.update(1)
    return results


def download_file(
    url: str,
    destination: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "wb") as f:
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            raise ConnectionError(f"Something happened, couldn't download file from: {url}")

        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
        f.flush()
