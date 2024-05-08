import importlib.util
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor as Executor
from concurrent.futures import as_completed
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import requests
from tqdm.auto import tqdm

from tti_eval.common import Split

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


def load_class_from_path(module_path: str, class_name: str):
    spec = importlib.util.spec_from_file_location(module_path, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def simple_random_split(
    dataset_size: int,
    seed: int = 42,
    train_split: float = 0.7,
    validation_split: float = 0.15,
) -> dict[Split, np.ndarray]:
    """
    Split the dataset into training, validation, and test sets using simple random splitting.

    :param dataset_size: The total size of the dataset.
    :param seed: Random seed for reproducibility. Defaults to 42.
    :param train_split: Percentage of the dataset to allocate to the training set. Defaults to 0.7.
    :param validation_split: Percentage of the dataset to allocate to the validation set. Defaults to 0.15.
    :return: A dictionary containing arrays with the indices of the data represented in the training,
        validation, and test sets.

    :raises ValueError: If the sum of `train_split` and `validation_split` is greater than 1,
        or if `train_split` or `validation_split` are less than 0.
    """
    if dataset_size < 3:
        raise ValueError(f"Expected a dataset with size at least 3, got {dataset_size}")

    if train_split < 0 or validation_split < 0:
        raise ValueError(f"Expected positive splits, got ({train_split=}, {validation_split=})")
    if train_split + validation_split >= 1:
        raise ValueError(
            f"Expected `train_split` and `validation_split` sum between 0 and 1, got {train_split + validation_split}"
        )
    rng = np.random.default_rng(seed)
    selection = rng.permutation(dataset_size)
    train_size = max(1, int(dataset_size * train_split))
    validation_size = max(1, int(dataset_size * validation_split))
    # Ensure that the TEST split has at least an element
    if train_size + validation_size == dataset_size:
        if train_size > 1:
            train_size -= 1
        if validation_size > 1:
            validation_size -= 1
    return {
        Split.TRAIN: selection[:train_size],
        Split.VALIDATION: selection[train_size : train_size + validation_size],
        Split.TEST: selection[train_size + validation_size :],
        Split.ALL: selection,
    }
