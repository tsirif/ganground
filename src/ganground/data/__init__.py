# -*- coding: utf-8 -*-
r"""
:mod:`ganground.data` -- Wrappers for dataset objects
=====================================================

.. module:: data
   :platform: Unix
   :synopsis: Builder functions for a dataset

"""
from abc import (ABCMeta, abstractmethod)
import os

import numpy
import torch

from ganground.utils import Factory
from ganground.random import PRNG
from ganground.state import State

from ganground.data.utils import MultiEpochSampler

__all__ = ['AbstractDataset', 'Dataset']


def _worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    seed = worker_info.seed
    # This value is determined by main process RNG and the worker id.
    password = "{} is the seed set for worker {}.".format(seed, worker_id)
    PRNG.seed(password)


def prepare_splits(splits, N):
    splits = numpy.asarray(splits, dtype=numpy.float64)
    splits = numpy.around((splits / sum(splits)) * N).astype(int)
    s = sum(splits) - N
    if s > 0:
        i = numpy.argmax(splits)
    else:
        i = numpy.argmin(splits)
    splits[i] = splits[i] - s
    return tuple(splits.tolist())


class AbstractDataset(object, metaclass=ABCMeta):

    def __init__(self, root, num_threads=1, download=False, load=True,
                 splits=(1,), **options):
        self.state = State()
        self.root = os.path.join(os.path.expanduser(root))
        self.root = os.path.join(self.root, self.__class__.__name__)
        assert(num_threads >= 0)
        self.num_threads = num_threads
        self.splits = splits
        self.options = options
        self.n_epochs = 0

        if download is True and self.check_exists(self.root) is False:
            self.download(self.root)

        self._data = []
        if load is True:
            self.load()

    def download(self, root):
        pass

    def check_exists(self, root):
        return True

    @abstractmethod
    def prepare(self, root, **options):
        """Return a `torch.utils.data.Dataset` implementation."""
        pass

    def transform(self, batch):
        return batch

    @property
    def data(self):
        if not self._data or \
                any(not isinstance(ds, torch.utils.data.Dataset) for ds in self._data):
            raise ValueError("Call `load` method first.")
        return self._data

    def load(self):
        if not self.check_exists(self.root):
            raise RuntimeError(self.__class__.__name__ + ' not found.' +
                               ' You can use download=True to download it')
        self._data = self.prepare(self.root, **self.options)
        self.n_data = len(self._data)
        self.splits = prepare_splits(self.splits, self.n_data)
        self.n_splits = len(self.splits)
        self._data = torch.utils.data.random_split(self._data, self.splits)

    def build_loader(self, batch_size, sampler, split=0):
        """Return a `torch.utils.data.DataLoader` interface."""
        return torch.utils.data.DataLoader(dataset=self.data[split],
                                           batch_size=batch_size,
                                           sampler=sampler,
                                           #  drop_last=True,
                                           num_workers=self.num_threads,
                                           pin_memory=self.state.is_cuda,
                                           worker_init_fn=_worker_init_fn,
                                           )

    def _fetch(self, loader, stream):
        batch = next(loader)
        if self.state.is_cuda:
            with torch.cuda.stream(stream):
                batch = [x.cuda(non_blocking=True) for x in batch]
                batch = self.transform(batch)
        else:
            batch = self.transform(batch)
        return batch

    def infinite_sampler(self, name: str, batch_size: int, split=0):
        fetch_stream = None
        if self.state.is_cuda:
            fetch_stream = torch.cuda.Stream()
        batches_seen = self.state.samplers(name)  # TODO
        sampler = MultiEpochSampler(self.data[split],
                                    batches_seen,
                                    batch_size)
        loader = iter(self.build_loader(batch_size, sampler, split=split))
        next_batch = self._fetch(loader, fetch_stream)
        while True:
            if self.state.is_cuda:
                torch.cuda.current_stream().wait_stream(fetch_stream)
            current_batch = next_batch
            if self.state.is_cuda:
                for x in current_batch:
                    x.record_stream(torch.cuda.current_stream())
            next_batch = self._fetch(loader, fetch_stream)
            self.state._samplers[name] += 1  # TODO
            yield current_batch


class Dataset(AbstractDataset, metaclass=Factory): pass
