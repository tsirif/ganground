# -*- coding: utf-8 -*-
r"""
:mod:`ganground.data.utils` -- Utility classes for data loading
===============================================================

.. module:: utils
   :platform: Unix
   :synopsis: Tools for facilitating data loading.

"""
import logging

import numpy
import torch


logger = logging.getLogger(__name__)


class MultiEpochSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly over multiple epochs
    Arguments:
        data_source (Dataset): dataset to sample from
        num_epochs (int): Number of times to loop over the dataset
        start_itr (int): which iteration to begin from
        batch_size (int): how many indices sampler should return

    """

    def __init__(self, data_source,
                 batches_seen=0, batch_size=128):
        self.n_data = len(data_source)
        self.batches_seen = batches_seen
        self.batch_size = batch_size

    def __iter__(self):
        def generate_indices(n, b_seen, b_size):
            s_seen = b_seen * b_size
            e_seen = s_seen // n
            extra = s_seen % n
            for _ in range(e_seen):
                numpy.random.permutation(n)
            rand_indices = numpy.random.permutation(n)
            rand_indices = rand_indices[extra:]
            while True:
                for index in rand_indices:
                    yield index
                rand_indices = numpy.random.permutation(n)

        return iter(generate_indices(self.n_data,
                                     self.batches_seen, self.batch_size))

    def __len__(self):
        raise NotImplementedError()
