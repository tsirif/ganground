# -*- coding: utf-8 -*-
r"""
:mod:`ganground.measure` -- Abstractions representing probability measures
==========================================================================

.. module:: measure
   :platform: Unix
   :synopsis: Define software abstractions for probability measures.

"""
from abc import (ABCMeta, abstractmethod)

from ganground.data import AbstractDataset
from ganground.optim import Trainable
from ganground.nn import Module


class Measure(object, metaclass=ABCMeta):

    @abstractmethod
    def sample(self):
        pass


#  class SourceMeasure(Measure):
#      """Describe a prior source of entropy, like a Gaussian distribution."""

#      def __init__(self, batch_size: int):
#          self.batch_size = batch_size

#      def sample(self):
#          pass


class EmpiricalMeasure(Trainable, Measure):
    """Describe a structured `source` of entropy; a dataset."""

    def __init__(self, name: str, dataset: AbstractDataset, batch_size: int, split=0):
        super(EmpiricalMeasure, self).__init__(None, None)
        self.dataset = dataset
        self.batch_size = batch_size
        self.split = split
        self.sampler = dataset.infinite_sampler(name, batch_size, split=split)

    def sample(self):
        batch = next(self.sampler)
        if len(batch) == 1:
            return batch[0]
        return batch


# TODO what happens if multiple tensors in a batch? need for a marginal measure?


class InducedMeasure(Trainable, Measure):
    """Describe the induced measure of a `source` through a measurable `model`."""

    def __init__(self, name: str, model: Module, *source, **opt_options):
        super(InducedMeasure, self).__init__(name, model, **opt_options)
        assert(len(source) > 0)
        self.source = source

    def sample(self):
        if self.model is None:
            sample = [s.sample() for s in self.source]
            if len(sample) == 1:
                return sample[0]
            return sample
        return self.model(*[s.sample() for s in self.source])
