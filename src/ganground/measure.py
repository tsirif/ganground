# -*- coding: utf-8 -*-
r"""
:mod:`ganground.measure` -- Abstractions representing probability measures
==========================================================================

.. module:: measure
   :platform: Unix
   :synopsis: Define software abstractions for probability measures.

"""
import logging

from abc import (ABCMeta, abstractmethod)

from ganground.data import AbstractDataset
from ganground.optim import Trainable
from ganground.nn import Module

logger = logging.getLogger(__name__)
__all__ = ['Measure', 'EmpiricalMeasure', 'InducedMeasure']


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
        super(EmpiricalMeasure, self).__init__(name, None)
        logger.debug("Create empirical measure '%s' from dataset '%s:%d' (bs=%d)",
                     name, dataset.__class__.__name__, split, batch_size)
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
        logger.debug("Create induced measure '%s' from '%s#%s'",
                     name, model.name, [s.name for s in source])
        assert(len(source) > 0)
        self.source = source

    def sample(self, detach_source=False):
        if detach_source is True:
            samples = [s.sample().detach() for s in self.source]
        else:
            samples = [s.sample() for s in self.source]
        if self.model is None:
            if len(samples) == 1:
                return samples[0]
            return samples
        return self.model(*samples)
