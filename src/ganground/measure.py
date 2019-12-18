# -*- coding: utf-8 -*-
r"""
:mod:`ganground.measure` -- Abstractions representing probability measures
==========================================================================

.. module:: measure
   :platform: Unix
   :synopsis: Define software abstractions for probability measures.

"""
from abc import (ABCMeta, abstractmethod)
import logging

import torch

from ganground.data import AbstractDataset
from ganground.optim import Trainable
from ganground.nn import Module

logger = logging.getLogger(__name__)
__all__ = ['Measure', 'EmpiricalMeasure', 'InducedMeasure']


class Measure(object, metaclass=ABCMeta):

    def __init__(self, *args, **kwargs):
        super(Measure, self).__init__(*args, **kwargs)
        self.holding = False
        self.detaching = False
        self._held_samples = None

    @abstractmethod
    def sample_(self, **kwargs):
        pass

    def sample(self, **kwargs):
        if self.holding is False or self._held_samples is None:
            sam = self.sample_(**kwargs)
            sam = sam.detach() if self.detaching else sam

        if self.holding is False:
            return sam

        if self._held_samples is None:
            self._held_samples = sam

        return self._held_samples

    def hold_samples(self, value=True, detach=False):
        class hold_context(object):
            def __enter__(self_):
                self.holding = value
                self.detaching = detach
                return self_

            def __exit__(self_, *exc):
                self.holding = False
                self.detaching = False
                self._held_samples = None

        return hold_context()


#  class SourceMeasure(Measure):
#      """Describe a prior source of entropy, like a Gaussian distribution."""

#      def __init__(self, batch_size: int):
#          self.batch_size = batch_size

#      def sample(self):
#          pass


class EmpiricalMeasure(Measure, Trainable):
    """Describe a structured `source` of entropy; a dataset."""

    def __init__(self, name: str, dataset: AbstractDataset, batch_size: int,
                 split=0, resume=True):
        super(EmpiricalMeasure, self).__init__(name, None)
        logger.debug("Create empirical measure '%s' from dataset '%s:%d' (bs=%d)",
                     name, dataset.__class__.__name__, split, batch_size)
        self.dataset = dataset
        self.batch_size = batch_size
        self.split = split
        self.sampler = dataset.infinite_sampler(name, batch_size,
                                                split=split, resume=resume)

    def sample_(self, **kwargs):
        batch = next(self.sampler)
        if len(batch) == 1:
            return batch[0]
        return batch


# TODO what happens if multiple tensors in a batch? need for a marginal measure?


class InducedMeasure(Measure, Trainable):
    """Describe the induced measure of a `source` through a measurable `model`."""

    def __init__(self, name: str, model: Module, *source, **opt_options):
        super(InducedMeasure, self).__init__(name, model, **opt_options)
        #  logger.debug("Create induced measure '%s' from '%s#%s'",
        #               name, model.name, [s.name for s in source])
        assert(len(source) > 0)
        self.source = source

    def sample_(self, detach_source=False, **kwargs):
        samples_ = [s.sample() for s in self.source]
        samples = list()
        for x in samples_:
            if torch.is_tensor(x):
                samples.append(x)
            else:
                samples.extend(x)
        if detach_source is True:
            samples = [s.detach() for s in samples]
        if self.model is None:
            if len(samples) == 1:
                return samples[0]
            return samples
        return self.model(*samples)
