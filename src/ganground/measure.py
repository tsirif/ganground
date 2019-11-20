# -*- coding: utf-8 -*-
r"""
:mod:`ganground.measure` -- Abstractions representing probability measures
==========================================================================

.. module:: measure
   :platform: Unix
   :synopsis: Define software abstractions for probability measures.

"""
from abc import (ABCMeta, abstractmethod)

from ganground.optim import Trainable


class Measure(object, metaclass=ABCMeta):

    @abstractmethod
    def sample(self):
        pass


class SourceMeasure(Measure):
    # TODO This is to describe a prior source of entropy
    # like a Gaussian distribution (by default it could support all
    # distributions in PyTorch).
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def sample(self):
        pass


class EmpiricalMeasure(SourceMeasure):
    # TODO This is to describe a well-structured source of entropy
    # through a dataset (training or test).
    def __init__(self, batch_size, dataset):
        super(EmpiricalMeasure, self).__init__(batch_size)

    def sample(self):
        pass


class InducedMeasure(Trainable, Measure):
    # TODO this is to describe the induced measure of a `source` one
    # push-through a measurable function `model`
    def __init__(self, model, *source, **opt_options):
        super(InducedMeasure, self).__init__(model, **opt_options)
        self.source = source

    def sample(self):
        return self.model(*[s.sample() for s in self.source])
