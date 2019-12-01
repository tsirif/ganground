# -*- coding: utf-8 -*-
r"""
:mod:`ganground.optim` -- Everything related to a model's optimization
======================================================================

.. module:: optim
   :platform: Unix
   :synopsis: Contains helper functions and abstractions for the training.

"""
import copy
import logging

from nauka.utils.torch.optim import fromSpec as build_optimizer
import torch

from ganground.nn import Module
from ganground.state import State

logger = logging.getLogger(__name__)
__all__ = ['update_average_model', 'Trainable']


def update_average_model(target_net, source_net, beta):
    param_dict_src = dict(source_net.named_parameters())
    for p_name, p_target in target_net.named_parameters():
        p_source = param_dict_src[p_name]
        assert(p_source is not p_target)
        with torch.no_grad():
            p_target.add_(p_source.sub(p_target).mul(1. - beta))

    buffer_dict_src = dict(source_net.named_buffers())
    for b_name, b_target in target_net.named_buffers():
        b_source = buffer_dict_src[b_name]
        assert(b_source is not b_target)
        with torch.no_grad():
            # Batch Norm statistics are already averaged...
            b_target.copy_(b_source)


class Trainable(object):

    class step_context(object):
        def __init__(self, trainable):
            self.trainable = trainable
            self.optimizer = trainable.optimizer

        def __enter__(self):
            if self.optimizer is None:
                return
            logger.debug("Trainable '%s': Enter step context with model '%s'",
                         self.trainable.name, self.trainable.model.name)
            self.trainable.requires_grad_(True)
            logger.debug("Trainable '%s': optimizer.zero_grad of module '%s'",
                         self.trainable.name, self.trainable._model.name)
            self.optimizer.zero_grad()
            return self.optimizer

        def __exit__(self, *exc):
            if self.optimizer is None:
                return
            logger.debug("Trainable '%s': optimizer.step of module '%s'",
                         self.trainable.name, self.trainable._model.name)
            self.optimizer.step()
            ema = self.trainable.ema
            if ema:
                logger.debug("Trainable '%s': update ema module '%s'",
                             self.trainable.name, self.trainable._avg_model.name)
                update_average_model(self.trainable._avg_model,
                                     self.trainable._model, ema)
            logger.debug("Trainable '%s': Exit step context",
                         self.trainable.name)

    def __init__(self, name: str, model: Module,
                 spec=None, ema=0, **opt_options):
        self.name = name
        self._model = model
        self._avg_model = model
        self.training = False
        self.optimizer = None

        if model is None or spec is None:
            return

        logger.debug("Trainable '%s': Create optimizer for module '%s'",
                     self.name, self._model.name)
        self.optimizer = build_optimizer(self._model.parameters(),
                                         spec, **opt_options)
        State().register_optimizer(self)
        self.ema = None
        if ema:
            assert(1 > ema > 0)
            self.ema = ema
            self._avg_model = copy.deepcopy(self._model)
            self._avg_model.name += "-ema"
            logger.debug("Trainable '%s': Create ema module '%s'",
                         self.name, self._avg_model.name)
            State().register_module(self._avg_model)
            self._avg_model.eval()
            for param in self._avg_model.parameters():
                param.requires_grad_(False)

    def train(self):
        logger.debug("Trainable '%s': Enable training mode",
                     self.name)
        self.training = True
        if self._model is not None:
            self._model.train()
            self._avg_model.train()

    def eval(self):
        logger.debug("Trainable '%s': Enable eval mode",
                     self.name)
        self.training = False
        if self._model is not None:
            self._model.eval()
            self._avg_model.eval()

    def requires_grad_(self, value=True):
        if self._model is not None:
            logger.debug("Trainable '%s': module '%s' requires grad=%s",
                         self.name, self._model.name, value)
            for param in self._model.parameters():
                param.requires_grad_(value)

    @property
    def model(self):
        return self._model if self.training else self._avg_model

    @property
    def optimizer_step(self):
        return self.step_context(self)
