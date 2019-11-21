# -*- coding: utf-8 -*-
r"""
:mod:`ganground.optim` -- Everything related to a model's optimization
======================================================================

.. module:: optim
   :platform: Unix
   :synopsis: Contains helper functions and abstractions for the training.

"""
import copy

from nauka.utils.torch.optim import fromSpec as build_optimizer
import torch

from ganground.nn import Module
from ganground.state import State


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
            self.optimizer = trainable._optimizer

        def __enter__(self):
            if self.optimizer is None:
                return
            self.trainable.train()
            self.trainable.requires_grad_(True)
            self.optimizer.zero_grad()
            return self.optimizer

        def __exit__(self, *exc):
            if self.optimizer is None:
                return
            self.optimizer.step()
            ema = self.trainable.ema
            if ema > 0:
                update_average_model(self.trainable._avg_model,
                                     self.trainable._model, ema)

    def __init__(self, name: str, model: Module,
                 ema=0, spec=None, **opt_options):
        self.name = name
        self._model = model
        self._avg_model = model
        if model is None:
            self.optimizer = None
            return

        self.optimizer = None
        if spec is not None:
            self.optimizer = build_optimizer(model.parameters(),
                                             spec, **opt_options)
            State().register_optimizer(self)
            assert(1 > ema >= 0)
            self.ema = ema
            if ema > 0:
                self._avg_model = copy.deepcopy(model)
                State().register_module(self._avg_model)
                self._avg_model.eval()
        self.eval()

    def train(self):
        self.training = True
        self._model.train()

    def eval(self):
        self.training = False
        self._model.eval()

    def requires_grad_(self, value=True):
        if self._model is not None:
            for param in self._model.parameters():
                param.requires_grad_(value)

    @property
    def model(self):
        return self._model if self.training else self._avg_model

    @property
    def optimizer_step(self):
        return self.step_context(self)
