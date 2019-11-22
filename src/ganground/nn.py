# -*- coding: utf-8 -*-
r"""
:mod:`ganground.nn` -- Helper functions for neural network module of PyTorch
============================================================================

.. module:: nn
   :platform: Unix
   :synopsis: Defines structures helpful for coding with PyTorch

"""
import torch


class Module(torch.nn.Module):
    """Named `torch.nn.module`."""

    def __init__(self, name, *args, **kwargs):
        super(Module, self).__init__(*args, **kwargs)
        self._name = name

    @property
    def name(self):
        return self._name
