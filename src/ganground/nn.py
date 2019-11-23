# -*- coding: utf-8 -*-
r"""
:mod:`ganground.nn` -- Helper functions for neural network module of PyTorch
============================================================================

.. module:: nn
   :platform: Unix
   :synopsis: Defines structures helpful for coding with PyTorch

"""
import torch

from ganground.state import State


class Module(torch.nn.Module):
    """Named `torch.nn.module`."""

    def __init__(self, name, *args, **kwargs):
        super(Module, self).__init__(*args, **kwargs)
        self.name = name

        # TODO Perhaps here weight initialization code could lie
        state = State()
        if state.is_cuda:
            self.cuda(device=state.device)
        else:
            self.cpu()
