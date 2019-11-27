# -*- coding: utf-8 -*-
r"""
:mod:`ganground.nn` -- Helper functions for neural network module of PyTorch
============================================================================

.. module:: nn
   :platform: Unix
   :synopsis: Defines structures helpful for coding with PyTorch

"""
import torch
from torch import nn

from ganground.state import State


def weights_init(m, nonlinearity=None, output_nonlinearity=None):
    """Helper function to initialize parameters in a network.

    Use `network.apply(parameters_init)`.

        Args:
            m: `torch.nn.Module` to initialize, part of a larger network
            nonlinearity: type applied throughout network except its output
            output_nonlinearity: type applied at the network's output

    """
    nonlinearity = nonlinearity or 'ReLU'
    nonlinearity = 'leaky_relu' if nonlinearity == 'LeakyReLU' else nonlinearity
    nonlinearity = 'relu' if nonlinearity == 'ReLU' else nonlinearity
    if 'final' in getattr(m, 'name', ''):  # 'final' denotes last layer
        nonlinearity = output_nonlinearity or 'linear'
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        if nonlinearity in ('relu', 'leaky_relu'):
            nn.init.kaiming_uniform_(m.weight, a=0.02, mode='fan_in',
                                     nonlinearity=nonlinearity)
        else:
            gain = nn.init.calculate_gain(nonlinearity)
            nn.init.xavier_uniform_(m.weight, gain=gain)


class Module(torch.nn.Module):
    """Named `torch.nn.module`."""

    def __init__(self, name, *args, **kwargs):
        super(Module, self).__init__(*args, **kwargs)
        self.name = name

    def finalize_init(self):
        # FIXME self.apply(weights_init)
        self.to(device=State().device)
