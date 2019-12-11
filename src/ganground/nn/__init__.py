# -*- coding: utf-8 -*-
r"""
:mod:`ganground.nn` -- Base package for standard nn architectures in PyTorch
============================================================================

.. module:: nn
   :platform: Unix
   :synopsis: Defines common benchmark architectures and helper functions.

"""
import logging
import functools
import math

import torch
from torch import nn

from ganground.state import State
from ganground.nn.layer import *

logger = logging.getLogger(__name__)


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
    if isinstance(m, nn.Embedding):
        nn.init.orthogonal_(m.weight)


class Module(torch.nn.Module):
    """Named `torch.nn.module`."""

    def __init__(self, name, *args, **kwargs):
        super(Module, self).__init__(*args, **kwargs)
        logger.debug("Create module '%s'", name)
        self.name = name

    def finalize_init(self):
        logger.debug("Finalize module '%s'", self.name)
        winit = functools.partial(
            weights_init,
            nonlinearity=getattr(self, 'nonlinearity', None),
            output_nonlinearity=getattr(self, 'output_nonlinearity', None))
        self.apply(winit)
        State().register_module(self)
        self.to(device=State().device)
        self.eval()


def calc_decoder_rounds(n, u, h):
    no = n - u
    rounds = max(no - u, 0) * [None]
    rounds += min(u, no) * ['up', None]
    rounds += max(u - no, 0) * ['up']
    no = n - h
    hrounds = max(no - h, 0) * [1]
    hrounds += min(h, no) * [1, 2]
    hrounds += max(h - no, 0) * [2]
    return list(zip(rounds, hrounds))


class Decoder(Module):
    cond_block_suffix = '_cond'

    def __init__(self, name, dim_in, shape_out,
                 dim_h=64, dim_h_max=1024, n_steps=3, incl_attblock=-1,
                 hierarchical=False, n_targets=0, dim_embed=0,
                 output_nonlinearity=None,
                 f_size=3, wide=True, spectral_norm=False, **layer_args):
        super(Decoder, self).__init__(name)
        Conv2d = SNConv2d if spectral_norm else nn.Conv2d
        Linear = SNLinear if spectral_norm else nn.Linear
        self.output_nonlinearity = output_nonlinearity

        self.hierarchical = hierarchical
        if hierarchical is True:
            dim_in_step = dim_in // (n_steps + 1)
            self.dim_in_net = dim_in - dim_in_step * n_steps
        else:
            dim_in_step = 0
            self.dim_in_net = dim_in

        self.embedding = None
        if n_targets > 0:
            assert(dim_embed > 0)
            self.embedding = nn.Embedding(n_targets, dim_embed)
            dim_in_step += dim_embed

        if not isinstance(incl_attblock, (list, tuple)):
            incl_attblock = (incl_attblock,)
        incl_attblock = tuple(x if x >= 0 else n_steps + x for x in incl_attblock)

        dim_h_out, dim_x_out, dim_y_out = shape_out
        dim_h_ = dim_h

        dim_x = max(dim_x_out // 2**n_steps, 4)
        dim_y = max(dim_y_out // 2**n_steps, 4)
        dim_h = min(dim_h_ * 2**n_steps, dim_h_max)
        dim_out = dim_x * dim_y * dim_h

        up_steps = int(math.log(dim_x_out // dim_x, 2))
        h_steps = int(math.log(dim_h // dim_h_, 2))
        logger.info("Building ResNet Decoder. steps={},up={},h={},attblock={}".format(
            n_steps, up_steps, h_steps, incl_attblock))
        rounds = calc_decoder_rounds(n_steps, up_steps, h_steps)

        self.init = nn.Sequential()
        self.init.add_module(
            'linear_({}/{})'.format(self.dim_in_net, dim_out),
            Linear(self.dim_in_net, dim_out, bias=True))
        self.init.add_module(
            'reshape_{}to{}x{}x{}'.format(dim_out, dim_h, dim_x, dim_y),
            View(-1, dim_h, dim_x, dim_y))

        self.steps = nn.ModuleList()
        dim_out = dim_h
        for i, (resample, div) in enumerate(rounds):
            step = nn.ModuleDict()

            dim_in = dim_out
            if i in incl_attblock:
                attblock = SelfAttention(dim_in, spectral_norm=spectral_norm)
                step['attblock_{}x{}x{}'.format(dim_in, dim_x, dim_y)] = attblock

            dim_out //= div
            resblock = ResBlock(dim_in, dim_out, dim_x, dim_y, f_size,
                                resample=resample, wide=wide,
                                dim_in_cond=dim_in_step,
                                spectral_norm=spectral_norm, **layer_args)
            name = resblock.name + (self.cond_block_suffix if dim_in_step > 0 else '')
            step[name] = resblock
            dim_x *= (2 if resample == 'up' else 1)
            dim_y *= (2 if resample == 'up' else 1)

            self.steps.append(step)

        assert(dim_out == dim_h_)
        assert(dim_x == dim_x_out)
        assert(dim_y == dim_y_out)
        name = 'conv_({}/{})'.format(dim_out, dim_h_out)
        final = finish_layer_2d('pre_' + name,
                                dim_x, dim_y, dim_out, **layer_args)
        pad = (f_size - 1) // 2  # padding that preserves image w and h
        final[name] = Conv2d(dim_out, dim_h_out, f_size, padding=pad, bias=False)
        self.final = nn.Sequential(final)

        self.finalize_init()

    def forward_from_embedding(self, x, y=None, nonlinearity=None, **nonlin_args):
        if nonlinearity is None:
            nonlinearity = self.output_nonlinearity

        # Prepare input to network
        if self.hierarchical:
            z = x[:, :self.dim_in_net]
            chunks = torch.split(x[:, self.dim_in_net:], len(self.steps), 1)
            if self.embedding is not None:
                assert(y is not None)
                y = [torch.cat([y, inp], 1) for inp in chunks]
            else:
                y = chunks
        elif self.embedding is not None:
            z = x
            assert(y is not None)
            y = [y] * len(self.steps)
        else:
            z = x
            y = [None] * len(self.steps)

        # Forward pass
        h = self.init(z)
        for i, step in enumerate(self.steps):
            h = apply_layer_dict(step, h, y=y[i],
                                 conditional_suffix=self.cond_block_suffix)
        h = self.final(h)

        return apply_nonlinearity(h, nonlinearity, **nonlin_args)

    def forward(self, *args, **nonlin):
        x = args[0]
        y = None
        if len(args) >= 2:
            y = args[1]
        if self.embedding is not None:
            assert(y is not None)
            y = self.embedding(y)
        return self.forward_from_embedding(x, y=y, **nonlin)


def calc_encoder_rounds(n, u, h):
    dec_rounds = calc_decoder_rounds(n, u, h)
    rounds = [('down' if up else None, div) for (up, div) in dec_rounds]
    return list(reversed(rounds))


class Encoder(Module):
    def __init__(self, name, shape_in, dim_out,
                 dim_h=64, dim_h_max=1024, n_steps=3, incl_attblock=0,
                 n_targets=0,
                 output_nonlinearity=None,
                 f_size=3, wide=True, spectral_norm=False, **layer_args):
        super(Encoder, self).__init__(name)
        Conv2d = SNConv2d if spectral_norm else nn.Conv2d
        Linear = SNLinear if spectral_norm else nn.Linear
        Embedding = SNEmbedding if spectral_norm else nn.Embedding

        self.output_nonlinearity = output_nonlinearity
        if not isinstance(incl_attblock, (list, tuple)):
            incl_attblock = (incl_attblock,)
        incl_attblock = tuple(x if x >= 0 else n_steps + x for x in incl_attblock)

        dim_h_in, dim_x_in, dim_y_in = shape_in
        dim_out_ = dim_out
        dim_out = dim_h_ = dim_h

        dim_x = max(dim_x_in // 2**n_steps, 4)
        dim_y = max(dim_y_in // 2**n_steps, 4)
        dim_h = min(dim_h_ * 2**n_steps, dim_h_max)

        down_steps = int(math.log(dim_x_in // dim_x, 2))
        h_steps = int(math.log(dim_h // dim_h_, 2))
        logger.info("Building ResNet Encoder. steps={},down={},h={},attblock={}".format(
            n_steps, down_steps, h_steps, incl_attblock))
        rounds = calc_encoder_rounds(n_steps, down_steps, h_steps)

        pad = (f_size - 1) // 2  # padding that preserves image w and h
        self.init = Conv2d(dim_h_in, dim_out, f_size, padding=pad, bias=False)

        self.steps = nn.ModuleList()
        for i, (resample, mul) in enumerate(rounds):
            step = nn.ModuleDict()

            dim_in = dim_out
            dim_out *= mul
            resblock = ResBlock(dim_in, dim_out, dim_x_in, dim_y_in, f_size,
                                resample=resample, wide=wide,
                                spectral_norm=spectral_norm, **layer_args)
            step[resblock.name] = resblock
            dim_x_in //= (2 if resample == 'down' else 1)
            dim_y_in //= (2 if resample == 'down' else 1)

            if i in incl_attblock:
                attblock = SelfAttention(dim_out, spectral_norm=spectral_norm)
                step['attblock_{}x{}x{}'.format(dim_out, dim_x_in, dim_y_in)] = attblock

            self.steps.append(step)

        assert(dim_out == dim_h)
        assert(dim_x_in == dim_x)
        assert(dim_y_in == dim_y)
        self.final_activation = finish_layer_2d('final', dim_x, dim_y, dim_h,
                                                inplace_nonlin=True, **layer_args)
        self.final_linear = Linear(dim_h, dim_out_, bias=True)
        self.embedding = None
        if n_targets > 0:
            assert(dim_out_ == 1)
            self.embedding = Embedding(n_targets, dim_h)

        self.finalize_init()

    def forward(self, *args, nonlinearity=None, **nonlin_args):
        if nonlinearity is None:
            nonlinearity = self.output_nonlinearity
        x = args[0]
        y = None
        if len(args) >= 2:
            y = args[1]

        x = self.init(x)
        for step in self.steps:
            x = apply_layer_dict(step, x)
        x = apply_layer_dict(self.final_activation, x).sum((2, 3))

        out = self.final_linear(x)
        class_out = 0
        if self.embedding is not None:
            assert(y is not None)
            class_out = self.embedding(y).mul(x).sum(1, keepdim=True)

        return apply_nonlinearity(out + class_out, nonlinearity, **nonlin_args)
