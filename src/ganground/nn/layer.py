# -*- coding: utf-8 -*-
r"""
:mod:`ganground.nn.layer` -- Define commonly used neural network blocks
=======================================================================

.. module:: layer
   :platform: Unix
   :synopsis: Defines structures helpful for coding neural networks with PyTorch.

"""
import functools
import logging

from torch import nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import torch

logger = logging.getLogger(__name__)


class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)


def SNConv2d(*args, n_power_iterations=1, eps=1e-12, dim=None, **kwargs):
    return spectral_norm(nn.Conv2d(*args, **kwargs), name='weight', eps=eps,
                         n_power_iterations=n_power_iterations, dim=dim)


def SNLinear(*args, n_power_iterations=1, eps=1e-12, dim=None, **kwargs):
    return spectral_norm(nn.Linear(*args, **kwargs), name='weight', eps=eps,
                         n_power_iterations=n_power_iterations, dim=dim)


def SNEmbedding(*args, n_power_iterations=1, eps=1e-12, dim=None, **kwargs):
    return spectral_norm(nn.Embedding(*args, **kwargs), name='weight', eps=eps,
                         n_power_iterations=n_power_iterations, dim=dim)


class ConditionalBatchNorm(nn.Module):
    def __init__(self, dim_in, num_features, class_bn,
                 spectral_norm=False, **bn_kwargs):
        super(ConditionalBatchNorm, self).__init__()
        assert(issubclass(class_bn, nn.modules.batchnorm._BatchNorm))
        bn_kwargs['affine'] = False
        self.num_features = num_features
        self.batch_norm = class_bn(num_features, **bn_kwargs)
        Linear = SNLinear if spectral_norm else nn.Linear
        self.gain = Linear(dim_in, num_features, bias=False)
        self.bias = Linear(dim_in, num_features, bias=False)

    def forward(self, x, y):
        shape = x.dim() * [1]
        shape[0] = -1
        shape[1] = self.num_features
        gain = self.gain(y).add(1).view(*shape)
        bias = self.bias(y).view(*shape)
        return bias.addcmul(bias, self.batch_norm(x), gain)


def get_nonlinearity(nonlinearity=None, **kwargs):

    def get_from_nn(cls, **kwargs_):
        return cls(**kwargs_)

    if not nonlinearity:
        return

    if callable(nonlinearity):
        if isinstance(nonlinearity, nn.Module):
            cls = type(nonlinearity)
            nonlinearity = get_from_nn(cls, **kwargs)
        else:
            nonlinearity = functools.partial(nonlinearity, **kwargs)

    elif hasattr(nn, nonlinearity):
        cls = getattr(nn, nonlinearity)
        nonlinearity = get_from_nn(cls, **kwargs)

    elif hasattr(nn.functional, nonlinearity):
        nonlinearity = getattr(nn.functional, nonlinearity)
        nonlinearity = functools.partial(nonlinearity, **kwargs)

    else:
        raise ValueError(
            "Could not resolve non linearity: {}".format(repr(nonlinearity)))

    return nonlinearity


def finish_layer_2d(name, dim_x, dim_y, dim_out, models_output=None,
                    dropout=False, inplace_dropout=False,
                    layer_norm=False, batch_norm=False,
                    spectral_norm=False, dim_in_cond=0,
                    nonlinearity=None, inplace_nonlin=False, **nonlin_args):
    if layer_norm and batch_norm:
        logger.warning('Ignoring batch_norm because layer_norm is True')
    assert((dim_in_cond > 0 and batch_norm) or dim_in_cond == 0)

    models = models_output or nn.ModuleDict()

    if dropout:
        models[name + '_do'] = nn.Dropout2d(p=dropout, inplace=inplace_dropout)

    if layer_norm:
        models[name + '_ln'] = nn.LayerNorm((dim_out, dim_x, dim_y))
    elif batch_norm:
        if dim_in_cond > 0:
            models[name + '_cbn'] = ConditionalBatchNorm(dim_in_cond,
                                                         dim_out,
                                                         nn.BatchNorm2d,
                                                         spectral_norm=spectral_norm)
        else:
            models[name + '_bn'] = nn.BatchNorm2d(dim_out)

    if nonlinearity:
        inplace_nonlin = inplace_nonlin or layer_norm or batch_norm
        nonlin_args['inplace'] = inplace_nonlin
        nonlinearity = get_nonlinearity(nonlinearity, **nonlin_args)
        models['{}_{}'.format(name, nonlinearity.__class__.__name__)] = nonlinearity

    return models


def finish_layer_1d(name, dim_out, models_output=None,
                    dropout=False, inplace_dropout=False,
                    layer_norm=False, batch_norm=False,
                    spectral_norm=False, dim_in_cond=0,
                    nonlinearity=None, inplace_nonlin=False, **nonlin_args):
    if layer_norm and batch_norm:
        logger.warning('Ignoring batch_norm because layer_norm is True')
    assert((dim_in_cond > 0 and batch_norm) or dim_in_cond == 0)

    models = models_output or nn.ModuleDict()

    if dropout:
        models[name + '_do'] = nn.Dropout(p=dropout, inplace=inplace_dropout)

    if layer_norm:
        models[name + '_ln'] = nn.LayerNorm(dim_out)
    elif batch_norm:
        if dim_in_cond > 0:
            models[name + '_cbn'] = ConditionalBatchNorm(dim_in_cond,
                                                         dim_out,
                                                         nn.BatchNorm1d,
                                                         spectral_norm=spectral_norm)
        else:
            models[name + '_bn'] = nn.BatchNorm1d(dim_out)

    if nonlinearity:
        inplace_nonlin = inplace_nonlin or layer_norm or batch_norm
        nonlin_args['inplace'] = inplace_nonlin
        nonlinearity = get_nonlinearity(nonlinearity, **nonlin_args)
        models['{}_{}'.format(name, nonlinearity.__class__.__name__)] = nonlinearity

    return models


def apply_nonlinearity(x, nonlinearity, **nonlin_args):
    nonlinearity = get_nonlinearity(nonlinearity, **nonlin_args)
    if nonlinearity:
        # XXX take special care for PReLU later
        x = nonlinearity(x)
    return x


def apply_layer_dict(models, x, y=None, conditional_suffix=None):
    conditional_suffix = conditional_suffix or tuple()
    if not isinstance(conditional_suffix, tuple):
        conditional_suffix = (conditional_suffix,)
    conditional_suffix += ('_cbn',)  # for conditional batch norm

    for name, model in models.items():
        if name.endswith(conditional_suffix):
            assert(y is not None)
            x = model(x, y)
        else:
            x = model(x)

    return x


class SelfAttention(nn.Module):
    def __init__(self, dim_h, spectral_norm=False):
        super(SelfAttention, self).__init__()
        Conv2d = SNConv2d if spectral_norm else nn.Conv2d

        self.theta = Conv2d(dim_h, dim_h // 8, kernel_size=1,
                            padding=0, bias=False)
        self.phi = Conv2d(dim_h, dim_h // 8, kernel_size=1,
                          padding=0, bias=False)
        self.g = Conv2d(dim_h, dim_h // 2, kernel_size=1,
                        padding=0, bias=False)
        self.o = Conv2d(dim_h // 2, dim_h, kernel_size=1,
                        padding=0, bias=False)

        # Learnable gain parameter
        self.gamma = nn.Parameter(torch.tensor(0.), requires_grad=True)

    def forward(self, x):
        _, c, h, w = x.size()
        theta = self.theta(x).view(-1, c // 8, h * w).transpose(1, 2)
        phi = F.max_pool2d(self.phi(x), 2).view(-1, c // 8, (h * w) // 4)
        beta = F.softmax(theta.bmm(phi), -1).transpose(1, 2)
        g = F.max_pool2d(self.g(x), 2).view(-1, c // 2, (h * w) // 4)
        o = self.o(g.bmm(beta).view(-1, c // 2, h, w))
        return self.gamma * o + x


class ResBlock(nn.Module):
    def __init__(self, dim_in, dim_out, dim_x, dim_y, f_size,
                 alpha=1, resample=None, wide=True,
                 dim_in_cond=0, spectral_norm=False, name=None, **layer_args):
        super(ResBlock, self).__init__()
        if resample not in ('up', 'down', None):
            raise ValueError('invalid resample value: {}'.format(resample))
        assert(f_size % 2 == 1)
        if name is None:
            name = 'resblock_({}/{}_{})'.format(dim_in, dim_out, str(resample))
        self.name = name

        self.resample = resample
        self.alpha = alpha
        pad = (f_size - 1) // 2  # padding that preserves image w and h
        Conv2d = SNConv2d if spectral_norm else nn.Conv2d
        dim_c_hidden = max(dim_in, dim_out) if wide is True else min(dim_in, dim_out)
        dim_x_hidden, dim_y_hidden = dim_x, dim_y
        if resample == 'up':
            dim_x_hidden, dim_y_hidden = dim_x * 2, dim_y * 2

        self.skip = None
        if dim_in != dim_out:
            self.skip = Conv2d(dim_in, dim_out, 1, padding=0, bias=False)

        self.pre = finish_layer_2d(name + '_pre', dim_x, dim_y, dim_in,
                                   dim_in_cond=dim_in_cond, spectral_norm=spectral_norm,
                                   **layer_args)

        self.main = nn.ModuleDict()
        conv1 = Conv2d(dim_in, dim_c_hidden, f_size, padding=pad, bias=False)
        self.main[name + '_stage1'] = conv1
        finish_layer_2d(name + '_stage1',
                        dim_x_hidden, dim_y_hidden, dim_c_hidden,
                        models_output=self.main,
                        dim_in_cond=dim_in_cond, spectral_norm=spectral_norm,
                        inplace_nonlin=True, **layer_args)
        conv2 = Conv2d(dim_c_hidden, dim_out, f_size, padding=pad, bias=False)
        self.main[name + '_stage2'] = conv2

    def _shortcut(self, x):
        if self.resample == 'up':
            x = F.interpolate(x, scale_factor=2)
        if self.skip is not None:
            x = self.skip(x)
        if self.resample == 'down':
            x = F.avg_pool2d(x, kernel_size=2)
        return x

    def forward(self, x, y=None):
        h = apply_layer_dict(self.pre, x, y=y)
        if self.resample == 'up':
            h = F.interpolate(h, scale_factor=2)
        h = apply_layer_dict(self.main, h, y=y)
        if self.resample == 'down':
            h = F.avg_pool2d(h, kernel_size=2)
        return self.alpha * h + self._shortcut(x)
