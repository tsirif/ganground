# -*- coding: utf-8 -*-
r"""
:mod:`ganground.state` -- Defines the state for a possible experiment
=====================================================================

.. module:: state
   :platform: Unix
   :synopsis: Define a singleton state for an experiment

"""
import logging
import os

import torch

from ganground.utils import SingletonType
from ganground.optim import Trainable
from ganground.nn import Module


logger = logging.getLogger(__name__)


class State(object, metaclass=SingletonType):

    def __init__(self, name, args):
        self.args = args

        self._modules = dict()
        self._optimizers = dict()

        self.info = object()
        self.info.name = name  # Experiment must be named
        self.info.epochs = 0  # Passes through the training set
        self.info.iters = 0  # Optimization loops completed
        self.info.inters = 0  # Checkpoint/Evaluation loops completed
        self.info.train_seed = args.seed  # Initial password for training
        # If a held out seed is used for evaluation
        self.info.eval_seed = args.eval_seed
        self.info.is_distributed = False
        if 'WORLD_SIZE' in os.environ:
            self.info.is_distributed = int(os.environ['WORLD_SIZE']) > 1
        self.info.world_size = 1
        self.info.gpu = -1  # CPU is used
        self.local_rank = None  # No distributed
        if self.info.is_distributed:
            self.local_rank = self.info.gpu = args.local_rank
        elif args.cuda:
            self.info.gpu = args.cuda[0]
        self.device = self.info.gpu
        if self.info.is_distributed:
            torch.distributed.init_process_group(backend='nccl',
                                                 init_method='env://')
            self.info.world_size = torch.distributed.get_world_size()

    @property
    def is_master_rank(self):
        return True if self.info.world_size == 1 else self.local_rank == 0

    @property
    def is_distributed(self):
        return self.info.is_distributed

    @property
    def is_cuda(self):
        return self.info.gpu >= 0

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device_: int):
        if device_ >= 0:
            #  assert(torch.cuda.is_available() and torch.cuda.device_count() > device_)
            self._device = torch.device('cuda', device_)
            logger.info('Using CUDA device: %d', device_)
        else:
            self._device = torch.device('cpu')
            logger.info('Using CPU.')

    def register_module(self, module: Module):
        if module.name not in self._modules:
            self._modules[module.name] = module
        else:
            module_ = self._modules[module.name]
            if not isinstance(module_, Module):
                module.load_state_dict(module_)
            else:
                msg = "Module with name '{}' already registered in the State"
                raise AssertionError(msg.format(module.name))

    def register_optimizer(self, trainable: Trainable):
        if trainable.name not in self._optimizers:
            self._optimizers[trainable.name] = trainable.optimizer
        else:
            optimizer_ = self._optimizers[trainable.name]
            if not isinstance(optimizer_, Trainable):
                trainable.optimizer.load_state_dict(optimizer_)
            else:
                msg = "Trainable with name '{}' already registered in the State"
                raise AssertionError(msg.format(trainable.name))

    def dump(self, path):
        state = object()
        state.modules = {name: module.state_dict()
                         for name, module in self._modules.items()}
        state.optimizers = {name: opti.state_dict()
                            for name, opti in self._optimizers.items()}
        state.info = self.info
        state.args = self.args
        torch.save(state, os.path.join(path, "snapshot.pkl"))

    def load(self, path):
        state = torch.load(os.path.join(path, "snapshot.pkl"))
        self._modules = state.modules
        self._optimizers = state.optimizers
        self.info = state.info
        # TODO Verify the hypothesis is that the args are exactly the same
