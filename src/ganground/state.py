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
import copy

import torch

from ganground.utils import SingletonType
from nauka.utils import PlainObject
from ganground.tracking import Wandb

logger = logging.getLogger(__name__)


class State(object, metaclass=SingletonType):

    def __init__(self, name, project, args):
        self.args = args

        # Training components management
        self._modules = dict()
        self._optimizers = dict()
        self._samplers = dict()

        # Traning info management
        self.info = PlainObject()
        self.info.name = name  # Experiment must be named
        self.info.iter = 0  # Optimization loops completed
        self.info.inter = 0  # Checkpoint/Evaluation loops completed
        self.info.train_seed = args.seed  # Initial password for training
        # If a held out seed is used for evaluation
        self.info.eval_seed = args.eval_seed

        # Device management and distributed training
        self.info.is_distributed = False
        if 'WORLD_SIZE' in os.environ:
            self.info.is_distributed = int(os.environ['WORLD_SIZE']) > 1
        self.info.world_size = 1
        self.gpu = -1  # CPU is used
        self.local_rank = None  # No distributed
        if self.info.is_distributed:
            self.gpu = self.local_rank = args.local_rank
        elif args.cuda:
            self.gpu = args.cuda[0]
        self.device = self.gpu
        if self.info.is_distributed:
            torch.distributed.init_process_group(backend='nccl',
                                                 init_method='env://')
            self.info.world_size = torch.distributed.get_world_size()
            logger.info("Initialized distributed training with world size: %d",
                        self.info.world_size)

        '''
        # Visualization management
        # deepcopy of args
        # throw away args that we don't want
        # call wandb init
        # call wandb config
        '''
        self.tracking = Wandb(name=name, id=name, project=project)
        self.tracking.set_config(copy.deepcopy(args))

    @property
    def is_master_rank(self):
        return True if self.info.world_size == 1 else self.local_rank == 0

    @property
    def is_distributed(self):
        return self.info.is_distributed

    @property
    def is_cuda(self):
        return self.gpu >= 0

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device_: int):
        if device_ >= 0:
            #  assert(torch.cuda.is_available() and torch.cuda.device_count() > device_)
            self._device = torch.device('cuda', device_)
            logger.info("Using CUDA device: %d", device_)
        else:
            self._device = torch.device('cpu')
            logger.info("Using CPU")

    def register_module(self, module):
        from ganground.nn import Module
        if module.name in self._modules:
            module_ = self._modules[module.name]
            if not isinstance(module_, Module):
                module.load_state_dict(module_)
            else:
                msg = "Module with name '%s' already registered in the State"
                logger.warning(msg, module.name)
        self._modules[module.name] = module
        return module

    def register_optimizer(self, trainable):
        from ganground.optim import Trainable
        if trainable.name in self._optimizers:
            optimizer_ = self._optimizers[trainable.name]
            if not isinstance(optimizer_, Trainable):
                trainable.optimizer.load_state_dict(optimizer_)
            else:
                msg = "Trainable with name '{}' already registered in the State"
                logger.warning(msg, trainable.name)
        self._optimizers[trainable.name] = trainable.optimizer
        return trainable.optimizer

    def samplers(self, name: str):
        if name not in self._samplers:
            self._samplers[name] = 0
        return self._samplers[name]

    def dump(self, path):
        pytorch_path = os.path.join(path, "snapshot.pkl")
        state = PlainObject()
        state.modules = {name: module.state_dict()
                         for name, module in self._modules.items()}
        state.optimizers = {name: opti.state_dict()
                            for name, opti in self._optimizers.items()}
        state.samplers = self._samplers
        state.info = self.info
        state.args = self.args
        torch.save(state, pytorch_path)

        # wandb save
        self.tracking.save()

    def load(self, path):
        pytorch_path = os.path.join(path, "snapshot.pkl")
        state = torch.load(pytorch_path)
        self._modules = state.modules
        self._optimizers = state.optimizers
        self._samplers = state.samplers
        assert(self.info.is_distributed == state.info.is_distributed)
        assert(self.info.world_size == state.info.world_size)
        self.info = state.info

        # wandb restore
        self.tracking.restore()

    def log_setting(self):
        logger.info("Models:\n%s", self._modules)
        logger.info("Optimizers:\n%s", self._optimizers)
        logger.info("Info:\n%s", self.info.__dict__)
        logger.info("Args:\n%s", self.args.__dict__)

    def watch(self):
        self.tracking.watch(tuple(self._modules.values()))