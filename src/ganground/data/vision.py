# -*- coding: utf-8 -*-
r"""
:mod:`ganground.data.vision` -- Torchvision datasets
====================================================

.. module:: vision
   :platform: Unix
   :synopsis: Wrap datasets from torchvision.

The goal is to initially support datasets commonly used for
generative model benchmarking (MNIST, CIFAR10, LSUN, ...).

"""
#  import errno
#  import itertools as it
import os

import numpy as np
import torch
from torch.utils.data import TensorDataset
import torchvision
from torchvision.transforms import transforms

from ganground.data import AbstractDataset


class _Torchvision(AbstractDataset):
    def __init__(self, *args, **kwargs):
        super(_Torchvision, self).__init__(*args, **kwargs)
        self.torch_dataset = getattr(torchvision.dataset,
                                     self.__class__.__name__)
        assert(self.torch_dataset is not None)

    def download(self, root):
        dataset = self.torch_dataset(root, download=True)
        del dataset

    def prepare(self, root, **options):
        dataset = self.torch_dataset(root, **options)
        data, targets = zip(*dataset[:])
        data =   


    def transform(self, batch):
        return batch
