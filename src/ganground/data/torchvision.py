# -*- coding: utf-8 -*-
r"""
:mod:`ganground.data.torchvision` -- Wrappers for torchvision datasets
======================================================================

.. module:: torchvision
   :platform: Unix
   :synopsis: Integrates image datasets from torchvision

"""
import torchvision
from torchvision import transforms as tform
from torchvision.transforms import functional as F

from ganground.data import AbstractDataset


# TODO Verify that this works!
class _Torchvision(AbstractDataset):

    def _get_data_class(self):
        dname = self.__class__.__name__
        dclass = getattr(torchvision.datasets, dname)
        assert(dclass is not None)
        return dclass

    def download(self, root):
        dclass = self._get_data_class()
        dclass(root, download=True)

    def prepare(self, root, train=True,
                transform=None, target_transform=None, **options):
        dclass = self._get_data_class()

        if transform is None:
            transform = list()
            transform.append(tform.RandomHorizontalFlip())
            transform.append(tform.ToTensor())
            transform = tform.Compose(transform)
        return dclass(root, train=train, transform=transform,
                      target_transform=target_transform, **options)

    def transform(self, batch):
        # Scale [0, +1] to [-1, +1]
        batch = F.normalize(mean=(0.5, 0.5, 0.5),
                            std=(0.5, 0.5, 0.5), inplace=True)
        return batch


class CIFAR10(_Torchvision):
    pass
