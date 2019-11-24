# -*- coding: utf-8 -*-
"""
:mod:`ganground.data.toysets` -- 2D synthetic toy datasets
=============================================================

.. module:: toysets
   :platform: Unix
   :synopsis: Synthetic datasets for interpretable experimentation

Collection of 2D datasets used primarily for benchmarking of
generative algorithms and interpretable experiments in the input space.

"""
from abc import abstractmethod

import numpy
import sklearn.datasets
import torch
from torch.utils.data import TensorDataset

from ganground.data import AbstractDataset


class _Synthetic(AbstractDataset):
    def prepare(self, root, size=1, **options):
        assert(size > 0)
        dataset = self.build(size, **options)
        dataset = dataset.astype('float32')
        numpy.random.shuffle(dataset)
        return TensorDataset(torch.from_numpy(dataset))

    @abstractmethod
    def build(self, size, **options):
        pass


class G25_2D(_Synthetic):
    def build(self, size):
        dataset = []
        for _ in range(size // 25):
            for x in range(-2, 3):
                for y in range(-2, 3):
                    point = numpy.random.randn(2) * 0.05
                    point[0] += 2 * x
                    point[1] += 2 * y
                    dataset.append(point)
        dataset = numpy.array(dataset)
        dataset /= 2.828  # stdev
        return dataset


class G8_2D(_Synthetic):
    def build(self, size):
        dataset = []
        scale = 2.
        centers = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1. / numpy.sqrt(2), 1. / numpy.sqrt(2)),
            (1. / numpy.sqrt(2), -1. / numpy.sqrt(2)),
            (-1. / numpy.sqrt(2), 1. / numpy.sqrt(2)),
            (-1. / numpy.sqrt(2), -1. / numpy.sqrt(2))
        ]
        centers = [(scale * x, scale * y) for x, y in centers]
        for _ in range(size // len(centers)):
            for i in range(len(centers)):
                point = numpy.random.randn(2) * .02
                point[0] += centers[i][0]
                point[1] += centers[i][1]
                dataset.append(point)
        dataset = numpy.array(dataset)
        dataset /= 1.414  # stdev
        return dataset


class G1_2D(_Synthetic):
    def build(self, size):
        return numpy.random.randn(size, 2).astype('float32')


class Swissroll(_Synthetic):
    def build(self, size):
        dataset, _ = sklearn.datasets.make_swiss_roll(
            n_samples=size,
            noise=0.25)
        dataset = dataset[:, [0, 2]]
        dataset /= 7.5  # stdev plus a little
        return dataset


class C1_2D(_Synthetic):
    def build(self, size):
        phi = numpy.random.uniform(-numpy.pi, numpy.pi, size)
        dataset = numpy.concatenate([numpy.cos(phi), numpy.sin(phi)]).reshape(2, size).T
        return dataset


class C3_2D(_Synthetic):
    def build(self, size):
        dataset = []
        scale = 0.5
        centers = [
            (1, 0),
            (-0.5, numpy.sqrt(3) / 2),
            (-0.5, - numpy.sqrt(3) / 2),
        ]
        set_size = size // 3
        for center in centers:
            phi = numpy.random.uniform(-numpy.pi, numpy.pi, set_size)
            d = numpy.concatenate([numpy.cos(phi), numpy.sin(phi)]).reshape(2, set_size).T
            d *= scale
            d += numpy.array([center])
            dataset.append(d)
        return numpy.concatenate(dataset)


class Concentric(_Synthetic):
    def build(self, size):
        radius = 2
        scales = numpy.array([1, 1 - 1/64, 1 - 1/32, 1 - 1/16, 1 - 1/8, 1 - 1/4, 1 - 1/2])
        scales = radius * scales
        phis = numpy.random.uniform(-numpy.pi, numpy.pi, size)
        radii = numpy.random.choice(scales, size=size)
        return numpy.concatenate([radii * numpy.cos(phis), radii * numpy.sin(phis)]).reshape(2, size).T


class L1_2D(_Synthetic):
    def build(self, size, slope=3, y0=1):
        x = numpy.random.uniform(-1, 1, size)
        return numpy.concatenate([x, slope * x + y0]).reshape(2, size).T


class L4_2D(_Synthetic):
    def build(self, size):
        dataset = []
        sets = [(-1.5, -0.5, 1), (-1.5, -0.5, -1), (0.5, 1.5, 1), (0.5, 1.5, -1)]
        set_size = size // 4
        for set in sets:
            x = numpy.random.uniform(set[0], set[1], set_size)
            d = numpy.concatenate([x, numpy.ones_like(x) * set[2]]).reshape(2, set_size).T
            dataset.append(d)
        return numpy.concatenate(dataset)
