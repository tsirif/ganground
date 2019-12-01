# -*- coding: utf-8 -*-
r"""
:mod:`ganground.metric.objective` -- Adversarial loss objective definitions
===========================================================================

.. module:: objective
   :platform: Unix
   :synopsis: Define objective types.

"""
# TODO assertions regarding dimensions of samples in `objectives.py`
from abc import abstractmethod
import math

import torch.nn.functional as F

from ganground.metric.kernel import (mmd2, cross_mean_kernel_wrap, Kernel)
from ganground.utils import (AbstractSingletonType, SingletonFactory)

LOG_2 = math.log(2.)
__all__ = [
    'AbstractObjective', 'Objective',
    'JSD', 'GAN', 'W1', 'WGAN', 'RGAN', 'RAGAN', 'MMD2', 'KL', 'KLDV',
    'Hellinger',
]


class AbstractObjective(object, metaclass=AbstractSingletonType):

    def estimate_measure_loss(self, cp, cq,
                              cp_to_neg=False, nonsat=False,
                              calcpp=True, calcqq=True, **obj_kwargs):
        if not nonsat:
            loss = self.estimate_metric(
                cp, cq, cp_to_neg=cp_to_neg,
                calcpp=calcpp, calcqq=calcqq, **obj_kwargs)
        else:
            loss = - self.estimate_metric(
                cq, cp, cp_to_neg=cp_to_neg,
                calcpp=calcpp, calcqq=calcqq, **obj_kwargs)

        return loss

    @abstractmethod
    def estimate_metric(self, cp, cq, cp_to_neg=False,
                        calcpp=True, calcqq=True, **obj_kwargs):
        pass


class JSD(AbstractObjective):

    def estimate_metric(self, cp, cq, cp_to_neg=False,
                        calcpp=True, calcqq=True, **obj_kwargs):
        if cp_to_neg is False:
            pos = F.logsigmoid(cp).mean() if calcpp else 0
            neg = F.logsigmoid(-cq).mean() if calcqq else 0
        else:
            neg = F.logsigmoid(-cp).mean() if calcpp else 0
            pos = F.logsigmoid(cq).mean() if calcqq else 0
        return 0.5 * (pos + neg) + LOG_2


class GAN(JSD): pass


class W1(AbstractObjective):

    def estimate_metric(self, cp, cq, cp_to_neg=False,
                        calcpp=True, calcqq=True, **obj_kwargs):
        if cp_to_neg is False:
            pos = cp.mean() if calcpp else 0
            neg = - cq.mean() if calcqq else 0
        else:
            neg = - cp.mean() if calcpp else 0
            pos = cq.mean() if calcqq else 0
        return pos + neg


class WGAN(W1): pass


class RGAN(AbstractObjective):

    def estimate_metric(self, cp, cq, cp_to_neg=False,
                        calcpp=True, calcqq=True, **obj_kwargs):
        if cp_to_neg is False:
            metric = F.logsigmoid(cp - cq).mean()
        else:
            metric = F.logsigmoid(cq - cp).mean()
        return metric


class RAGAN(AbstractObjective):

    def estimate_metric(self, cp, cq, cp_to_neg=False,
                        calcpp=True, calcqq=True, **obj_kwargs):
        if cp_to_neg is False:
            metric = F.logsigmoid(cp.mean() - cq).mean() +\
                F.logsigmoid(cp - cq.mean()).mean()
        else:
            metric = F.logsigmoid(cq.mean() - cp).mean() +\
                F.logsigmoid(cq - cp.mean()).mean()
        return metric


class MMD2(AbstractObjective):

    def estimate_metric(self, cp, cq, cp_to_neg=False,
                        calcpp=True, calcqq=True, kernel='gaussian',
                        **obj_kwargs):
        kernel_ = Kernel(kernel)
        kernel_ = cross_mean_kernel_wrap(kernel_,
                                         calcpp=calcpp, calcqq=calcqq,
                                         try_pdist=kernel in Kernel.rbf)
        return mmd2(*kernel_(cp, cq, **obj_kwargs))


class KL(AbstractObjective):
    """
    We use the Fenchel Dual formulation to compute this. We compute other representation which is a lower bound on the Dual of KL.
    """
    def estimate_metric(self, cp, cq, cp_to_neg=False,
                        calcpp=True, calcqq=True, **obj_kwargs):
        if cp_to_neg is False:
            pos = 1 + cp.mean() if calcpp else 0
            neg = - cq.exp().mean() if calcqq else 0
        else:
            neg = -1 - cp.mean() if calcpp else 0
            pos = cq.exp().mean() if calcqq else 0
        return pos + neg


class KLDV(AbstractObjective):
    """
    We use the Fenchel Dual formulation to compute this. We compute Donsker-Varadhan representation which is a lower bound on the Dual of KL.
    """
    def estimate_metric(self, cp, cq, cp_to_neg=False,
                        calcpp=True, calcqq=True, **obj_kwargs):
        if cp_to_neg is False:
            pos = cp.mean() if calcpp else 0
            neg = - cq.exp().mean().log() if calcqq else 0
        else:
            neg = - cp.mean() if calcpp else 0
            pos = cq.exp().mean().log() if calcqq else 0
        return pos + neg


class Hellinger(AbstractObjective):
    """
    We use the Fenchel Dual formulation to compute this.
    """
    def estimate_metric(self, cp, cq, cp_to_neg=False,
                        calcpp=True, calcqq=True, **obj_kwargs):
        if cp_to_neg is False:
            pos = cp.mean() if calcpp else 0
            fenchel_cq = cq/(1 - cq)
            neg = - (fenchel_cq).mean() if calcqq else 0
        else:
            neg = - cp.mean() if calcpp else 0
            fenchel_cq = cq/(1 - cq)
            pos = (fenchel_cq).mean() if calcqq else 0
        return pos + neg


class Objective(AbstractObjective, metaclass=SingletonFactory): pass
