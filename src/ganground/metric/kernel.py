# -*- coding: utf-8 -*-
r"""
:mod:`ganground.metric.kernel` -- Kernel function definitions
=============================================================

.. module:: kernel
   :platform: Unix
   :synopsis: Helper functions for Maximum Mean Discrepancy, plus kernel
      function definitions.

"""
from abc import abstractmethod
import logging

import torch
import torch.nn.functional as F

from ganground.utils import (AbstractSingletonType, SingletonFactory)

logger = logging.getLogger(__name__)
__all__ = [
    'mmd2', 'cross_mean_kernel_wrap',
    '_pairwise_dot', '_pairwise_dist', '_pairwise_pow_dist',
    'AbstractKernel', 'Kernel', 'Gaussian', 'Laplacian', 'Exp',
    'Linear', 'Poly', 'Tanh', 'IMQ'
]


def mmd2(PPk, QQk, PQk):
    """Calculate squared Maximum Mean Discrepancy distance.

    Args:
        PPk: None, scalar torch tensor containing the mean PPk
             or the full pairwise distance matrix
        QQk: scalar torch tensor containing the mean QQk
             or the full pairwise distance matrix
        PQk: scalar torch tensor containing the mean PQk
             or the full pairwise distance matrix

    """
    assert(PQk is not None)
    # Allow `PPk` to be None, if we want to compute mmd2 for the generator
    if PPk is None:
        PPk_ = 0
    elif len(PPk.shape) == 2:
        m = PPk.size(0)
        PPk_ = (PPk.sum() - PPk.trace()) / (m**2 - m) if m != 1 else 0
    elif len(PPk.shape) == 1:
        PPk_ = PPk.mean()
    elif len(PPk.shape) == 0:
        PPk_ = PPk
    else:
        raise ValueError("Not supported `PPk`.")

    if QQk is None:
        QQk_ = 0
    elif len(QQk.shape) == 2:
        n = QQk.size(0)
        QQk_ = (QQk.sum() - QQk.trace()) / (n**2 - n) if n != 1 else 0
    elif len(QQk.shape) == 1:
        QQk_ = QQk.mean()
    elif len(QQk.shape) == 0:
        QQk_ = QQk
    else:
        raise ValueError("Not supported `QQk`.")

    if PQk.size():
        PQk_ = PQk.mean()
    else:
        PQk_ = PQk

    return PPk_ + QQk_ - 2 * PQk_


################################################################################
#                           Helper Kernel Functions                            #
################################################################################


def _find_tensor_device(tensor):
    try:
        dev = tensor.get_device()
    except RuntimeError as e:
        if 'CPU backend' in str(e):
            return 'cpu'
        else:
            raise

    if dev == -1:
        return 'cpu'

    return 'cuda:' + str(dev)


def cross_mean_kernel_wrap(kernel_fn, calcpp=True, calcqq=True,
                           try_pdist=False):

    def kernel(cp, cq, **kernel_args):
        m = cp.size(0)
        n = cq.size(0)
        cp_cq_len = m + n
        if try_pdist:
            #  logger.debug("Using pytorch.pdist with concatenation to calc cross means.")
            # It will use F.pdist to calculate cpp, cpq, and cqq
            cp_cq = torch.cat((cp, cq), dim=0).contiguous()
            res_type = cp_cq.dtype
            res_device = _find_tensor_device(cp_cq)
            res = kernel_fn(cp_cq, cp_cq, **kernel_args).to(torch.float64)
            inds = torch.triu_indices(cp_cq_len, cp_cq_len, offset=1,
                                      dtype=torch.float64, device=res_device)
            inds_r = inds[0, :]
            inds_c = inds[1, :]

            # Split to get Ecxx, Ecxy, and Ecyy
            if calcpp:
                mask = inds_c.sub(m).mul_(-1).clamp_(0, 1)
                n = mask.sum()
                cpp = mask.mul_(res).sum().div_(n).to(res_type)
            else:
                cpp = None

            if calcqq:
                mask = inds_r.sub(m - 1).clamp_(0, 1)
                n = mask.sum()
                cqq = mask.mul_(res).sum().div_(n).to(res_type)
            else:
                cqq = None

            mask = inds_r.sub(m).mul_(-1).clamp_(0, 1)
            mask = inds_c.sub(m - 1).clamp_(0, 1).mul_(mask)
            n = mask.sum()
            cpq = mask.mul_(res).sum().div_(n).to(res_type)

        else:
            #  logger.debug("Invoking three times the kernel to calc cross means.")
            if calcpp:
                cpp = kernel_fn(cp, cp, **kernel_args)
            else:
                cpp = None
            if calcqq:
                cqq = kernel_fn(cq, cq, **kernel_args)
            else:
                cqq = None
            cpq = kernel_fn(cp, cq, **kernel_args)

        return cpp, cqq, cpq

    return kernel


def _pairwise_dist(cx, cy, p=2, _pow_flag=False):
    """Compute pairwise distances between two Tensors of size m x `shape`
    and n x `shape`.

    This should be done as efficiently as possible. Discussions:
    https://github.com/pytorch/pytorch/issues/9406

    """
    def _are_equal(cx, cy):
        if cx is cy:
            return True
        return torch.equal(cx, cy)

    res = None
    m = cx.size(0)
    n = cy.size(0)
    imsize = cx.view(m, -1).size(-1)
    cx_eq_cy = _are_equal(cx, cy)

    if cx_eq_cy:
        #  logger.debug("Calc pairwise distance with pytorch.pdist.")
        # Calculate only triangular, fast, cheaper, stable. Looks like this:
        # torch.cat([torch.full((n - i - 1,), i, dtype=torch.int64) for i in range(n)])
        res = F.pdist(cx.view(m, -1), p=p)
    elif p == 2 and m * n * imsize * (torch.finfo(cx.dtype).bits // 8) > 4 * 1024**2:
        #  logger.debug("Calc pairwise distance with quadratic expansion.")
        # If more than 4MB needed to repr a full matrix
        # Faster and cheaper, but less stable (quadratic expansion)
        # Still slower than the first choice
        cx_ = cx.view(m, -1)
        cy_ = cy.view(n, -1)
        cx_norm = cx_.pow(2).sum(dim=-1, keepdim=True)
        cy_norm = cy_.pow(2).sum(dim=-1, keepdim=True).transpose(-2, -1)
        res = cx_norm + cy_norm - 2 * cx_.matmul(cy_.transpose(-2, -1))

        if cx_eq_cy:
            # Ensure zero diagonal
            diag_inds = torch.arange(m)
            res[diag_inds, diag_inds] = 0

        # Zero out negative values
        res.clamp_min_(0)
        if _pow_flag:
            _pow_flag[0] = True
        else:
            res = res.sqrt()
    else:
        #  logger.debug("Calc pairwise distance with naive broadcasting.")
        # More expensive - Θ(n^2 d), but numerically more stable
        cx_ = cx.view(m, 1, -1)
        cy_ = cy.view(1, n, -1)
        # XXX does not support broadcasting yet #15918 and #15901
        #  res = F.pairwise_distance(cx_, cy_, p=p, eps=1e-8)
        res = torch.norm(cx_ - cy_, p=p, dim=-1)

    return res


def _pairwise_pow_dist(cx, cy, p=2):
    pow_flag = [False]
    res = _pairwise_dist(cx, cy, p=p, _pow_flag=pow_flag)

    if pow_flag[0]:
        return res

    return res.pow(p)


def _pairwise_dot(cx, cy):
    # TODO Find a more memory efficient way to do this
    # Can pytorch cosine similarity be used perhaps??
    m = cx.size(0)
    n = cy.size(0)
    cx_ = cx.view(m, 1, -1)
    cy_ = cy.view(1, n, -1)
    return cx_.mul(cy_).sum(dim=-1)


################################################################################
#                         Kernel Function Definitions                          #
################################################################################


class AbstractKernel(object, metaclass=AbstractSingletonType):
    @abstractmethod
    def __call__(self, cx, cy, **kernel_kwargs):
        pass


class Gaussian(AbstractKernel):
    def __call__(self, cx, cy, sigma=1):
        XY = _pairwise_pow_dist(cx, cy)
        return XY.div_(- 2 * sigma**2).exp_()


class Laplacian(AbstractKernel):
    def __call__(self, cx, cy, p=2, sigma=1):
        XY = _pairwise_dist(cx, cy, p=p)
        return XY.div_(-sigma).exp_()


class Exp(AbstractKernel):
    def __call__(self, cx, cy, sigma=1):
        XY = _pairwise_dot(cx, cy)
        return XY.div_(sigma).exp_()


class Linear(AbstractKernel):
    def __call__(self, cx, cy, c=0):
        return _pairwise_dot(cx - c, cy - c)


class Poly(AbstractKernel):
    def __call__(self, cx, cy, a=1, c=1, d=2):
        XY = _pairwise_dot(cx, cy)
        return XY.mul_(a).add_(c).pow_(d)


class Tanh(AbstractKernel):
    def __call__(self, cx, cy, a=1, c=0):
        XY = _pairwise_dot(cx, cy)
        return XY.mul_(a).add_(c).tanh_()


class IMQ(AbstractKernel):
    def __call__(self, cx, cy, c=1, d=2, sigma=1):
        XY = _pairwise_pow_dist(cx, cy)
        return XY.div_(sigma**2).add_(c).pow_(-d / 2)


class Kernel(AbstractKernel, metaclass=SingletonFactory):
    rbf = ('gaussian', 'exp', 'laplacian', 'imq')
