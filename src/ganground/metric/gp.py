# -*- coding: utf-8 -*-
r"""
:mod:`ganground.metric.gp` -- Regularization with Gradient Penalty
==================================================================

.. module:: objective
   :platform: Unix
   :synopsis: Complementary objective that penalize a critic's gradient wrt
      to its input.

"""
import torch
from torch import autograd

from ganground.metric import AbstractObjective
from ganground.state import State
from ganground.utils import AbstractSingletonType


def _get_x_y(inps):
    if isinstance(inps, (list, tuple)):
        x, y = inps
    else:
        x, y = inps, None
    return x, y


def get_gradient_wrt(critic, inps):
    x, y = _get_x_y(inps)
    x = x.detach().requires_grad_()
    with torch.set_grad_enabled(True):
        outs = critic(x, y)
    gradient = autograd.grad(
        outputs=outs,
        inputs=x,
        grad_outputs=torch.ones_like(outs),
        create_graph=True, retain_graph=True, only_inputs=True)[0]
    return gradient, outs


class _GradientPenalty(AbstractObjective, metaclass=AbstractSingletonType):

    def estimate_measure_loss(self, p, q, critic):
        return 0


class GPP(_GradientPenalty):

    def estimate_metric(self, p, q, critic):
        gradient, _ = get_gradient_wrt(critic, p)
        penalty = gradient.view(gradient.size(0), -1).pow(2).sum(-1).mean()
        return - penalty


class GPQ(_GradientPenalty):
    gpp = GPP()

    def estimate_metric(self, p, q, critic):
        return self.gpp.estimate_metric(q, p, critic)


class Roth(_GradientPenalty):
    # Appropriate for JSD with cp_to_neg=False

    def estimate_metric(self, p, q, critic):
        grad_p, cp = get_gradient_wrt(critic, p)
        grad_p_c = torch.sigmoid(cp).sub(1).pow(2)
        gp = grad_p.view(grad_p.size(0), -1).pow(2).sum(-1).mul(grad_p_c).mean()

        grad_q, cq = get_gradient_wrt(critic, q)
        grad_q_c = torch.sigmoid(cq).pow(2)
        gq = grad_q.view(grad_q.size(0), -1).pow(2).sum(-1).mul(grad_q_c).mean()

        return - (gp + gq)


class OGP(_GradientPenalty):
    # This assumes that p and q have been generated using the same conditional
    # information

    def estimate_metric(self, p, q, critic):
        px, py = _get_x_y(p)
        qx, qy = _get_x_y(q)
        bs = px.size(0)
        size = px.size()
        epsilon = torch.rand(bs, 1, device=State().device)
        inter = px.view(bs, -1) * epsilon + qx.view(bs, -1) * (1 - epsilon)
        # Interpolate labels as well ??
        gradient, _ = get_gradient_wrt(critic, (inter.view(*size), py))
        penalty = gradient.view(gradient.size(0), -1).norm(p=2, dim=-1).sub(1).pow(2).mean()
        return - penalty
