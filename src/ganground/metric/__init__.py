# -*- coding: utf-8 -*-
r"""
:mod:`ganground.metric` -- Metrics and divergences between probability measures
===============================================================================

.. module:: metric
   :platform: Unix
   :synopsis: Defines the abstractions to work with metrics on measure spaces.

"""
from ganground.measure import (InducedMeasure, Measure)
from ganground.metric.objective import Objective
from ganground.optim import Trainable


class Metric(Trainable):

    def __init__(self, name: str, P: Measure, Q: Measure,
                 critic=None, **opt_options):
        super(Metric, self).__init__(name, critic, **opt_options)
        self.P = P
        self.Q = Q

################################################################################
#                             Low level interface                              #
################################################################################

    @property
    def critic(self):
        return self.model

    def estimate(self, obj_type: str, **obj_kwargs):
        cP = InducedMeasure(None, self.critic, self.P)
        cQ = InducedMeasure(None, self.critic, self.Q)
        obj = Objective(obj_type)
        return obj.estimate_metric(cP.sample(), cQ.sample(),
                                   **obj_kwargs)

    def loss(self, obj_type: str, **obj_kwargs):
        cP = InducedMeasure(None, self.critic, self.P)
        cQ = InducedMeasure(None, self.critic, self.Q)
        obj = Objective(obj_type)
        return obj.estimate_measure_loss(cP.sample(), cQ.sample(),
                                         **obj_kwargs)

################################################################################
#                             High level interface                             #
################################################################################

    def separate(self, obj_type: str, **obj_kwargs):
        self.P.requires_grad_(False)
        self.Q.requires_grad_(False)
        with self.optimizer_step as opt:
            assert(opt is not None)  # Calling `separate` implies a critic model
            metric = self.estimate(obj_type, **obj_kwargs)
            # TODO wandb log (step=State().info.iter)
            loss = - metric  # Separation means "maximization" of the metric
            loss.backward()

    def minimize(self, obj_type: str, **obj_kwargs):
        self.requires_grad_(False)
        with self.P.optimizer_step as p_opt, self.Q.optimizer_step as q_opt:
            calcpp = p_opt is not None
            calcqq = q_opt is not None
            assert(calcpp or calcqq)
            loss = self.loss(obj_type,
                             calcpp=calcpp, calcqq=calcqq, **obj_kwargs)
            # TODO wandb log (step=State().info.iter)
            loss.backward()
