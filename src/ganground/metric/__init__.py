# -*- coding: utf-8 -*-
r"""
:mod:`ganground.metric` -- Metrics and divergences between probability measures
===============================================================================

.. module:: metric
   :platform: Unix
   :synopsis: Defines the abstractions to work with metrics on measure spaces.

"""
import logging

from ganground.measure import (InducedMeasure, Measure)
from ganground.optim import Trainable

from ganground.metric.kernel import (AbstractKernel, Kernel)
from ganground.metric.objective import (AbstractObjective, Objective)

logger = logging.getLogger(__name__)
__all__ = [
    'Metric', 'AbstractKernel', 'Kernel', 'AbstractObjective', 'Objective',
]


class Metric(Trainable):

    def __init__(self, name: str, P: Measure, Q: Measure,
                 critic=None, **opt_options):
        super(Metric, self).__init__(name, critic, **opt_options)
        self.P = P
        self.Q = Q
        logger.debug("Create metric %s(P=%s,Q=%s) with critic '%s'",
                     self.name, P.name, Q.name, critic.name if critic else None)

################################################################################
#                             Low level interface                              #
################################################################################

    @property
    def critic(self):
        return self.model

    def estimate(self, obj_type: str, **obj_kwargs):
        # Create InducesMeasure here because critic module may change from
        # training to eval time
        cP = InducedMeasure('critic#P', self.critic, self.P)
        cQ = InducedMeasure('critic#Q', self.critic, self.Q)
        obj = Objective(obj_type)
        logger.debug("Estimating (training=%s) metric '%s' with '%s'",
                     self.training, self.name, obj)
        return obj.estimate_metric(cP.sample(detach_source=True),
                                   cQ.sample(detach_source=True),
                                   **obj_kwargs)

    def loss(self, obj_type: str, **obj_kwargs):
        cP = InducedMeasure('critic#P', self.critic, self.P)
        cQ = InducedMeasure('critic#Q', self.critic, self.Q)
        obj = Objective(obj_type)
        logger.debug("Loss (training=%s) from metric '%s' with '%s'",
                     self.training, self.name, obj)
        return obj.estimate_measure_loss(cP.sample(), cQ.sample(),
                                         **obj_kwargs)

################################################################################
#                             High level interface                             #
################################################################################

    def separate(self, obj_type: str, **obj_kwargs):
        logger.debug("Separating (%s) P=%s and Q=%s",
                     self.name, self.P.name, self.Q.name)
        self.P.requires_grad_(False)
        self.Q.requires_grad_(False)
        with self.optimizer_step as opt:
            assert(opt is not None)  # Calling `separate` implies a critic model
            metric = self.estimate(obj_type, **obj_kwargs)
            loss = - metric  # Separation means "maximization" of the metric
            logger.debug("loss=%s", loss)
            loss.backward()
        return metric.detach()

    def minimize(self, obj_type: str, **obj_kwargs):
        logger.debug("Minimizing metric %s(P=%s, Q=%s)",
                     self.name, self.P.name, self.Q.name)
        self.requires_grad_(False)
        with self.P.optimizer_step as p_opt, self.Q.optimizer_step as q_opt:
            calcpp = p_opt is not None
            calcqq = q_opt is not None
            assert(calcpp or calcqq)
            loss = self.loss(obj_type,
                             calcpp=calcpp, calcqq=calcqq, **obj_kwargs)
            logger.debug("calcpp=%s,calcqq=%s,loss=%s", calcpp, calcqq, loss)
            loss.backward()
        return loss.detach()
