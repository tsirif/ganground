# -*- coding: utf-8 -*-
r"""
:mod:`ganground.metric` -- Metrics and divergences between probability measures
===============================================================================

.. module:: metric
   :platform: Unix
   :synopsis: Defines the abstractions to work with metrics on measure spaces.

"""
from abc import (abstractmethod, ABCMeta)
from argparse import (Action, Namespace)
import logging
import re

from ganground.measure import (InducedMeasure, Measure)
from ganground.optim import Trainable
from ganground.utils import SingletonFactory

from ganground.metric.kernel import (AbstractKernel, Kernel)

logger = logging.getLogger(__name__)
__all__ = [
    'Metric', 'AbstractKernel', 'Kernel', 'AbstractObjective', 'Objective',
    'ObjectiveBuilder', 'ObjectiveAction',
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

    def estimate(self, obj_spec, **obj_kwargs):
        # Create InducesMeasure here because critic module may change from
        # training to eval time
        cP = InducedMeasure('critic#P', self.critic, self.P)
        cQ = InducedMeasure('critic#Q', self.critic, self.Q)
        obj = ObjectiveBuilder(**vars(obj_spec))
        logger.debug("Estimating (training=%s) metric '%s' with '%s'",
                     self.training, self.name, obj_spec)
        return obj.estimate_metric(cP.sample(detach_source=True),
                                   cQ.sample(detach_source=True),
                                   **obj_kwargs)

    def loss(self, obj_spec, **obj_kwargs):
        cP = InducedMeasure('critic#P', self.critic, self.P)
        cQ = InducedMeasure('critic#Q', self.critic, self.Q)
        obj = ObjectiveBuilder(**vars(obj_spec))
        logger.debug("Loss (training=%s) from metric '%s' with '%s'",
                     self.training, self.name, obj_spec)
        return obj.estimate_measure_loss(cP.sample(), cQ.sample(),
                                         **obj_kwargs)

################################################################################
#                             High level interface                             #
################################################################################

    def separate(self, obj_spec, reg_spec=None, **obj_kwargs):
        logger.debug("Separating (%s) P=%s and Q=%s",
                     self.name, self.P.name, self.Q.name)
        if reg_spec:
            reg_spec = vars(reg_spec)
        else:
            reg_spec = dict()
        reg = ObjectiveBuilder(**reg_spec)
        self.P.requires_grad_(False)
        self.Q.requires_grad_(False)
        with self.optimizer_step as opt:
            with self.P.hold_samples(), self.Q.hold_samples():
                assert(opt is not None)  # Calling `separate` implies a critic model
                metric = self.estimate(obj_spec, **obj_kwargs)
                regularize = reg.estimate_metric(self.P.sample(),
                                                 self.Q.sample(),
                                                 self.critic)
                loss = - metric - regularize  # Separation means "maximization"
                logger.debug("loss=%s", loss)
                loss.backward()
        return metric.detach()

    def minimize(self, obj_spec, reg_spec=None, **obj_kwargs):
        logger.debug("Minimizing metric %s(P=%s, Q=%s)",
                     self.name, self.P.name, self.Q.name)
        if reg_spec:
            reg_spec = vars(reg_spec)
        else:
            reg_spec = dict()
        reg = ObjectiveBuilder(**reg_spec)
        self.requires_grad_(False)
        with self.P.optimizer_step as p_opt, self.Q.optimizer_step as q_opt:
            with self.P.hold_samples(), self.Q.hold_samples():
                calcpp = p_opt is not None
                calcqq = q_opt is not None
                assert(calcpp or calcqq)
                loss = self.loss(obj_spec,
                                 calcpp=calcpp, calcqq=calcqq, **obj_kwargs)
                regularize = reg.estimate_measure_loss(self.P.sample(),
                                                       self.Q.sample(),
                                                       self.critic)
                loss = loss + regularize
                logger.debug("calcpp=%s,calcqq=%s,loss=%s", calcpp, calcqq, loss)
                loss.backward()
        return loss.detach()


class AbstractObjective(object, metaclass=ABCMeta):

    @abstractmethod
    def estimate_measure_loss(self, cp, cq, **kwargs):
        pass

    @abstractmethod
    def estimate_metric(self, cp, cq, **kwargs):
        pass


class Objective(AbstractObjective, metaclass=SingletonFactory):
    exclude = ('ObjectiveBuilder',)


class ObjectiveBuilder(AbstractObjective):

    def __init__(self, *args, **kwargs):
        self.coeffs = dict()
        self.objs = dict()
        for obj in args:
            obj_type = str(obj)
            self.objs[obj_type] = Objective(obj_type)
            self.coeffs[obj_type] = 1
        for obj, coeff in kwargs.items():
            obj_type = str(obj)
            self.objs[obj_type] = Objective(obj_type)
            self.coeffs[obj_type] = float(coeff)

    def estimate_measure_loss(self, cp, cq, *args, **kwargs):
        losses = [self.coeffs[obj_type] * obj.estimate_measure_loss(cp, cq, *args, **kwargs)
                  for obj_type, obj in self.objs.items()]
        return sum(losses)

    def estimate_metric(self, cp, cq, *args, **kwargs):
        losses = [self.coeffs[obj_type] * obj.estimate_metric(cp, cq, *args, **kwargs)
                  for obj_type, obj in self.objs.items()]
        return sum(losses)


class ObjectiveAction(Action):
    prog = re.compile(r"(?:([-+])?(\d*\.\d+|\d+)?(\w+))")

    def __init__(self, **kwargs):
        default = kwargs.get('default')
        kwargs['default'] = self.parse_string(default) if default else None
        kwargs['nargs'] = None
        kwargs['type'] = str
        kwargs['metavar'] = '((+/-)(coeff==1)objective)*'
        super(ObjectiveAction, self).__init__(**kwargs)

    def __call__(self, parser, ns, values, option_string=None):
        setattr(ns, self.dest, self.parse_string(values))

    @classmethod
    def parse_string(cls, string):
        coeffs = dict()
        for sign, c, obj in cls.prog.findall(string):
            sign = sign or '+'
            c = c or '1'
            coeffs[obj] = float(sign + c)
        return Namespace(**coeffs)
