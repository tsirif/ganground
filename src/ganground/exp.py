# -*- coding: utf-8 -*-
r"""
:mod:`ganground.exp` -- Base class for all experiments
======================================================

.. module:: exp
   :platform: Unix
   :synopsis: Create a skeleton of experiment process and training orchestration.

"""
from abc import (abstractmethod, abstractproperty, ABCMeta)
import argparse
import binascii
import copy
import json
import logging
import os
import sys

import nauka


from ganground.logging import getLogger
from ganground.random import (PRNG, _pbkdf2)
from ganground.state import State
from ganground.tracking import Wandb

logger = logging.getLogger(__name__)
__all__ = ['Experiment']


def nested_ns_to_dict(ns):
    ns = copy.copy(vars(ns))
    nsd = {k: nested_ns_to_dict(v) for k, v in ns.items()
           if isinstance(v, argparse.Namespace)}
    ns.update(nsd)
    return ns


def flatten_ns_to_dict(ns):
    nsd = dict()
    stack = list(vars(ns).items())
    while stack:
        k, v = stack.pop()
        if isinstance(v, argparse.Namespace):
            stack.extend([(k + '.' + k_, v_) for k_, v_ in vars(v).items()])
        else:
            nsd[k] = v
    return nsd


class ExperimentInterface(object, metaclass=ABCMeta):

    @abstractproperty
    def name(self):
        """Return experiment's name.

        INSTRUCTIONS
        ------------
        Create a **unique name** for this experiment using its
        configuration, described in `self.args`, as well as special
        designated names, which can be found in `self.args.name`
        (perhaps as prefix or suffix).

        """
        pass

    @abstractproperty
    def is_done(self):
        """Evaluate experiment's terminating condition and return it.

        INSTRUCTIONS
        ------------
        Use running variables regarding the experiment contained in
        `self.info` and/or `self`, as well as constraints from `self.args`
        to implement a terminating condition for the experiment.

        """
        pass

    @abstractmethod
    def define(self):
        """Setup the objects needed to describe the training algorithm.

        INSTRUCTIONS
        ------------
        This method must use the arguments `self.args` to create and
        combine the components of the learning process:
           1. datasets,
           2. modules/networks,
           3. measures,
           4. metrics

        .. note:: All components are required to have a unique `name` string
           identifier!

        .. info:: Those will be used in the `self.train` method to describe
           the training algorithm.

        """
        pass

    @abstractmethod
    def execute(self):
        """Implement the training algorithm given the components defined in
        `self.define`.

        INSTRUCTIONS
        ------------
        This method should implement a training interval (possibly multiple
        dataset epochs) as a loop up to some iterations (increment `self.iter`
        until some period argument in `self.args`). Then, the loop should be
        followed by evaluation and visualization code.

        .. info:: This routine will be executed multiple times until the
           experiment's termination criterion, described by `self.is_done`,
           is fulfilled.

        """
        pass


class Experiment(nauka.exp.Experiment, ExperimentInterface):
    @classmethod
    def project_workdir(cls, args):
        if args.workdir:
            return args.workdir
        else:
            return os.path.join(args.basedir, cls.__name__)

    @classmethod
    def project_logdir(cls, args):
        pworkdir = cls.project_workdir(args)
        return os.path.join(pworkdir, "logs")

    def __init__(self, args):
        """Initialize Experiment.

        :param args: experiment's arguments, taken from a subcommand.

        """
        args = type(args)(**args.__dict__)
        args.__dict__.pop("__argp__", None)
        args.__dict__.pop("__argv__", None)
        args.__dict__.pop("__cls__", None)
        self.args = args
        # This is the time where state is created!
        self.state = State(self.name, self.__class__.__name__,
                           args, self.hyperparams)
        workdir = os.path.join(self.project_workdir(args), self.name)
        super(Experiment, self).__init__(workdir)
        self.mkdirp(self.logdir)
        logger.info("Initializing experiment with name: {}".format(self.name))
        self.logging = getLogger(self.__class__.__name__)

        # TODO move to State object
        self.tracking = None
        if args.tracking is not None:
            self.tracking = Wandb(args.tracking.key, args.tracking.entity,
                                  project=self.__class__.__name__,
                                  name=self.name,
                                  config=flatten_ns_to_dict(self.state.hyperparams),
                                  id=self.hash(),
                                  dir=self.workdir,
                                  )

    @property
    def hyperparams(self):
        args = copy.deepcopy(self.args)
        del args.verbosity
        del args.workdir
        del args.basedir
        del args.datadir
        del args.tmpdir
        del args.name
        del args.cuda
        del args.tracking
        del args.fastdebug
        return args

    @property
    def info(self):
        return self.state.info

    @property
    def device(self):
        return self.state.device

    @property
    def iter(self):
        return self.state.info.iter

    @iter.setter
    def iter(self, iter_):
        self.state.info.iter = iter_
        # TODO Substitute with progress bar
        sys.stdout.write("{}/{}\r".format(self.iter, self.args.train_iters))
        sys.stdout.flush()
        self.logging.debug("=========  Iter: %d/%d  ===========",
                           self.iter, self.args.train_iters)


    @property
    def inter(self):
        return self.state.info.inter

    @inter.setter
    def inter(self, inter_):
        self.state.info.inter = inter_

    @property
    def datadir(self):
        """Returns the root directory where datasets reside."""
        return self.args.datadir

    @property
    def workdir(self):
        return self.workDir

    @property
    def logdir(self):
        """Return the directory where experiment log will reside."""
        return os.path.join(self.workdir, "logs")

    @property
    def exitcode(self):
        return 0 if self.is_done else 1

    def hash(self, length=32):
        hparams = nested_ns_to_dict(self.hyperparams)
        hparams['mingle'] = self.args.name
        s = _pbkdf2(length,
                    json.dumps(hparams, sort_keys=True),
                    salt="hyperparameters", rounds=100001)
        return binascii.hexlify(s).decode('utf-8', errors='strict')

    # TODO summarize function
    def log(self, **kwargs):
        self.logging.debug("Iter=%d: %s", self.iter, kwargs)
        if self.tracking:
            self.tracking.log(kwargs, step=self.iter)

    def dump(self, path):
        """Dump state to the directory `path`

        When invoked by the snapshot machinery, `path/` may be assumed to
        already exist. The state must be saved under that directory, but
        the contents of that directory and any hierarchy underneath it are
        completely freeform, except that the subdirectory `path/.experiment`
        must not be touched.

        When invoked by the snapshot machinery, the path's basename as given
        by os.path.basename(path) will be the number this snapshot will be
        be assigned, and it is equal to self.nextSnapshotNum.

        """
        self.state.dump(path)
        return self

    def load(self, path):
        """Load state from given `path`.

        Restore the experiment to a state as close as possible to the one
        the experiment was in at the moment of the dump() that generated the
        checkpoint with the given `path`.

        """
        self.state.load(path)
        return self

    def fromScratch(self):
        """Start a fresh experiment, from scratch."""
        password = "Seed: {} Init".format(self.args.seed)
        PRNG.seed(password)
        if self.tracking:
            self.tracking.init()
        self.define()
        if self.tracking:
            self.tracking.watch(self.state.modules)
        return self

    def fromSnapshot(self, path):
        """Start an experiment from a snapshot.

        Most likely, this method will invoke self.load(path) at an opportune
        time in its implementation.

        Returns `self`.
        """
        return self.load(path).fromScratch()

    def interval(self):
        """An interval is defined as the computation- and time-span between two
        snapshots.

        Hard Requirements
        -----------------
           - By definition, one may not invoke snapshot() within an interval.
           - Corollary: The work done by an interval is either fully recorded
             or not recorded at all.

        For reproducibility purposes, all PRNGs are reseeded at the beginning
        of every interval.
        """
        password = "Seed: {} Interval: {:d}".format(self.args.seed,
                                                    self.inter)
        PRNG.seed(password)
        self.execute()
        self.inter += 1
        return self

    def run(self):
        """Run by intervals until experiment completion."""
        try:
            self.state.log_setting()
            while not self.is_done:
                self.interval().snapshot().purge()
        except KeyboardInterrupt:
            pass
        return self
