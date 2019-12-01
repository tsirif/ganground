# -*- coding: utf-8 -*-
r"""
:mod:`ganground.tracking` -- Interface for tracking quantities during a run
===========================================================================

.. module:: tracking
   :platform: Unix
   :synopsis: It implements an interface to WandB to log experiment data

"""
from argparse import (Action, Namespace)
import logging

import wandb

from ganground.utils import SingletonType
from ganground.state import State

logger = logging.getLogger(__name__)
__all__ = ['WandbAction', 'Wandb']


class WandbAction(Action):

    def __init__(self, **kwargs):
        kwargs['dest'] = 'tracking'
        default = kwargs.get('default', '')
        kwargs['default'] = Namespace(**self.parse_string(default))
        kwargs['metavar'] = '(ENTITY(:KEY))|(None)'
        kwargs['nargs'] = None
        kwargs['type'] = str
        kwargs.setdefault('help', "WandB entity of the project and user's API key.")
        super(WandbAction, self).__init__(**kwargs)

    def __call__(self, parser, ns, values, option_string):
        parsed = self.parse_string(values)
        setattr(ns, self.dest, Namespace(**parsed) if parsed is not None else None)

    @classmethod
    def parse_string(cls, string):
        if not string:
            return dict(key=None, entity=None)
        if string in ('None', 'none', 'No', 'no', 'False', 'false', 'N', 'n'):
            return None
        split = string.split(':', 1)
        entity = split[0]
        key = split[1] if len(split) == 2 else None
        return dict(key=key, entity=entity)


class Wandb(object, metaclass=SingletonType):

    def __init__(self, key, entity, **kwargs):
        logger.debug("key: %s | entity: %s\nkwargs:%s", key, entity, kwargs)
        self.api_key = key
        wandb.login(key=self.api_key)

        self.job_type = None  # The type of job running
        # nauka's subcommand run
        self.dir = kwargs.get('dir')  # An absolute path to a directory where metadata will be stored
        # abspath to logdir
        self.config = kwargs.get('config') or dict()  # The hyper parameters to store with the run
        self.project = kwargs.get('project')  # The project to push metrics to
        self.entity = entity  # The entity to push metrics to
        self.tags = None  # A list of tags to apply to the run
        # Gotten from nauka's presets/VCS
        self.group = None  # A unique string shared by all runs in a given group
        self.resume = kwargs.get('id')  # True, the run auto resumes
        self.force = True  # Force authentication with wandb
        self.name = kwargs.get('name')  # A display name which does not have to be unique
        # Non-unique name of run (human-readable)
        self.notes = None,  # A multiline string associated with the run
        self.id = kwargs.get('id'),  # A globally unique (per project) identifier for the run
        self.anonymous = None  # Controls whether anonymous logging is allowed.

    def init(self):
        wandb.init(job_type=self.job_type,
                   dir=self.dir,
                   config=self.config,
                   project=self.project,
                   entity=self.entity,
                   tags=self.tags,
                   group=self.group,
                   resume=self.resume,
                   force=self.force,
                   name=self.name,
                   #  notes=self.notes,
                   #  id=self.id,
                   #  anonymous=self.anonymous,
                   )

    def log(self, kwargs, step=None):
        wandb.log(kwargs, step=step)

    #  def save(self, path):
    #      assert(not self._has_init)
    #      wandb.save(os.path.join(path, 'snapshot.pkl'))

    #  def restore(self, path):
    #      assert(not self._has_init)
    #      wandb.restore('snapshot.pkl', root=path)  # TODO

    def watch(self, model_tuple):
        wandb.watch(model_tuple)
