# -*- coding: utf-8 -*-
r"""
:mod:`ganground.logging` -- Logging Utility Management
======================================================

.. module:: logging
   :platform: Unix
   :synopsis: Utility functions for setting up logging functionality

"""
from argparse import Action
import logging as py_logging


def getLogger(suffix: str = ''):
    # TODO Perhaps have a timestamp here as well...
    return py_logging.getLogger('ganground.runtime' + ('.' + suffix if suffix else ''))


class LogAction(Action):
    help_string = \
        "Set logging verbosity:: '-v': INFO, '-vv': DEBUG, more 'v's make root logging more verbose."

    def __init__(self, **kwargs):
        kwargs['dest'] = 'verbosity'
        kwargs.setdefault('default', 0)
        kwargs['nargs'] = 0
        kwargs['type'] = int
        kwargs['help'] = self.help_string
        self.package_logger = py_logging.getLogger('ganground')
        self.rlevel = py_logging.WARNING
        self.plevel = py_logging.WARNING
        #  self.package_logger.propagate = False
        super(LogAction, self).__init__(**kwargs)

    def __call__(self, parser, ns, values, option_string=None):
        count = getattr(ns, self.dest, None)
        if count is None:
            count = 0
        count += 1
        self.reset(verbosity=count)
        setattr(ns, self.dest, count)

    def reset(self, verbosity=0):
        if verbosity == 1:
            self.plevel = py_logging.INFO
        elif verbosity >= 2:
            self.plevel = py_logging.DEBUG
            if verbosity == 3:
                self.rlevel = py_logging.INFO
            else:
                self.rlevel = py_logging.DEBUG

        py_logging.basicConfig(level=self.rlevel)
        self.package_logger.setLevel(self.plevel)
