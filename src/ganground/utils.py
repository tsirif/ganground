# -*- coding: utf-8 -*-
"""
:mod:`ganground.utils` -- Package-wide useful routines
======================================================

.. module:: utils
   :platform: Unix
   :synopsis: Helper functions useful in possibly all :mod:`ganground`'s modules.

"""
from abc import ABCMeta
from glob import glob
from importlib import import_module
import os

import pkg_resources


class SingletonError(ValueError):
    """Exception to be raised when someone provides arguments to build
    an object from a already-instantiated `SingletonType` class.
    """

    def __init__(self, cls):
        """Pass the same constant message to ValueError underneath."""
        msg = "A singleton instance of '{}' has already been instantiated."
        super().__init__(msg.format(cls.__name__))


class SingletonType(type):
    """Metaclass that implements the singleton pattern for a Python class."""

    def __init__(cls, name, bases, dictionary):
        """Create a class instance variable and initiate it to None object."""
        super(SingletonType, cls).__init__(name, bases, dictionary)
        cls.instance = None

    def __call__(cls, *args, **kwargs):
        """Create an object if does not already exist, otherwise return what there is."""
        if cls.instance is None:
            cls.instance = super(SingletonType, cls).__call__(*args, **kwargs)
        elif args or kwargs:
            raise SingletonError(cls)
        return cls.instance


class AbstractSingletonType(SingletonType, ABCMeta):
    """This will create singleton base classes, that need to be subclassed and implemented."""

    pass


def get_all_subclasses(parent):
    """Get set of subclasses recursively"""
    subclasses = set()
    for subclass in parent.__subclasses__():
        subclasses.add(subclass)
        subclasses |= get_all_subclasses(subclass)

    return subclasses


class Factory(ABCMeta):
    """Instantiate appropriate wrapper for the infrastructure based on input
    argument, ``of_type``.

    Attributes
    ----------
    types : dict of subclasses of ``cls.__base__``
       Updated to contain all possible implementations currently. Check out code.

    """

    def __init__(cls, names, bases, dictionary):
        """Search in directory for attribute names subclassing `bases[0]`"""
        super(Factory, cls).__init__(names, bases, dictionary)

        cls.modules = []
        base = import_module(cls.__base__.__module__)
        try:
            py_files = glob(os.path.abspath(os.path.join(base.__path__[0], '[A-Za-z]*.py')))
            py_mods = map(lambda x: '.' + os.path.split(os.path.splitext(x)[0])[1], py_files)
            for py_mod in py_mods:
                cls.modules.append(import_module(py_mod, package=cls.__base__.__module__))
        except AttributeError:
            # This means that base class and implementations reside in a module
            # itself and not a subpackage.
            pass

        # Get types advertised through entry points!
        for entry_point in pkg_resources.iter_entry_points(cls.__name__):
            entry_point.load()

        cls.find_types()

    def find_types(cls):
        cls.types = list(get_all_subclasses(cls.__base__))
        exclude = set([kls.__name__
                       for kls in cls.types
                       if kls.__name__.startswith('_')])
        exclude.add(cls.__name__)
        if hasattr(cls, 'exclude'):
            exclude |= set(cls.exclude)
        cls.types = {class_.__name__: class_
                     for class_ in cls.types
                     if class_.__name__ not in exclude}

    def __call__(cls, of_type, *args, **kwargs):
        """Create an object, instance of ``cls.__base__``, on first call.

        :param of_type: Name of class, subclass of ``cls.__base__``, wrapper
           of a database framework that will be instantiated on the first call.
        :param args: positional arguments to initialize ``cls.__base__``'s instance (if any)
        :param kwargs: keyword arguments to initialize ``cls.__base__``'s instance (if any)

        .. seealso::
           `Factory.typenames` for values of argument `of_type`.

        .. seealso::
           Attributes of ``cls.__base__`` and ``cls.__base__.__init__`` for
           values of `args` and `kwargs`.

        .. note:: New object is saved as `Factory`'s internal state.

        :return: The object which was created on the first call.
        """
        cls.find_types()
        for name, inherited_class in cls.types.items():
            if name.lower() == of_type.lower():
                return inherited_class.__call__(*args, **kwargs)

        error = "Could not find implementation of {0}, type = '{1}'".format(
            cls.__base__.__name__, of_type)
        error += "\nCurrently, there is an implementation for types:\n"
        error += str(tuple(cls.types.keys()))
        raise NotImplementedError(error)


class SingletonFactory(Factory, AbstractSingletonType):
    """Wrapping `Factory` with `SingletonType`. Keep compatibility with `AbstractSingletonType`."""

    pass
