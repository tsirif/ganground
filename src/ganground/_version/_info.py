# -*- coding: utf-8 -*-
try:
    from _setup import get_versions
except ModuleNotFoundError:
    from ._setup import get_versions
VERSIONS = get_versions()
del get_versions

__descr__ = "**ganground** is intended to be a lightweight toolbox for Generative Adversarial Network training in **PyTorch**."
__version__ = VERSIONS['version']
__license__ = 'MIT'
__author__ = 'Christos Tsirigotis'
__author_email__ = 'tsirif@gmail.com'
__url__ = None

__all__ = [
    '__descr__', '__version__', '__license__', '__author__',
    '__author_email__', '__url__',
]
