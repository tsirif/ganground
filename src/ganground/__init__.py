# -*- coding: utf-8 -*-
from ._version import get_versions
VERSIONS = get_versions()
del get_versions

__descr__ = "**ganground** is intended to be a lightweight toolbox for Generative Adversarial Network training in **PyTorch**."
__version__ = VERSIONS['version']
__license__ = 'MIT'
__author__ = 'Christos Tsirigotis'
__author_email__ = 'tsirif@gmail.com'
__authors_list__ = {
    #  'anonymous_nickname': ('anonymous', 'anonymous@anony.mous'),  # TODO
}
__url__ = None
