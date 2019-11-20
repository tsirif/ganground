# -*- coding: utf-8 -*-
r"""
:mod:`ganground.random` -- Extract and set states of pseudo-PRNGs
================================================================

.. module:: random
   :platform: Unix
   :synopsis: Seed, set and get states of all possible PRNGs

Disclaimer: This module is a fork of functionality in the Nauka project.
Original Code Repo: https://github.com/obilaniu/Nauka
Original Code Copyright: Copyright (c) 2018 Olexa Bilaniuk
Original Code License: MIT License

"""
from abc import (abstractmethod, abstractproperty)
from contextlib import (contextmanager, ExitStack)
import hashlib

from ganground.utils import (AbstractSingletonType, SingletonFactory)


def _str_to_utf8bytes(x, errors="strict"):
    return x.encode("utf-8", errors=errors) if isinstance(x, str) else x


def _pbkdf2(dkLen, password, salt="", rounds=1, hash="sha256"):
    password = _str_to_utf8bytes(password)
    salt = _str_to_utf8bytes(salt)
    return hashlib.pbkdf2_hmac(hash, password, salt, rounds, dkLen)


def _pbkdf2_to_int(nbits, password, salt="",
                   rounds=1, hash="sha256", signed=False):
    nbits = int(nbits)
    dkLen = (nbits + 7) // 8
    ebits = nbits % 8
    ebits = 8 - ebits if ebits else 0
    buf = _pbkdf2(dkLen, password, salt, rounds, hash)
    return int.from_bytes(buf, "little", signed=signed) >> ebits


class AbstractPRNG(object, metaclass=AbstractSingletonType):
    """Interface class for pseudo-random number generators."""

    @abstractmethod
    def get_random_state(self, password, salt="", rounds=1, hash="sha256"):
        pass

    @abstractmethod
    def seed(self, password):
        pass

    @abstractproperty
    def state(self):
        pass

    @state.setter
    def state(self, state_):
        pass

    def __call__(self, password):
        class prng_context(object):
            def __enter__(self_):
                self_.prng_state = self.state
                self.seed(password)
                return self_

            def __exit__(self_, *exc):
                self.state = self_.prng_state
                return False

        return prng_context()


class NumpyPRNG(AbstractPRNG):
    import numpy as np

    def get_random_state(self, password, salt="", rounds=1, hash="sha256"):
        uint32le = self.np.dtype(self.np.uint32).newbyteorder("<")
        buf = _pbkdf2(624 * 4, password, salt, rounds, hash)
        buf = self.np.frombuffer(buf, dtype=uint32le).copy("C")
        return ("MT19937", buf, 624, 0, 0.0)

    def seed(self, password):
        state_ = self.get_random_state(password, salt="numpy.random")
        self.state = state_
        return state_

    @property
    def state(self):
        return self.np.random.get_state()

    @state.setter
    def state(self, state_):
        self.np.random.set_state(state_)


class MathPRNG(AbstractPRNG):
    import random

    def get_random_state(self, password, salt="", rounds=1, hash="sha256"):
        npRandomState = NumpyPRNG().get_random_state(password, salt, rounds, hash)
        twisterState = tuple(npRandomState[1].tolist()) + (624,)
        return (3, twisterState, None)

    def seed(self, password):
        state_ = self.get_random_state(password, salt="random")
        self.state = state_
        return state_

    @property
    def state(self):
        return self.random.getstate()

    @state.setter
    def state(self, state_):
        self.random.setstate(state_)


def _getIntFromPBKDF2(nbits, password, salt="",
                      rounds=1, hash="sha256", signed=False):
    nbits = int(nbits)
    assert nbits % 8 == 0
    dkLen = nbits // 8
    buf = _pbkdf2(dkLen, password, salt, rounds, hash)
    return int.from_bytes(buf, "little", signed=signed)


class TorchPRNG(AbstractPRNG):
    import torch

    def get_random_state(self, password, salt="", rounds=1, hash="sha256"):
        """This produces a seed, not a prng state!!!"""
        return _pbkdf2_to_int(64, password, salt, round, hash, signed=True)

    def seed(self, password):
        seed = self.get_random_state(password, salt="torch.random")
        self.torch.random.manual_seed(seed)
        return self.state

    @property
    def state(self):
        return self.torch.random.get_prng_state()

    @state.setter
    def state(self, state_):
        self.torch.random.set_prng_state(state_)


class TorchCudaPRNG(TorchPRNG):
    CUDA = TorchPRNG.torch.cuda.current_device() >= 0

    def seed(self, password):
        if self.CUDA is False:
            return
        seed = self.get_random_state(password, salt="torch.cuda")
        self.torch.cuda.manual_seed(seed)
        return self.state

    @property
    def state(self):
        if self.CUDA is False:
            return
        return self.torch.cuda.get_prng_state()

    @state.setter
    def state(self, state_):
        if self.CUDA is False:
            return
        self.torch.cuda.set_prng_state(state_)


class PRNG(AbstractPRNG, metaclass=SingletonFactory):
    """Class used to inject an `AbstractPRNG`.

    .. seealso:: `Factory` metaclass and `AbstractPRNG` interface.
    """

    @contextmanager
    @classmethod
    def reseed(cls, password=None):
        if password is None:
            yield
        else:
            with ExitStack() as stack:
                for prng_class in cls.types.values():
                    stack.enter_context(prng_class()(password))
                yield stack

    @classmethod
    def seed(cls, password):
        for prng_class in cls.types.values():
            prng_class().seed(password)

    @property
    @classmethod
    def state(cls):
        return {name: prng_class().state
                for name, prng_class in cls.types.items()}

    @state.setter
    @classmethod
    def state(cls, state_):
        for name, prng_state in state_.items():
            cls.types[name]().state = prng_state
