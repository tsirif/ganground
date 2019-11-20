# -*- coding: utf-8 -*-
import pytest
import torch


def pytest_addoption(parser):
    parser.addoption(
        "--cuda", action="store_true", default=False,
        help="Run tests both on cpu and a cuda device, if test requests `device`.")
    parser.addoption(
        "--cuda-only", action="store_true", default=False,
        help="Run tests on a cuda device **only**, if test requests `device`.")


def pytest_generate_tests(metafunc):
    torch.device('cpu')

    if 'device' in metafunc.fixturenames:
        def generate_device():
            device_names = []
            if metafunc.config.getoption('cuda_only'):
                device_names += ['cuda']
            else:
                device_names += ['cpu']
                if metafunc.config.getoption('cuda'):
                    device_names += ['cuda']
            for devname in device_names:
                yield pytest.param(torch.device(devname), id=devname)

        devgen = generate_device()
        metafunc.parametrize('device', devgen)
