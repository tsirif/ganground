#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Installation script for ganground."""
import os
import sys

from setuptools import setup

isfile = os.path.isfile
pjoin = os.path.join
repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, pjoin(repo_root, 'src/ganground/_version'))

print(sys.version)

import versioneer  # noqa
import _info as info  # noqa

# Commenting out some requirements because it is better to leave the exact
# installation process for those packages up to the user.
install_requires = [
    'matplotlib',
    'numpy >= 1.10',
    'scipy >= 1.0',
    # 'torch >= 1.1',
    #  'torchvision >= 0.3',
    'nauka',
    'scikit-learn',
    'imageio',
    'imageio-ffmpeg',
    'wandb',
    ]

packages = [
    'ganground',
    'ganground.data',
    'ganground.metric',
    'ganground._version',
    ]

setup_args = dict(
    name='ganground',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description=info.__descr__,
    long_description=open(pjoin(repo_root, 'README.rst')).read(),
    license=info.__license__,
    author=info.__author__,
    author_email=info.__author_email__,
    url=info.__url__,
    packages=packages,
    package_dir={'': 'src'},
    include_package_data=True,
    entry_points={
        'console_scripts': [
            ],
        },
    python_requires='>=3.6',
    install_requires=install_requires,
    tests_require=['pytest>=4.1.1'],
    setup_requires=['setuptools'],
    zip_safe=True
    )

setup_args['keywords'] = [
    'Machine Learning',
    'Generative Adversarial Networks',
    'Generative Models',
    'Optimization',
    'Unsupervised Learning',
    ]

setup_args['platforms'] = ['Linux']

setup_args['classifiers'] = [
    'Intended Audience :: Education',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Operating System :: POSIX',
    'Operating System :: Unix',
    'Programming Language :: Python',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Programming Language :: Python :: 3.7',
]

if __name__ == '__main__':
    setup(**setup_args)
