=========
ganground
=========

**ganground** is intended to be a lightweight toolbox
for Generative Adversarial Network training in **PyTorch**.

Right now it contains 2 modules: `kernels` and `objectives`


Kernels module
==============

This module facilitates the
estimation of *Maximum Mean Discrepancy* (MMD) metric.
It has implemented various kernels of interest.

For pairwise distance calculation, effort has been made to use `torch.pdist`,
wherever is is possible, else a choice between a quadratic approximation
and a naive implementation is made depending on the memory size of the
arguments.

For pairwise dot product calculation, there is only the naive approach.

Importantly for large memory arguments, `cross_mean_kernel_wrap`
can enhance an existing `*_kernel` function with O(1) optimizations.
Calling it returns a tuple of 3 elements `(Kxx, Kyy, Kxy)`.
`Kxx` contains the results of the pairwise computation and kernel application
among vectors in the `cx` batch.
`Kyy` contains the results of the pairwise computation and kernel application
among vectors in the `cy` batch.
`Kxy` contains the results of the pairwise computation and kernel application
among vectors between the `cx` and the `cy` batches.

1. Through kwarg `calc_cxx`, `Kxx` would only be calculated if it is `True`.
2. If kwarg `try_pdist` is `True`, `Kxx`, `Kyy` and `Kxy` will be calculated
   using a single kernel function application, by (1)
   concatenating `cx` and `cy`, (2) applying the kernel, (3) calculating the
   correct means with appropriate masks on the single result.

Objectives module
=================

This module contains functions `estimate_gener_loss` and `estimate_discr_loss`
for estimating the loss for a generator and a discriminator respectively.
Also, it contains two global variables which list all supported options
for objective function types: `GENERATOR_OBJECTIVES` and
`DISCRIMINATOR_OBJECTIVES`.

Testing
=======

Please, test module before using by invoking `pytest -vvv --log-level=DEBUG`
within the ganground directory. Notice `pytest>=4.1.1` dependency.
Test on the available CUDA device only by appending `--cuda-only` to the
shell command above. Test on both CPU and CUDA by appending `--cuda`.
