#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os
import sys

import matplotlib as mpl
#  mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import nauka
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.autograd as autograd

from ganground.data import Dataset
from ganground.nn import Module
from ganground.exp import Experiment
from ganground.measure import (Measure, EmpiricalMeasure, InducedMeasure)
from ganground.metric import Metric
from ganground.metric.kernel import (mmd2, AbstractKernel,
                                     cross_mean_kernel_wrap,
                                     _pairwise_dist)
from ganground.random import PRNG


logger = logging.getLogger(__name__)


# Plot settings
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 18
FIG_X_SIZE_IN = 12
FIG_Y_SIZE_IN = 12
DPI = 96
FPS = 10
CMAP_DIVERGING = mpl.cm.seismic
CMAP_SEQUENTIAL = mpl.cm.plasma
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


LAPLC_KERNEL_SIGMAS = (0.01, 0.025, 0.1, 0.25, 1)


class LaplacianMix(AbstractKernel):
    def __call__(self, cx, cy, sigmas=LAPLC_KERNEL_SIGMAS):
        if not isinstance(sigmas, (tuple, list, set)):
            sigmas = [sigmas]
        XY = _pairwise_dist(cx, cy, p=1)
        total = 0
        for sigma in sigmas:
            total = total + XY.div(-sigma).exp().div(len(sigmas))
        return total


laplacian_mix_kernel = cross_mean_kernel_wrap(LaplacianMix(), try_pdist=True)


class Noise(Measure):
    def __init__(self, name: str, *args, **kwargs):
        self.name = name
        self.args = args
        self.kwargs = kwargs

    def sample(self):
        return torch.randn(*self.args, **self.kwargs)


class Generator(Module):

    def __init__(self, name, args):
        super(Generator, self).__init__(name)
        self.nonlinearity = 'ReLU'

        DIM = args.model_width
        main = nn.Sequential(
            nn.Linear(2, DIM),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, 2),
        )
        self.main = main

    def forward(self, noise):
        return self.main(noise)


class Discriminator(Module):

    def __init__(self, name, args):
        super(Discriminator, self).__init__(name)
        if args.sn:
            def Linear(*args_, **kwargs_):
                return nn.utils.spectral_norm(nn.Linear(*args_, **kwargs_))
        else:
            Linear = nn.Linear

        self.nonlinearity = args.critic_nonlin
        if self.nonlinearity == 'ReLU':
            nonlin = nn.ReLU
        elif self.nonlinearity == 'LeakyReLU':
            nonlin = nn.LeakyReLU
        else:
            raise RuntimeError(
                'Not supported nonlinearity for critic: {}'.format(self.nonlinearity))

        DIM = args.model_width
        main = nn.Sequential(
            Linear(2, DIM),
            nonlin(inplace=True),
            Linear(DIM, DIM),
            nonlin(inplace=True),
            Linear(DIM, DIM),
            nonlin(inplace=True),
        )
        self.main = main
        self.output = Linear(DIM, 1, bias=(not args.no_critic_last_bias))

    def forward(self, inputs):
        return self.output(self.main(inputs))


class TwoDExperiment(Experiment):
    @property
    def g_d_iters(self):
        diters = 0
        if self.args.discr_iters != 0:
            diters = max(self.args.discr_iters, 1)
        giters = max(-self.args.discr_iters, 1)
        return giters, diters

    @property
    def name(self):
        default_name = "{:s}".format(self.args.obj_type)
        giters, diters = self.g_d_iters
        if giters != diters or giters != 1:
            default_name += "({:d}-{:d})".format(diters, giters)
        default_name += "-{:s}".format(self.args.dataset)

        #  if self.args.g_opt == 'SGD':
        #      default_name += '-g' + self.args.g_opt + \
        #          "({:.4f},{:.2f})".format(self.args.g_lr, self.args.g_mom)
        #  elif self.args.g_opt == 'Adam':
        #      default_name += '-g' + self.args.g_opt + \
        #          "({:.4f},{:.2f},{:.2f})".format(self.args.g_lr, self.args.g_mom,
        #                                          self.args.g_beta2)

        #  if self.args.d_opt == 'SGD':
        #      default_name += '-d' + self.args.d_opt + \
        #          "({:.4f},{:.2f})".format(self.args.d_lr, self.args.d_mom)
        #  elif self.args.d_opt == 'Adam':
        #      default_name += '-d' + self.args.d_opt + \
        #          "({:.4f},{:.2f},{:.2f})".format(self.args.d_lr, self.args.d_mom,
        #                                          self.args.d_beta2)

        if self.args.sn:
            default_name += "-SN"

        #  if self.args.gp:
        #      default_name += "-GP({:.2f})".format(self.args.gp)

        if self.args.g_ema:
            default_name += "-ema({:.3f})".format(self.args.g_ema)

        return default_name if not self.args.name else "-".join([default_name] + self.args.name)

    @property
    def is_done(self):
        return (self.iter >= self.args.train_iters or
                (self.args.fastdebug and self.iter >= self.args.fastdebug))

    def define(self):
        # Prepare dataset
        self.dataset = Dataset(self.args.dataset,  # type
                               self.datadir,
                               splits=(9, 1),  # 9/10 train, 1/10 eval
                               size=1000)  # 90000 for train, 10000 for eval

        # Create networks
        self.generator = Generator('gener', self.args)
        self.critic = Discriminator('critic', self.args)

        # Create measures
        self.P = EmpiricalMeasure('train_target', self.dataset,
                                  self.args.batch_size, split=0)
        self.P_eval = EmpiricalMeasure('eval_target', self.dataset,
                                       batch_size=1000, split=1)
        self.Z = Noise('gaussian', self.args.batch_size, 2, device=self.device)
        self.Q = InducedMeasure('model', self.generator, self.Z,
                                spec=self.args.g_opt, ema=self.args.g_ema)

        # Create metric
        self.metric = Metric('discr', self.P, self.Q, self.critic,
                             spec=self.args.d_opt)

        with PRNG.reseed(self.args.eval_seed):
            self.visualize()

    def execute(self):
        # Training
        for _ in range(self.args.train_period):
            sys.stdout.write("{}/{}\r".format(self.iter + 1, self.args.train_iters))
            sys.stdout.flush()
            if self.is_done:
                break
            self.iter += 1

            giters, diters = self.g_d_iters
            # Update Discriminator
            for _ in range(diters):
                dval = self.metric.separate(self.args.obj_type,
                                     cp_to_neg=self.args.p2neg)
            self.state.tracking.log({'discriminator loss': dval})

            # Update Generator
            for _ in range(giters):
                gval = self.metric.minimize(self.args.obj_type,
                                     nonsat=self.args.nonsat,
                                     cp_to_neg=self.args.p2neg)
            self.state.tracking.log({'generator loss': gval})

        # Evaluation and Visualization
        with PRNG.reseed(self.args.eval_seed):
            mmd_mean, mmd_std = self.visualize()
            self.state.tracking.log({'eval mmd mean': mmd_mean})
            self.state.tracking.log({'eval mmd std': mmd_std})

    def visualize(self):
        """Generate samples from a given fixed set of noise vectors and
        visualize them in a 2D plot along with contours of the discriminator,
        the norm of its gradient, and the target empirical distribution.
        """
        self.metric.eval()
        self.Q.eval()

        target_dist = []
        samples = []
        block_stats = []
        for _ in range(10):
            with torch.no_grad():
                cx = self.P_eval.sample()
                cy = self.Q.sample()
                while cy.size(0) < cx.size(0):
                    cy = torch.cat([cy, self.Q.sample()])
            cy = cy[:cx.size(0)]
            logger.info("len(cx)=%d , len(cy)=%d", len(cx), len(cy))

            target_dist.append(cx)
            samples.append(cy)
            stat = mmd2(*laplacian_mix_kernel(cx, cy)).sqrt()
            block_stats.append(stat.unsqueeze(-1))

        target_dist = torch.cat(target_dist)
        samples = torch.cat(samples)
        block_stats = torch.cat(block_stats)

        N_POINTS = 256
        RANGE = 3.

        fig, ax = plt.subplots(figsize=(FIG_X_SIZE_IN, FIG_Y_SIZE_IN), dpi=DPI)
        ax.set_xlim(-RANGE, RANGE)
        ax.set_ylim(-RANGE, RANGE)

        Xspace = np.linspace(-RANGE, RANGE, N_POINTS)
        x, y = np.meshgrid(Xspace, Xspace)
        points = np.concatenate((x.reshape(len(Xspace), len(Xspace), 1),
                                y.reshape(len(Xspace), len(Xspace), 1)), axis=2)
        points = torch.from_numpy(points).float()
        if self.args.cuda:
            points = points.cuda(device=self.device)
        points.requires_grad_()
        self.metric.requires_grad_()
        outs = self.critic(points)
        disc_map = torch.sigmoid(-outs).detach().cpu().numpy().squeeze()
        gradient = autograd.grad(
            outputs=outs,
            inputs=points,
            grad_outputs=torch.ones_like(outs),
            create_graph=False, retain_graph=False, only_inputs=True)[0]
        norm_grad_disc_map = gradient.norm(2, dim=2).cpu().numpy().squeeze()

        bot, top = Xspace[0], Xspace[-1]
        background = ax.imshow(disc_map, cmap=CMAP_DIVERGING,
                               vmin=0, vmax=1,
                               alpha=0.4, interpolation='lanczos',
                               extent=(bot, top, bot, top), origin='lower')
        CS = ax.contour(norm_grad_disc_map, cmap=CMAP_SEQUENTIAL, alpha=0.25,
                        extent=(bot, top, bot, top), origin='lower')
        #  ax.clabel(CS, inline=True, fmt='%.3f',
        #            colors='black', fontsize=MEDIUM_SIZE)

        #  fig.colorbar(background, ax=ax, format='%.3f',
        #               shrink=0.85, pad=0.02, aspect=40)

        ax.scatter(*target_dist.cpu().numpy().T, s=1,
                   marker='o', facecolors='none', edgecolors='blue')
        ax.scatter(*samples.cpu().numpy().T, c='orange',
                   s=1, marker='+')

        msg = "Eval Metric (×1e3) <mean±std>: {:4.4f}±{:4.4f} | Update Steps: {}"
        mmd_mean = block_stats.mean() * 1e3
        mmd_std = block_stats.std() * 1e3
        msg = msg.format(mmd_mean, mmd_std, self.iter)

        plt.title(msg, loc='left')
        image_path = os.path.join(self.logdir, 'step-' + str(self.iter) + '.png')
        bbox = Bbox.from_bounds(1.16, 1, 960 / DPI, 960 / DPI)
        fig.savefig(image_path, bbox_inches=bbox)
        plt.close(fig)
        return mmd_mean, mmd_std

    def animate(self):
        import imageio
        import glob
        import re

        writer = imageio.get_writer(os.path.join(self.logdir, 'training.mp4'),
                                    fps=FPS)
        paths = glob.glob(os.path.join(self.logdir, 'step-*.png'))
        paths.sort(key=lambda x: int(re.findall(r"step-(.*)\.png", x)[0]))
        for path in paths:
            im = imageio.imread(path)
            writer.append_data(im)
        writer.close()

        return self


class root(nauka.ap.Subcommand):
    """root nauka command, it serves as a parser among all.

    It is used to define directives like ``super2 <subcommand> <list of options>``.
    """

    @classmethod
    def addArgs(cls, argp):
        """Add common managerial type arguments in root command."""
        argp.add_argument(
            '-v', '--verbose',
            action='count', default=0,
            help="logging levels of information about the process (-v: INFO. -vv: DEBUG)")

    class train(nauka.ap.Subcommand):
        """Define ``train`` subcommand."""

        @classmethod
        def addArgs(cls, argp):
            """Add arguments in ``train`` subcommand."""
            mtxp = argp.add_mutually_exclusive_group()
            mtxp.add_argument("-w", "--workDir", default=None, type=str,
                              help="Full, precise path to an experiment's working directory.")
            mtxp.add_argument("-b", "--baseDir", action=nauka.ap.BaseDir)
            argp.add_argument("-d", "--dataDir", action=nauka.ap.DataDir)
            argp.add_argument("-t", "--tmpDir", action=nauka.ap.TmpDir)
            argp.add_argument("-n", "--name", default=[], type=str, action="append",
                              help="Build a name for the experiment.")
            argp.add_argument("--cuda", action=nauka.ap.CudaDevice)
            argp.add_argument(
                "-s", "--seed", default='3141592653679', type=str,
                help="Seed for PRNGs.")
            argp.add_argument(
                "-es", "--eval-seed", default=None, type=str,
                help="Frozen seed for PRNGs during evaluation/visualization.")
            argp.add_argument("--fastdebug", action=nauka.ap.FastDebug)

            argp.add_argument("--train-iters", "-it", default=100000, type=int,
                              help="Number of generator iterations to train for")
            argp.add_argument("--discr-iters", default=1, type=int,
                              help="How many discriminator iterations per generator iteration.")
            argp.add_argument("--train-period", default=100, type=int,
                              help="Period of training steps before evaluation and visualization.")
            argp.add_argument("--batch-size", "--bs", default=64, type=int,
                              help="Batch Size")

            optp = argp.add_argument_group(
                "Optimization", "Tunables for the optimization procedure.")

            optp.add_argument("--d-opt", action=nauka.ap.Optimizer,
                              default='adam:lr=0.0002,beta1=0,beta2=0.9',
                              help="Discriminator optimizer.")
            optp.add_argument("--g-opt", action=nauka.ap.Optimizer,
                              default='adam:lr=0.0001,beta1=0,beta2=0.9',
                              help="Generator optimizer.")
            optp.add_argument("--g-ema", default=None, type=float,
                              help="Exponential moving average to update test generator with.")

            taskp = argp.add_argument_group(
                "Task", "Variations on the task to be solved.")
            taskp.add_argument("--dataset", default="g8_2d", type=str,
                               choices=tuple(Dataset.types.keys()),
                               help="Dataset Selection.")

            modelp = argp.add_argument_group(
                "Architecture", "Tunables in Deep Neural Network architecture"
                " and training.")
            modelp.add_argument("--model-width", default=256, type=int,
                                help="How many units per layer.")
            modelp.add_argument("--critic-nonlin", default='ReLU', type=str,
                                choices=['ReLU', 'LeakyReLU'],
                                help="Which nonlinearity function to apply to critic's activations.")
            modelp.add_argument("--no-critic-last-bias", '-nclb', action='store_true', default=False,
                                help="Disable output layer's bias.")
            modelp.add_argument("--obj-type", default='jsd', type=str,
                                #  choices=DISCRIMINATOR_OBJECTIVES,
                                help="Type of advesarial objective function.")
            modelp.add_argument("--nonsat", action='store_true', default=False,
                                help="Use non-saturating version for `--obj-type`, if available.")
            modelp.add_argument("--p2neg", action='store_true', default=False,
                                help="Target distribution targets negative critic outputs.")
            #  modelp.add_argument("--gp", default=None, type=float,
            #                      help="Gradient norm penalty regularization constant.")
            modelp.add_argument("--sn", action='store_true', default=False,
                                help="Enable spectral normalization in discr modules.")

        @classmethod
        def run(cls, a):
            """Execute ``train`` procedure.

            :param a: arguments of subcommand, ``train simple``
            """
            verbose = a.verbose
            if verbose == 1:
                logging.basicConfig(level=logging.INFO)
            elif verbose == 2:
                logging.basicConfig(level=logging.DEBUG)
            return TwoDExperiment(a).rollback().run().animate().exitcode


def main(argv=sys.argv):
    """Create master parser for all commands and invoke appropriate ``run`` method."""
    argp = root.addAllArgs()
    a = argp.parse_args(argv[1:])
    a.__argv__ = argv
    return a.__cls__.run(a)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
