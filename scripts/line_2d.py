#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
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

import ganground as gg
from ganground.metric.kernel import (mmd2, AbstractKernel,
                                     cross_mean_kernel_wrap,
                                     _pairwise_dist)
from ganground.state import State


# Plot settings
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 18
FIG_X_SIZE_IN = 12
FIG_Y_SIZE_IN = 12
DPI = 96
FPS = 10
CMAP_DIVERGING = mpl.cm.bwr_r
CMAP_SEQUENTIAL = mpl.cm.plasma_r
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


class Noise(gg.Measure):
    def __init__(self, name: str, *args, **kwargs):
        super(Noise, self).__init__()
        self.name = name
        self.args = args
        self.kwargs = kwargs

    def sample_(self):
        return torch.randn(*self.args, **self.kwargs)


class Generator(gg.nn.Module):

    def __init__(self, name, args):
        super(Generator, self).__init__(name)
        self.slope = nn.Parameter(torch.tensor(args.slope, dtype=torch.float32))
        self.y0 = nn.Parameter(torch.tensor(args.y0, dtype=torch.float32))
        self.finalize_init()

    def forward(self, noise):
        y = self.slope * noise + self.y0
        return torch.cat((noise, y), axis=-1)


class Critic(gg.nn.Module):

    def __init__(self, name, args):
        super(Critic, self).__init__(name)
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
        critic = nn.Sequential(
            Linear(2, DIM),
            nonlin(inplace=True),
            Linear(DIM, DIM),
            nonlin(inplace=True),
            Linear(DIM, DIM),
            nonlin(inplace=True),
        )
        self.critic = critic
        self.critic_output = Linear(DIM, 1, bias=(not args.no_critic_last_bias))
        self.finalize_init()

    def forward(self, inputs):
        return self.critic_output(self.critic(inputs))


class Line2D(gg.Experiment):
    @property
    def g_d_iters(self):
        diters = 0
        if self.args.discr_iters != 0:
            diters = max(self.args.discr_iters, 1)
        giters = max(-self.args.discr_iters, 1)
        return giters, diters

    @property
    def hyperparams(self):
        args = super(Line2D, self).hyperparams
        del args.train_iters
        del args.train_period
        del args.eval_seed
        return args

    @property
    def name(self):
        default_name = ''
        if self.args.nogen is True:
            default_name += 'nogen'
        else:
            default_name += 'gen'
        default_name += "-sl({:.3f})-y0({:.3f})".format(self.args.slope, self.args.y0)
        obj_type = dict()
        obj_type.update(**vars(self.args.obj))
        if self.args.reg:
            obj_type.update(**vars(self.args.reg))
        str_list = []
        for n, c in obj_type.items():
            if c == 1:
                str_list.append(n)
            else:
                str_list.append('{}({:.4f})'.format(n, c))
        default_name += "-{:s}".format('-'.join(str_list))
        giters, diters = self.g_d_iters
        if giters != diters or giters != 1:
            default_name += "({:d}-{:d})".format(diters, giters)

        default_name += '-d'
        default_name += self.args.d_opt.name
        if self.args.d_opt.name == 'sgd':
            default_name += '({:.4f},{:.2f})'.format(self.args.d_opt.lr,
                                                     self.args.d_opt.mom)
        elif self.args.d_opt.name == 'adam':
            default_name += '({:.4f},{:.2f},{:.2f})'.format(self.args.d_opt.lr,
                                                            self.args.d_opt.beta1,
                                                            self.args.d_opt.beta2)
        if self.args.sn:
            default_name += "-SN"

        if self.args.nogen is False:
            default_name += '-g'
            default_name += self.args.g_opt.name
            if self.args.g_opt.name == 'sgd':
                default_name += '({:.4f},{:.2f})'.format(self.args.g_opt.lr,
                                                        self.args.g_opt.mom)
            elif self.args.g_opt.name == 'adam':
                default_name += '({:.4f},{:.2f},{:.2f})'.format(self.args.g_opt.lr,
                                                                self.args.g_opt.beta1,
                                                                self.args.g_opt.beta2)
            if self.args.g_ema:
                default_name += "-ema({:.3f})".format(self.args.g_ema)

        return "-".join([default_name] + self.args.name + [self.hash(10)])

    @property
    def is_done(self):
        return (self.iter >= self.args.train_iters or
                (self.args.fastdebug and self.iter >= self.args.fastdebug))

    def define(self):
        self.Z = Noise('1d_g', self.args.batch_size, 1, device=self.device)
        self.g1 = Generator('gener1', self.args)
        self.Q = gg.InducedMeasure('model', self.g1, self.Z,
                                   spec=self.args.g_opt, ema=self.args.g_ema)

        args_ = copy.deepcopy(self.args)
        args_.slope = 0
        args_.y0 = 0
        self.g2 = Generator('gener2', args_)
        self.P = gg.InducedMeasure('target', self.g2, self.Z, spec=None)

        self.critic = Critic('critic', self.args)
        self.metric = gg.Metric('discr', self.P, self.Q, self.critic,
                                spec=self.args.d_opt)

        if self.iter == 0:
            self.info.final_metric = list()
            with gg.PRNG.reseed(self.args.eval_seed):
                self.eval(viz=True)

    def execute(self):
        # Training
        self.metric.train()
        self.Q.train()
        metric_summary = []
        loss_summary = []
        for _ in range(self.args.train_period):
            if self.is_done:
                break
            self.iter += 1

            giters, diters = self.g_d_iters
            # Update Discriminator
            for _ in range(diters):
                metric = self.metric.separate(self.args.obj, self.args.reg,
                                              cp_to_neg=self.args.p2neg)
                metric_summary.append(metric.unsqueeze(0))

            # Update Generator
            if self.args.nogen is False:
                for _ in range(giters):
                    gval = self.metric.minimize(self.args.obj, self.args.reg,
                                                nonsat=self.args.nonsat,
                                                cp_to_neg=self.args.p2neg)
                    loss_summary.append(gval.unsqueeze(0))

            self.log(metric=torch.cat(metric_summary).mean())
            if self.args.nogen is False:
                self.log(**{'generator loss': torch.cat(loss_summary).mean()})
                self.log(slope=self.g1.slope.data.clone().detach())
                self.log(y0=self.g1.y0.data.clone().detach())

            with gg.PRNG.reseed(self.args.eval_seed):
                self.eval(viz=self.iter % self.args.train_period == 0 or self.is_done)

    def eval(self, viz=False):
        """Generate samples from a given fixed set of noise vectors and
        visualize them in a 2D plot along with contours of the discriminator,
        the norm of its gradient, and the target empirical distribution.
        """
        self.metric.eval()
        self.Q.eval()

        target_dist = []
        samples = []
        block_stats = []
        block_stats2 = []
        for _ in range(16):
            with self.P.hold_samples(), self.Q.hold_samples():
                with torch.no_grad():
                    cx = self.P.sample()
                    cy = self.Q.sample()
                    #  while cy.size(0) < cx.size(0):
                    #      cy = torch.cat([cy, self.Q.sample()])
                #  cy = cy[:cx.size(0)]
                assert(cx.size(0) == cy.size(0))
                target_dist.append(cx)
                samples.append(cy)

                stat = mmd2(*laplacian_mix_kernel(cx, cy)).sqrt()
                block_stats.append(stat.unsqueeze(-1))

                stat2 = self.metric.estimate(self.args.obj, cp_to_neg=self.args.p2neg)
                block_stats2.append(stat2.unsqueeze(-1))

        target_dist = torch.cat(target_dist)
        samples = torch.cat(samples)
        block_stats = torch.cat(block_stats)
        block_stats2 = torch.cat(block_stats2)

        mmd_mean = block_stats.mean() * 1e3
        mmd_std = block_stats.std() * 1e3
        self.log(**{'eval mmd mean (×1e3)': mmd_mean})
        self.log(**{'eval mmd std (×1e3)': mmd_std})

        metric_mean = block_stats2.mean()
        self.info.final_metric.append(metric_mean)
        metric_std = block_stats2.mean()
        self.log(**{'eval metric mean': metric_mean})
        self.log(**{'eval metric std': metric_std})

        if viz is True:
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
            outs = self.critic(points)
            disc_map = torch.sigmoid(outs).detach().cpu().numpy().squeeze()

            bot, top = Xspace[0], Xspace[-1]
            ax.imshow(disc_map, cmap=CMAP_DIVERGING,
                      vmin=0.45, vmax=0.55,
                      alpha=0.4, interpolation='lanczos',
                      extent=(bot, top, bot, top), origin='lower')
            CS = ax.contour(disc_map, cmap=CMAP_SEQUENTIAL, alpha=0.3,
                            extent=(bot, top, bot, top), origin='lower')
            ax.clabel(CS, inline=True, fmt='%.3f',
                      colors='black', fontsize=MEDIUM_SIZE)

            ax.scatter(*target_dist.cpu().numpy().T, s=10,
                       marker='o', facecolors='none', edgecolors='blue')
            ax.scatter(*samples.cpu().numpy().T, c='red',
                       s=10, marker='+')

            msg = "Eval Metric (×1e3) <mean±std>: {:4.4f}±{:4.4f} | Update Steps: {}"
            msg = msg.format(mmd_mean, mmd_std, self.iter)

            plt.title(msg, loc='left')
            image_path = os.path.join(self.logdir, 'step-' + str(self.iter) + '.png')
            bbox = Bbox.from_bounds(1.16, 1, 960 / DPI, 960 / DPI)
            fig.savefig(image_path, bbox_inches=bbox)
            self.log(viz=fig)
            plt.close(fig)

        return self

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
        argp.add_argument('-v', action=gg.logging.LogAction)

    class train(nauka.ap.Subcommand):
        """Define ``train`` subcommand."""

        @classmethod
        def addArgs(cls, argp):
            """Add arguments in ``train`` subcommand."""
            mtxp = argp.add_mutually_exclusive_group()
            mtxp.add_argument("-work", "--workdir", default=None, type=str,
                              help="Full, precise path to an experiment's working directory.")
            mtxp.add_argument("-base", "--basedir", action=nauka.ap.BaseDir)
            argp.add_argument("-data", "--datadir", action=nauka.ap.DataDir)
            argp.add_argument("-tmp", "--tmpdir", action=nauka.ap.TmpDir)
            argp.add_argument("-n", "--name", default=[], type=str, action="append",
                              help="Build a name for the experiment.")
            argp.add_argument("--cuda", action=nauka.ap.CudaDevice)
            argp.add_argument("-t", action=gg.tracking.WandbAction,
                              default="pgm_gan_a19")
            argp.add_argument("--fastdebug", action=nauka.ap.FastDebug)

            argp.add_argument(
                "-s", "--seed", default='0x6a09e667f3bcc908', type=str,
                help="Seed for PRNGs. Default is 64-bit fractional expansion of sqrt(2).")
            argp.add_argument(
                "-es", "--eval-seed", default='pi=3.14159', type=str,
                help="Frozen seed for PRNGs during evaluation/visualization.")
            argp.add_argument("--train-iters", "-it", default=100, type=int,
                              help="Number of generator iterations to train for")
            argp.add_argument("--discr-iters", default=1, type=int,
                              help="How many discriminator iterations per generator iteration.")
            argp.add_argument("--train-period", default=10, type=int,
                              help="Period of training steps before evaluation and visualization.")
            argp.add_argument("--batch-size", "--bs", default=64, type=int,
                              help="Batch Size")
            argp.add_argument("--nogen", default=False, action='store_true',
                              help="Do not train the generator.")

            optp = argp.add_argument_group(
                "Optimization", "Tunables for the optimization procedure.")

            optp.add_argument("--d-opt", action=nauka.ap.Optimizer,
                              default='adam:lr=0.01,beta1=0,beta2=0.9',
                              help="Discriminator optimizer.")
            optp.add_argument("--g-opt", action=nauka.ap.Optimizer,
                              default='adam:lr=0.005,beta1=0,beta2=0.9',
                              help="Generator optimizer.")
            optp.add_argument("--g-ema", default=None, type=float,
                              help="Exponential moving average to update test generator with.")

            modelp = argp.add_argument_group(
                "Architecture", "Tunables in Deep Neural Network architecture"
                " and training.")
            modelp.add_argument("--slope", default=(1/np.sqrt(2)), type=float,
                                help="Slope of initial model support.")
            modelp.add_argument("--y0", default=1., type=float,
                                help="Intersection with y-axis of initial model support.")
            modelp.add_argument("--model-width", default=256, type=int,
                                help="How many units per layer.")
            modelp.add_argument("--critic-nonlin", default='ReLU', type=str,
                                choices=['ReLU', 'LeakyReLU'],
                                help="Which nonlinearity function to apply to critic's activations.")
            modelp.add_argument("--no-critic-last-bias", '-nclb', action='store_true', default=False,
                                help="Disable output layer's bias.")

            modelp.add_argument("--obj", action=gg.metric.ObjectiveAction,
                                default='jsd',
                                help="Advesarial objective function.")
            modelp.add_argument("--reg", action=gg.metric.ObjectiveAction,
                                default=None,
                                help="Regularizers used.")
            modelp.add_argument("--sn", action='store_true', default=False,
                                help="Enable spectral normalization in discr modules.")

            modelp.add_argument("--nonsat", action='store_true', default=False,
                                help="Use non-saturating version for `--obj-type`, if available.")
            modelp.add_argument("--p2neg", action='store_true', default=False,
                                help="Target distribution targets negative critic outputs.")

        @classmethod
        def run(cls, a):
            """Execute ``train`` procedure.

            :param a: arguments of subcommand, ``train simple``
            """
            return Line2D(a).rollback().run().animate().exitcode

        class optd(nauka.ap.Subcommand):
            """Define ``optd`` subcommand."""

            @classmethod
            def addArgs(cls, argp):
                argp.add_argument("--nps", default=25, type=int)
                argp.add_argument("--print-iter", '-piter', default=-1, type=int)

            @classmethod
            def run(cls, a):
                """Execute ``train`` procedure.

                :param a: arguments of subcommand, ``train simple``
                """
                a.nogen = False
                a.name.append('optd')
                N_POINTS = int(a.nps)
                piter = int(a.print_iter)
                iters = a.train_iters + 1 if not a.fastdebug else a.fastdebug + 1
                piter = piter if piter >= 0 else piter + iters
                del a.nps
                del a.print_iter
                RANGE = 2.

                SlopeSpace = np.linspace(-RANGE, RANGE, N_POINTS)
                Y0Space = np.linspace(-RANGE, RANGE, N_POINTS)
                x, y = np.meshgrid(SlopeSpace, Y0Space)
                points = np.concatenate(
                    (x.reshape(len(SlopeSpace), len(SlopeSpace), 1),
                     y.reshape(len(Y0Space), len(Y0Space), 1)), axis=2)
                metric_map = np.zeros((N_POINTS, N_POINTS, iters))
                for x, row in enumerate(points):
                    for y, (sl, y0) in enumerate(row):
                        try:
                            a_ = copy.deepcopy(a)
                        except:
                            a_ = a
                        a_.slope = sl
                        a_.y0 = y0
                        exp = Line2D(a_)
                        metric_map[x, y, :] = np.asarray(exp.rollback().run().info.final_metric)
                        State.instance = None

                fig, ax = plt.subplots(figsize=(FIG_X_SIZE_IN, FIG_Y_SIZE_IN), dpi=DPI)
                ax.set_xlim(-RANGE, RANGE)
                ax.set_ylim(-RANGE, RANGE)
                bot, top = SlopeSpace[0], SlopeSpace[-1]
                selected_map = metric_map[:, :, piter]
                bg = ax.imshow(selected_map, cmap=CMAP_SEQUENTIAL,
                               #  vmin=0.45, vmax=0.55,
                               alpha=0.5, interpolation='lanczos',
                               extent=(bot, top, bot, top), origin='lower')
                #  CS = ax.contour(selected_map, cmap=CMAP_SEQUENTIAL, alpha=1,
                #                  extent=(bot, top, bot, top), origin='lower')
                #  ax.clabel(CS, inline=True, fmt='%.3f',
                #            colors='black', fontsize=MEDIUM_SIZE)
                fig.colorbar(bg, ax=ax, format='%.3f',
                             shrink=0.85, pad=0.02, aspect=40)

                objective = "+".join(vars(a.obj).keys())  # TODO perhaps method
                msg = "{}(P(slope=0,y0=0), P(slope=x,y0=y)) | Update Steps: {}"
                plt.title(msg.format(objective, piter), loc='left')
                pworkdir = Line2D.project_logdir(a)
                Line2D.mkdirp(pworkdir)
                image_path = os.path.join(pworkdir,
                                          '{}_{}.png'.format(objective, piter))
                bbox = Bbox.from_bounds(1.16, 1, 960 / DPI, 960 / DPI)
                fig.savefig(image_path, bbox_inches=bbox)
                plt.close(fig)


def main(argv=sys.argv):
    """Create master parser for all commands and invoke appropriate ``run`` method."""
    argp = root.addAllArgs()
    a = argp.parse_args(argv[1:])
    a.__argv__ = argv
    return a.__cls__.run(a)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
