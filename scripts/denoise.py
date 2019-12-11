#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import math
import sys

import nauka
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import wandb

import ganground as gg
from ganground.state import State


class NoisyTarget(gg.nn.Module):

    def __init__(self, name, args):
        super(NoisyTarget, self).__init__(name)
        self.sigma = nn.Parameter(torch.tensor(args.sigma, dtype=torch.float32))
        self.finalize_init()

    def forward(self, *args):
        x = args[0]
        #  y = None
        #  if len(args) >= 2:
        #      y = args[1]
        epsilon = torch.randn_like(x)
        return torch.addcmul(x, self.sigma, epsilon)


class Denoise(gg.Experiment):
    @property
    def g_d_iters(self):
        diters = 0
        if self.args.discr_iters != 0:
            diters = max(self.args.discr_iters, 1)
        giters = max(-self.args.discr_iters, 1)
        return giters, diters

    @property
    def hyperparams(self):
        args = super(Denoise, self).hyperparams
        del args.train_iters
        del args.train_period
        del args.eval_seed
        return args

    @property
    def name(self):
        default_name = "{:s}".format(self.args.dataset)
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
        # Load a training dataset and bootstrap 8/10 size
        self.d1 = gg.Dataset(self.args.dataset, self.datadir,
                             splits=(8, 2),
                             train=True)
        self.P = gg.EmpiricalMeasure('clean', self.d1,
                                     self.args.batch_size, split=0)

        # Load a training dataset and bootstrap 8/10 size (second time)
        self.d2 = gg.Dataset(self.args.dataset, self.datadir,
                             splits=(8, 2),
                             train=True)
        self.P2 = gg.EmpiricalMeasure('clean2', self.d2,
                                      self.args.batch_size, split=0)
        self.distortion = NoisyTarget('gaussian_blur', self.args)
        self.noisy_P = gg.InducedMeasure('blurred', self.distortion, self.P2,
                                         spec=self.args.g_opt, ema=self.args.g_ema)

        # Load testing dataset
        self.d_test = gg.Dataset(self.args.dataset, self.datadir, train=False)

        # Determine sample shape (C x H x W)
        sample, label = self.d1.data[0][0]
        self.d_shape = tuple(sample.size())
        self.logging.debug('sample shape is: %s', self.d_shape)
        self.logging.debug('dataset size is: %d', self.d1.N)
        assert(self.d_shape[1] == self.d_shape[2])  # H == W
        # Calculate minimum number of downsampling steps until 4 x 4
        log_x_shape_f, log_x_shape_i = math.modf(math.log(self.d_shape[1], 2))
        default_n_steps = int(log_x_shape_i) - 2
        # Aggregate critic network arguments  (Check cmd-line arguments as well)
        critic_args = dict(
            dim_h=self.args.dim_h,  # Starting channel size, use conv C=3 -> C=dim_h
            dim_h_max=1024,  # Maximum channel size in network
            n_steps=self.args.n_steps or default_n_steps,  # Number of resblocks
            incl_attblock=self.args.atten,  # list of step number to include a self-attention block
            f_size=self.args.f_size,  # convolution kernel size
            spectral_norm=self.args.sn,  # Use Spectral Normalization?
            batch_norm=self.args.bn,  # Use Batch Normalization?
            nonlinearity=self.args.nonlin,  # Type of nonlinearity used
            )

        # Define critic network and metric
        self.critic = gg.nn.Encoder('critic', self.d_shape, dim_out=1, **critic_args)
        self.metric = gg.Metric('discr', self.P, self.noisy_P, self.critic,
                                spec=self.args.d_opt)

        if self.iter == 0:
            self.info.sigma_summary = torch.tensor([])
            self.eval()

    def execute(self):
        self.metric.train()
        self.noisy_P.train()
        metric_summary = []
        loss_summary = []
        sigma_summary = []
        for _ in range(self.args.train_period):
            sys.stdout.write("{}/{}\r".format(self.iter + 1, self.args.train_iters))
            sys.stdout.flush()
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
            for _ in range(giters):
                gval = self.metric.minimize(self.args.obj, self.args.reg,
                                            nonsat=self.args.nonsat,
                                            cp_to_neg=self.args.p2neg)
                loss_summary.append(gval.unsqueeze(0))
                sigma_summary.append(self.distortion.sigma.data.detach().clone().unsqueeze(0))

        self.info.sigma_summary = torch.cat((self.info.sigma_summary,
                                             torch.cat(sigma_summary).cpu()))
        self.log(metric=torch.cat(metric_summary).mean(),
                 gener_loss=torch.cat(loss_summary).mean(),
                 sigma=self.info.sigma_summary[-1])
        self.eval()

    def eval(self):
        self.metric.eval()

        with gg.PRNG.reseed(self.args.eval_seed):
            self.P_t = gg.EmpiricalMeasure('clean_t', self.d_test, batch_size=8,
                                           resume=False)
            self.noisy_P_t = gg.InducedMeasure('blurred_t', self.distortion, self.P_t)
            self.noisy_P_t.eval()
            #  cP = gg.InducedMeasure('critic#P_t', self.critic, self.P_t)
            #  cQ = gg.InducedMeasure('critic#noisy_P_t', self.critic, self.noisy_P_t)
            with self.P_t.hold_samples():
                ground, _ = self.P_t.sample()
                distorted = self.noisy_P_t.sample()
            #  dist_t = self.metric.estimate(self.args.obj,
            #                                cp_to_neg=self.args.p2neg)
            #  cp = cP.sample()
            #  cq = cQ.sample()

        nrow = 8
        imgs = torch.stack((ground[:nrow], distorted[:nrow]), 0).view(-1, self.d_shape[0],
                                                                      self.d_shape[1], self.d_shape[2])
        imgs = imgs.float().detach().cpu()
        image_path = os.path.join(self.logdir, 'samples-' + str(self.iter) + '.png')
        torchvision.utils.save_image(imgs, image_path, nrow=nrow,
                                     normalize=True, range=(-1., 1.),
                                     pad_value=255, padding=2)
        #  self.log(samples=wandb.Image(image_path))

        return self

    def download(self):
        gg.Dataset(self.args.dataset, self.datadir,
                   download=True, load=False)
        return 0


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
            argp.add_argument("--train-iters", "-it", default=50000, type=int,
                              help="Number of generator iterations to train for")
            argp.add_argument("--discr-iters", default=1, type=int,
                              help="How many discriminator iterations per generator iteration.")
            argp.add_argument("--train-period", default=500, type=int,
                              help="Period of training steps before evaluation and visualization.")
            argp.add_argument("--batch-size", "--bs", default=64, type=int,
                              help="Batch Size")

            taskp = argp.add_argument_group(
                "Task", "Variations on the task to be solved.")
            taskp.add_argument("--dataset", default="cifar10", type=str,
                               help="Dataset Selection: " + str(tuple(gg.Dataset.types.keys())))

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
            modelp.add_argument("--sigma", default=0.666, type=float,
                                help="Initial standard deviation for gaussian measure blur.")
            modelp.add_argument("--nonlin", default='ReLU', type=str,
                                choices=['ReLU', 'LeakyReLU'],
                                help="Which nonlinearity function to apply to critic's activations.")
            modelp.add_argument("--dim-h", default=64, type=int,
                                help="Channel size after/before up/down sampling steps.")
            modelp.add_argument("--n-steps", default=None, type=int,
                                help="Number of resblocks used.")
            modelp.add_argument("--atten", default=[], type=str, action="append",
                                help="Number of steps to insert self-attention blocks.")
            modelp.add_argument("--f-size", default=3, type=int,
                                help="Convolutional network kernel size.")
            modelp.add_argument("--sn", action='store_true', default=False,
                                help="Enable spectral normalization in discr modules.")
            modelp.add_argument("--bn", action='store_true', default=False,
                                help="Enable batch normalization in discr modules.")

            modelp.add_argument("--obj", action=gg.metric.ObjectiveAction,
                                default='jsd',
                                help="Advesarial objective function.")
            modelp.add_argument("--nonsat", action='store_true', default=False,
                                help="Use non-saturating version for `--obj-type`, if available.")
            modelp.add_argument("--p2neg", action='store_true', default=False,
                                help="Target distribution targets negative critic outputs.")
            modelp.add_argument("--reg", action=gg.metric.ObjectiveAction,
                                default=None,
                                help="Regularizers used.")

        @classmethod
        def run(cls, a):
            """Execute ``train`` procedure.

            :param a: arguments of subcommand, ``train simple``
            """
            return Denoise(a).rollback().run().exitcode

        class download(nauka.ap.Subcommand):
            """Define ``download`` subcommand."""

            @classmethod
            def run(cls, a):
                """Execute ``download`` procedure.

                :param a: arguments of subcommand, ``train simple``
                """
                return Denoise(a).download()


def main(argv=sys.argv):
    """Create master parser for all commands and invoke appropriate ``run`` method."""
    argp = root.addAllArgs()
    a = argp.parse_args(argv[1:])
    a.__argv__ = argv
    return a.__cls__.run(a)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
