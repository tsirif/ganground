#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys

from mpl_toolkits.mplot3d import axes3d
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
import torch.optim as optim

import ganground as gg
from ganground.metric import ObjectiveBuilder


# Plot settings
CMAP = mpl.cm.plasma
#  CMAP = mpl.cm.inferno
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


class Dirac1D(gg.Experiment):
    @property
    def g_d_iters(self):
        diters = 0
        if self.args.discr_iters != 0:
            diters = max(self.args.discr_iters, 1)
        giters = max(-self.args.discr_iters, 1)
        return giters, diters

    @property
    def hyperparams(self):
        args = super(Dirac1D, self).hyperparams
        del args.train_iters
        del args.eval_seed
        return args

    @property
    def name(self):
        default_name = ''
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
        default_name += "{:s}".format('-'.join(str_list))

        giters, diters = self.g_d_iters
        if giters != diters or giters != 1:
            default_name += "({:d}-{:d})".format(diters, giters)

        default_name += '-sgd({:.4f},{:.2f})'.format(self.args.lr,
                                                     self.args.mom)

        if self.args.sn:
            default_name += "-SN"

        if not self.args.name:
            return "-".join([default_name, self.hash(10)])
        return "-".join([default_name] + self.args.name + [self.hash(10)])

    @property
    def is_done(self):
        return (self.iter >= self.args.train_iters or
                (self.args.fastdebug and self.iter >= self.args.fastdebug))

    @property
    def models(self):
        return self.info.models

    @models.setter
    def models(self, models_):
        self.info.models = models_

    def define(self):
        # Prepare target
        self.target = torch.Tensor(1).fill_(self.args.target)

        self.optimizers = nauka.utils.PlainObject()
        if self.iter == 0:
            self.models = nauka.utils.PlainObject()
            self.models.gener = torch.Tensor(1, 1).normal_(std=1.5)
        self.models.gener.requires_grad_()
        self._create_optimizer('gener')

        if self.iter == 0:
            self.models.discr = torch.Tensor(1, 2).normal_(std=1.5)
            if self.args.sn:
                self.models.discr[:, 0] = 1
        self.models.discr.requires_grad_()
        self._create_optimizer('discr')

        if self.iter == 0:
            self.info.trajectory = list()
            self.extend_trajectory()

    def _create_optimizer(self, name):
        optimizer = optim.SGD([getattr(self.models, name)],
                              self.args.lr, self.args.mom,
                              nesterov=False,
                              weight_decay=self.args.l2)
        optimizer = self.state.register_optimizer(name, optimizer)
        setattr(self.optimizers, name, optimizer)

    def extend_trajectory(self):
        discr_val = self.models.discr.view(-1) if self.models.discr is not None else torch.zeros(2)
        point = torch.cat((discr_val,
                           self.models.gener.view(-1))).view(-1, 1)
        self.info.trajectory.append(point.detach())

    def discriminate(self, x):
        psi_0, psi_1 = self.models.discr[:, 0], self.models.discr[:, 1]
        return psi_0 * x + psi_1

    def discr_loss(self):
        obj = ObjectiveBuilder(**vars(self.args.obj))
        if self.args.reg:
            reg_spec = vars(self.args.reg)
        else:
            reg_spec = dict()
        reg = ObjectiveBuilder(**reg_spec)

        p = self.target
        #  if self.args.std_p > 0:
        #      p = p + torch.randn_like(p) * self.args.std_p
        cp = self.discriminate(p)

        q = self.models.gener
        #  if self.args.std_q > 0:
        #      q = q + torch.randn_like(q) * self.args.std_q
        cq = self.discriminate(q)

        discr_loss = - obj.estimate_metric(cp, cq, cp_to_neg=self.args.p2neg)
        discr_loss = discr_loss - reg.estimate_metric(p, q, self.discriminate)

        return discr_loss, cp, cq

    def gener_loss(self):
        obj = ObjectiveBuilder(**vars(self.args.obj))
        p = self.target
        #  if self.args.std_p > 0:
        #      p = p + torch.randn_like(p) * self.args.std_p
        cp = self.discriminate(p)

        q = self.models.gener
        #  if self.args.std_q > 0:
        #      q = q + torch.randn_like(q) * self.args.std_q
        cq = self.discriminate(q)

        gener_loss = obj.estimate_measure_loss(cp, cq,
                                               cp_to_neg=self.args.p2neg,
                                               nonsat=self.args.nonsat)

        return gener_loss

    def execute(self):
        # Training
        sys.stdout.write("{}/{}\r".format(self.iter + 1, self.args.train_iters))
        sys.stdout.flush()
        self.iter += 1

        giters, diters = self.g_d_iters
        # Update Discriminator
        for _ in range(diters):
            self.optimizers.discr.zero_grad()
            discr_loss, c_p, c_q = self.discr_loss()
            discr_loss.backward()
            self.optimizers.discr.step()
            if self.args.sn:
                with torch.no_grad():
                    self.models.discr[:, 0] = 1

        # Update Generator
        for _ in range(giters):
            self.optimizers.gener.zero_grad()
            gener_loss = self.gener_loss()
            gener_loss.backward()
            self.optimizers.gener.step()

        self.extend_trajectory()

    def plot(self):
        trajectory = torch.cat(self.info.trajectory, 1).detach().numpy()
        self.logging.debug("Final trajectory contains: {}".format(trajectory.shape))

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        ax.plot(*trajectory, label='training trajectory')

        init = trajectory.T[0]
        final = trajectory.T[-1]
        self.logging.info("Initial point: {}".format(init))
        self.logging.info("Final point: {}".format(final))
        ax.plot([init[0]], [init[1]], [init[2]], 'g+', ms=10, label='initial')

        obj_type = list(vars(self.args.obj).keys())[0]
        if obj_type not in ('w1', 'rgan', 'mmd2'):
            margin = max(abs(final[0]) + 0.5, 3)
            x = np.linspace(-margin, margin, 128)
            y = - self.args.target * x
            ax.plot(x, y, [self.args.target] * 128, 'orange', lw=2, label='target')
            ax.plot([0], [0], [self.args.target], 'orange', marker='o', ms=5)
        else:
            margin = max(abs(final[1]) + 0.5, 3)
            y = np.linspace(-margin, margin, 128)
            ax.plot([0] * 128, y, [self.args.target] * 128, 'orange', lw=2, label='target')

        ax.legend()
        ax.set_xlabel(r"$\psi_0$")
        ax.set_ylabel(r"$\psi_1$")
        ax.set_zlabel(r"$\theta$")
        plt.show()
        return self

    def viz(self):
        self.target = torch.Tensor(1).fill_(self.args.target)
        grid = np.meshgrid(np.linspace(-3, 3, 25), np.linspace(-3, 3, 25))
        x = torch.from_numpy(grid[0].astype('float32')).view(-1, 1)
        y = torch.from_numpy(grid[1].astype('float32')).view(-1, 1)
        r = torch.cat((x, y), 1)
        V = torch.Tensor(2, 2)
        align = np.arctan2(-self.args.target, 1)
        V[0, 0] = np.cos(align)
        V[1, 0] = - np.sin(align)
        V[0, 1] = - V[1, 0]
        V[1, 1] = V[0, 0]
        r = torch.matmul(r, V)
        p = torch.cat((r, torch.zeros_like(x).view(-1, 1)), 1)

        rot = (self.args.rot - 90) * np.pi / 180.
        axis = np.array([1., -self.args.target])
        axis = axis / np.sqrt(axis.dot(axis))
        ux, uy = axis
        R = torch.Tensor(3, 3)
        R[0, 0] = np.cos(rot) + ux**2 * (1 - np.cos(rot))
        R[1, 0] = ux * uy * (1 - np.cos(rot))
        R[2, 0] = uy * np.sin(rot)
        R[0, 1] = R[1, 0]
        R[1, 1] = np.cos(rot) + uy**2 * (1 - np.cos(rot))
        R[2, 1] = - ux * np.sin(rot)
        R[0, 2] = - R[2, 0]
        R[1, 2] = - R[2, 1]
        R[2, 2] = np.cos(rot)

        p = torch.matmul(p, R)
        p[:, 2] = p[:, 2] + self.args.target
        p.requires_grad_()

        discr = p[:, :2]
        gener = p[:, 2]

        discr_loss, _, _ = self.discr_loss(discr, gener)
        gener_loss = self.gener_loss(discr, gener)

        grad_d = autograd.grad(outputs=discr_loss, inputs=discr,
                               grad_outputs=torch.ones_like(discr_loss),
                               only_inputs=True, allow_unused=True)[0]
        grad_g = autograd.grad(outputs=gener_loss, inputs=gener,
                               grad_outputs=torch.ones_like(gener_loss),
                               only_inputs=True)[0]

        field = - torch.cat((grad_d, grad_g.view(-1, 1)), 1)

        field_r = torch.matmul(field, R.transpose(1, 0))
        field_xy = field_r
        field_xy[:, :2] = torch.matmul(field_r[:, :2], V.transpose(1, 0))
        u = field_xy[:, 0]
        v = field_xy[:, 1]
        c = field_xy[:, 2]
        range_c = max(abs(c)).item()
        norm = mpl.colors.Normalize(vmin=-range_c, vmax=range_c)

        # 3D Debug
        #  fig = plt.figure()
        #  ax = fig.gca(projection='3d')
        #  ax.set_xlim(-3, 3)
        #  ax.set_ylim(-3, 3)
        #  ax.set_zlim(-self.args.target - 3, self.args.target + 3)
        #  a = np.linspace(-3, 3, 128)
        #  b = - self.args.target * a
        #  ax.plot(a, b, [self.args.target] * 128, 'orange', lw=2, label='target')
        #  ax.plot([0], [0], [self.args.target], 'orange', marker='o', ms=5)
        #  ax.plot([p[0, 0]], [p[0, 1]], [p[0, 2]], 'gx', label='(-3, -3)')
        #  ax.plot([p[24, 0]], [p[24, 1]], [p[24, 2]], 'rx', label='(3, -3)')
        #  ax.scatter(p[:, 0].detach().numpy(),
        #             p[:, 1].detach().numpy(),
        #             p[:, 2].detach().numpy(), label='points')
        #  ax.quiver(*p.detach().numpy().T, *field.detach().numpy().T,
        #            normalize=True, length=1)
        #  ax.legend()
        #  ax.set_xlabel(r"$\psi_0$")
        #  ax.set_ylabel(r"$\psi_1$")
        #  ax.set_zlabel(r"$\theta$")

        fig, ax = plt.subplots()
        ax.quiver(x.numpy(), y.numpy(), u.numpy(), v.numpy(),
                  color=CMAP(norm(c.numpy())), scale=1,
                  units='xy', angles='xy', scale_units='xy', pivot='tail')
        cbar_ax, _ = mpl.colorbar.make_axes(ax)
        mpl.colorbar.ColorbarBase(cbar_ax, cmap=CMAP, norm=norm)
        plt.show()


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
                              default=None)
            argp.add_argument("--fastdebug", action=nauka.ap.FastDebug)

            argp.add_argument(
                "-s", "--seed", default='0x6a09e667f3bcc908', type=str,
                help="Seed for PRNGs. Default is 64-bit fractional expansion of sqrt(2).")
            argp.add_argument(
                "-es", "--eval-seed", default='pi=3.14159', type=str,
                help="Frozen seed for PRNGs during evaluation/visualization.")
            argp.add_argument("--train-iters", "-it", default=1000, type=int,
                              help="Number of generator iterations to train for")
            argp.add_argument("--discr-iters", default=1, type=int,
                              help="How many discriminator iterations per generator iteration.")
            argp.add_argument("--target", default=1.0, type=float,
                              help="Target for the generator to match.")

            optp = argp.add_argument_group(
                "Optimization", "Tunables for the optimization procedure.")

            optp.add_argument("--l2", default=0, type=float, help="L2 penalty.")
            optp.add_argument("--lr", default=1e-1, type=float,
                              help="Learning rate of optimizers.")
            optp.add_argument("--mom", default=0, type=float,
                              help="(Polyak) momentum of stochastic optimizers.")

            modelp = argp.add_argument_group(
                "Architecture", "Tunables in Deep Neural Network architecture"
                " and training.")
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
            return Dirac1D(a).rollback().run().plot().exitcode

        class viz(nauka.ap.Subcommand):
            @classmethod
            def addArgs(cls, argp):
                argp.add_argument(
                    "-rot", default=0, type=float,
                    help=r"The rotation of a plane around the line "
                         r"`h_\psi(target) = 0, \theta = target`"
                         r", on which dynamics field is going to be evaluated and projected."
                         r" `-rot=0` is the plane `h_\psi(target) = 0`.")

            @classmethod
            def run(cls, a):
                """Execute ``train`` procedure."""
                root.run(a)
                Dirac1D(a).viz()
                return 0


def main(argv=sys.argv):
    """Create master parser for all commands and invoke appropriate ``run`` method."""
    argp = root.addAllArgs()
    a = argp.parse_args(argv[1:])
    a.__argv__ = argv
    return a.__cls__.run(a)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
