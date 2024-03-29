# -*- coding: utf-8 -*-
r"""
:mod:`ganground.data.clustersets` -- Synthetic datasets for clustering
======================================================================

.. module:: clustersets
   :platform: Unix
   :synopsis: Basically everything found in this website
      <https://cs.joensuu.fi/sipu/datasets/>_ (for now).

Collection of datasets (mostly 2D) used primarily for benchmarking of
inference algorithms and interpretable experiments in the input space.

"""
import errno
import itertools as it
import os

import torch
from torch.utils.data import TensorDataset

from ganground.data import AbstractDataset


class _SmallDataset(AbstractDataset):

    def download(self, root):
        """
        Download, and unzip in the correct location.
        Returns:

        """
        import urllib
        import zipfile
        # download files
        try:
            os.makedirs(root)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(root, filename)
            ext = os.path.splitext(file_path)[1]
            with open(file_path, 'wb') as f:
                f.write(data.read())
            if ext == '.zip':
                with zipfile.ZipFile(file_path) as zip_f:
                    zip_f.extractall(root)
                os.unlink(file_path)

        print('Done!')

    def prepare(self, root, standardize=False, **select):
        """

        Args:
            **select:

        Returns:

        """
        datafile, labelfile = self.files(**select)
        data_filepath = os.path.join(root, datafile)
        label_filepath = os.path.join(root, labelfile)
        data = []
        target = []
        with open(data_filepath) as data_f, open(label_filepath) as label_f:
            for x, y in zip(data_f, it.islice(label_f, self.sync_files, None)):
                data.append(list(map(int, x.split())))
                target.append(int(y))
        data = torch.Tensor(data)
        target = torch.Tensor(target, dtype=torch.int32)

        if standardize:
            data_mean = data.mean(dim=0, keepdim=True)
            data_std = data.std(dim=0, keepdim=True)
            data = (data - data_mean) / data_std

        return TensorDataset(data, target)

    def files(self, **select):
        return '', ''


class G2(_SmallDataset):
    """Download and use G2 dataset.

    Select arguments
    ----------------
    dim : int
       Dimension of the input space
    sd : int
       Standard deviation of the Gaussian used to generate the 2 modes

    See possible values below.

    -----------------------------------------------------------------------
    G2 datasets creation
    -----------------------------------------------------------------------

    The datasets include two Gaussian normal distributions:

    Dataset name:    G2-dim-sd
    Centroid 1:      [500,500, ...]
    Centroid 2:      [600,600, ...]
    Dimensions:      dim = 1,2,4,8,16, ... 1024
    St.Dev:          sd  = 10,20,30,40 ... 100

    They have been created using the following C-language code:

    Calculate random value in (0,1]:

    U = (double)(rand()+1)/(double)(RAND_MAX+1);
    V = (double)(rand()+1)/(double)(RAND_MAX+1);

    Box-Muller method to create two independent standard
    one-dimensional Gaussian samples:

    X = sqrt(-2*log(U))*cos(2*3.14159*V);  /* pi = 3.14159 */
    Y = sqrt(-2*log(U))*sin(2*3.14159*V);

    Adjust mean and deviation:

    X_final = 500 + s * X;    /* mean + deviation * X */
    Y_final = 600 + s * Y;

    The points are stored in the files so that:
    - First 1024 points are from the cluster 1
    - Rest  1024 points are from the cluster 2

    -----------------------------------------------------------------------

    P. Fränti R. Mariescu-Istodor and C. Zhong, "XNN graph"
    IAPR Joint Int. Workshop on Structural, Syntactic, and Statistical Pattern
    Recognition Merida, Mexico, LNCS 10029, 207-217, November 2016.
    """

    urls = ["http://cs.joensuu.fi/sipu/datasets/g2-txt.zip"]

    def prepare(self, root, standardize=False, dim=1, sd=10):
        """
        Make torch Tensors from g2-`dim`-`sd` and infer labels.
        Args:
            dim:
            sd:

        Returns:

        """
        filename = 'g2-{}-{}.txt'.format(dim, sd)
        data = []
        target = []
        with open(os.path.join(root, filename)) as in_f:
            for i, line in enumerate(in_f):
                a, b = list(map(int, line.split())), 0 if i < 1024 else 1
                data.append(a)
                target.append(b)
        data = torch.Tensor(data)
        target = torch.Tensor(target, dtype=torch.int32)

        if standardize:
            data = (data - 550) / 50

        return TensorDataset(data, target)

    def check_exists(self, root):
        return os.path.exists(os.path.join(root, 'g2-1-10.txt'))


class S_set(_SmallDataset):
    """Download and use S-sets dataset.

    Synthetic 2-d data with N=5000 vectors and k=15 Gaussian clusters
    with different degree of cluster overlapping.

    Select Arguments
    ----------------
    num : int
       Higher `num` means, higher chance of overlapping between the modes.
       Choose: [1, 2, 3, 4]

    P. Fränti and O. Virmajoki,
    "Iterative shrinking method for clustering problems",
    Pattern Recognition, 39 (5), 761-765, May 2006.
    """

    urls = [
        "http://cs.joensuu.fi/sipu/datasets/s1.txt",
        "http://cs.joensuu.fi/sipu/datasets/s2.txt",
        "http://cs.joensuu.fi/sipu/datasets/s3.txt",
        "http://cs.joensuu.fi/sipu/datasets/s4.txt",
        "http://cs.joensuu.fi/sipu/datasets/s-originals.zip"
    ]

    sync_files = 5

    def files(self, num=1):
        """
        Make torch Tensors from 's{num}.txt' and fetch labels.
        Args:
            num:

        Returns:

        """
        return 's{}.txt'.format(num), 's{}-label.pa'.format(num)

    def check_exists(self, root):
        return os.path.exists(os.path.join(root, 's1.txt'))


class A_set(_SmallDataset):
    """Download and use A-sets dataset.

    Synthetic 2-d data with varying number of vectors (N) and clusters (k).
    There are 150 vectors per cluster.

    Select Arguments
    ----------------
    num : int
       Higher `num` means, higher chance of overlapping between the modes.
       Choose: [1, 2, 3]

    A1: N=3000, k=20
    A2: N=5250, k=35
    A3: N=7500, k=50

    I. Kärkkäinen and P. Fränti,
    "Dynamic local search algorithm for the clustering problem",
    Research Report A-2002-6
    """

    urls = [
        "http://cs.joensuu.fi/sipu/datasets/a1.txt",
        "http://cs.joensuu.fi/sipu/datasets/a2.txt",
        "http://cs.joensuu.fi/sipu/datasets/a3.txt",
        "http://cs.joensuu.fi/sipu/datasets/a-gt-pa.zip"
    ]

    sync_files = 4

    def files(self, num=1):
        """

        Args:
            num:

        Returns:

        """
        return 'a{}.txt'.format(num), 'a{}-ga.pa'.format(num)

    def check_exists(self, root):
        """

        Returns:

        """
        return os.path.exists(os.path.join(root, 'a1.txt'))


class DIM_set(_SmallDataset):
    """Download and use the (high) DIM-sets dataset.

    High-dimensional data sets N=1024 and k=16 Gaussian clusters.

    Select Arguments
    ----------------
    dim : int
       Dimension of the input space. Choose: [32, 64, 128, 256, 512, 1024]

    P. Fränti, O. Virmajoki and V. Hautamäki,
    "Fast agglomerative clustering using a k-nearest neighbor graph",
    IEEE Trans. on Pattern Analysis and Machine Intelligence, 28 (11),
    1875-1881, November 2006.
    """

    urls = [
        "http://cs.joensuu.fi/sipu/datasets/dim032.txt",
        "http://cs.joensuu.fi/sipu/datasets/dim032.pa",
        "http://cs.joensuu.fi/sipu/datasets/dim064.txt",
        "http://cs.joensuu.fi/sipu/datasets/dim064.pa",
        "http://cs.joensuu.fi/sipu/datasets/dim128.txt",
        "http://cs.joensuu.fi/sipu/datasets/dim128.pa",
        "http://cs.joensuu.fi/sipu/datasets/dim256.txt",
        "http://cs.joensuu.fi/sipu/datasets/dim256.pa",
        "http://cs.joensuu.fi/sipu/datasets/dim512.txt",
        "http://cs.joensuu.fi/sipu/datasets/dim512.pa",
        "http://cs.joensuu.fi/sipu/datasets/dim1024.txt",
        "http://cs.joensuu.fi/sipu/datasets/dim1024.pa",
    ]

    sync_files = 5

    def files(self, dim=None):
        """

        Args:
            dim:

        Returns:

        """
        return 'dim{:03d}.txt'.format(dim), 'dim{:03d}.pa'.format(dim)

    def check_exists(self, root):
        """

        Returns:

        """
        return os.path.exists(os.path.join(root, 'dim032.txt'))


class Unbalance(_SmallDataset):
    """Download and use the Unbalance dataset.

    Synthetic 2-d data with N=6500 vectors and k=8 Gaussian clusters

    There are 3 "dense" clusters of 2000 vectors each and
    5 "sparse" clusters of 100 vectors each.

    M. Rezaei and P. Fränti,
    "Set-matching methods for external cluster validity",
    IEEE Trans. on Knowledge and Data Engineering, 28 (8), 2173-2186,
    August 2016.
    """

    urls = [
        "http://cs.joensuu.fi/sipu/datasets/unbalance.txt",
        "http://cs.joensuu.fi/sipu/datasets/unbalance-gt-pa.zip",
    ]

    sync_files = 4

    def files(self):
        """

        Returns:

        """
        return 'unbalance.txt', 'unbalance-gt.pa'

    def check_exists(self, root):
        """

        Returns:

        """
        data, labels = self.files()
        return os.path.exists(os.path.join(root, data)) and\
            os.path.exists(os.path.join(root, labels))


class _Shapes(_SmallDataset):
    """Wrap shapes datasets from the website."""

    def prepare(self, root, standardize=False):
        """
        Make torch Tensors from data and label files.
        Returns:

        """
        datafile = self.urls[0].rpartition('/')[2]
        data_filepath = os.path.join(root, datafile)
        data = []
        target = []
        with open(data_filepath) as data_f:
            for sample in data_f:
                x, y, label = tuple(map(float, sample.split()))
                data.append([x, y])
                target.append(int(label) - 1)
        data = torch.Tensor(data)
        target = torch.Tensor(target, dtype=torch.int32)

        if standardize:
            data_mean = data.mean(dim=0, keepdim=True)
            data_std = data.std(dim=0, keepdim=True)
            data = (data - data_mean) / data_std

        return TensorDataset(data, target)

    def check_exists(self, root):
        """

        Returns:

        """
        datafile = self.urls[0].rpartition('/')[2]
        return os.path.exists(os.path.join(root, datafile))


class Aggregation(_Shapes):
    """Download and use the Aggregation dataset.

    N=788, k=7, D=2

    A. Gionis, H. Mannila, and P. Tsaparas,
    Clustering aggregation.
    ACM Transactions on Knowledge Discovery from Data (TKDD), 2007.
    """

    urls = ["http://cs.joensuu.fi/sipu/datasets/Aggregation.txt"]


class Compound(_Shapes):
    """Download and use the Compound dataset.

    N=399, k=6, D=2

    C.T. Zahn,
    Graph-theoretical methods for detecting and describing gestalt clusters.
    IEEE Transactions on Computers, 1971.
    """

    urls = ["http://cs.joensuu.fi/sipu/datasets/Compound.txt"]


class Pathbased(_Shapes):
    """Download and use the Pathbased dataset.

    N=300, k=3, D=2

    H. Chang and D.Y. Yeung,
    Robust path-based spectral clustering.
    Pattern Recognition, 2008.
    """

    urls = ["http://cs.joensuu.fi/sipu/datasets/pathbased.txt"]


class Spiral(_Shapes):
    """Download and use the Spiral dataset.

    N=312, k=3, D=2

    H. Chang and D.Y. Yeung,
    Robust path-based spectral clustering.
    Pattern Recognition, 2008.
    """

    urls = ["http://cs.joensuu.fi/sipu/datasets/spiral.txt"]


class D31(_Shapes):
    """Download and use the D31 dataset.

    N=3100, k=31, D=2

    C.J. Veenman, M.J.T. Reinders, and E. Backer,
    A maximum variance cluster algorithm.
    IEEE Trans. Pattern Analysis and Machine Intelligence 2002.
    """

    urls = ["http://cs.joensuu.fi/sipu/datasets/D31.txt"]


class R15(_Shapes):
    """Download and use the R15 dataset.

    N=600, k=15, D=2

    C.J. Veenman, M.J.T. Reinders, and E. Backer,
    A maximum variance cluster algorithm.
    IEEE Trans. Pattern Analysis and Machine Intelligence 2002.
    """

    urls = ["http://cs.joensuu.fi/sipu/datasets/R15.txt"]


class Jain(_Shapes):
    """Download and use the ORIGINAL (2 moons) Jain dataset.

    N=373, k=2, D=2

    A. Jain and M. Law,
    Data clustering: A user's dilemma.
    Lecture Notes in Computer Science, 2005.
    """

    urls = ["http://cs.joensuu.fi/sipu/datasets/jain.txt"]


class Flame(_Shapes):
    """Download and use the Flame dataset.

    N=240, k=2, D=2

    L. Fu and E. Medico,
    FLAME, a novel fuzzy clustering method for the analysis of DNA microarray
    data. BMC bioinformatics, 2007.
    """

    urls = ["http://cs.joensuu.fi/sipu/datasets/flame.txt"]
