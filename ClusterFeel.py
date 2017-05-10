import numpy as np
import pickle as pic
from sklearn.cluster import KMeans
import argparse
import sys
import os


class ClusterFeel:
    @staticmethod
    def normalize_path(inpath):
        if not inpath:
            return ''
        if not inpath.__contains__(':'):
            if inpath[0] != '\\' and inpath[0] != '/':
                inpath = '\\' + inpath
            path = os.path.dirname(os.path.realpath('__file__')) + inpath
        else:
            path = inpath
        return path

    def __init__(self, playlist_count, n_init, max_iter, tol, k_source, ltarget, style, log):
        self.playlist_count = int(playlist_count)
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.k_source = self.normalize_path(k_source)
        self.ltarget = self.normalize_path(ltarget)
        self.style = style
        self.K = None
        self.labels = None
        self.log = log

    def create_cluster_obj(self):
        if self.log: print('> Clustering object created')
        if self.style == 'sklearn':
            self.K = KMeans(n_clusters=self.playlist_count, init='k-means++', n_init=self.n_init,
                            max_iter=self.max_iter, tol=self.tol, precompute_distances='auto', verbose=0, copy_x=True,
                            n_jobs=1)

    def cluster_fit(self, x):
        if self.log: print('> Fitting data')
        if self.style == 'sklearn':
            return self.K.fit(x)

    def cluster_predict(self, x):
        if self.log: print('> Predicting clusters')
        if self.style == 'sklearn':
            self.labels = self.K.predict(x)
            return self.labels

    def cluster_fit_predict(self, x):
        if self.log: print('> Fitting and predicting')
        if self.style == 'slearn':
            return self.K.fit_predict(x)

    def save_centroids(self):
        if self.log: print('> Saving centroids to {}'.format(self.k_source))
        if self.style == 'sklearn':
            pic.dump(self.K.cluster_centers_, open(self.k_source, 'wb+'))

    def load_centroids(self):
        if self.log: print('> Loading centroids from {}'.format(self.k_source))
        if self.style == 'sklearn':
            self.K.cluster_centers_ = pic.load(open(self.k_source, 'rb+'))

    def save_labels(self):
        if self.log: print('> Saving labels to {}'.format(self.ltarget))
        pic.dump(self.labels, open(self.ltarget, 'wb+'))

    def load_labels(self):
        if self.log: print('> Loading labels from {}'.format(self.ltarget))
        self.labels = pic.load(open(self.ltarget, 'rb+'))



if sys.argv[0] == 'ClusterFeel.py':
    pa = argparse.ArgumentParser()
    pa.add_argument('--listnum', help='Number of playlists to generate')
    pa.add_argument('--n_init', help='Number of cluster initializations')
    pa.add_argument('--max_iter', help='Maximum number of iterations')
    pa.add_argument('--tol', help='Relative tolerance to declare conversion')
    pa.add_argument('--fesource', help='Feature source: filename to load features')
    pa.add_argument('--ksource', help='Cluster source: filename to load centroids')
    pa.add_argument('--ltarget', help='Label target: save to...')
    pa.add_argument('--style', help='Library to cluster, default: sklearn, avail.: -')
    pa.add_argument('--act', help='Action: 0 loads centroids saves labels, '
                                  '1 fits data and saves labels,'
                                  '2 fits data and saves labels and centroids')
    args = pa.parse_args()
    if getattr(args, 'listnum') is not None:
        PLAYLIST_COUNT = getattr(args, 'listnum')
    else:
        PLAYLIST_COUNT = 2
    if getattr(args, 'n_init') is not None:
        N_INIT = getattr(args, 'n_init')
    else:
        N_INIT = 10
    if getattr(args, 'max_iter') is not None:
        MAX_ITER = getattr(args, 'max_iter')
    else:
        MAX_ITER = 500
    if getattr(args, 'tol') is not None:
        TOL = getattr(args, 'tol')
    else:
        TOL = 1e-16
    if getattr(args, 'fesource') is not None:
        FE_SOURCE = getattr(args, 'fesource')
    else:
        FE_SOURCE = 'features.fmf'
    if getattr(args, 'ksource') is not None:
        K_SOURCE = getattr(args, 'ksource')
    else:
        K_SOURCE = 'centroids.fmf'
    if getattr(args, 'ltarget') is not None:
        L_TARGET = getattr(args, 'ltarget')
    else:
        L_TARGET = 'labels.fmf'
    if getattr(args, 'style') is not None:
        STYLE = getattr(args, 'style')
    else:
        STYLE = 'sklearn'
    act = getattr(args, 'act')
    if getattr(args, 'act') is None:
        print('When called separately, ClusterFeel needs --act key;')
        exit(1)
    cluster = ClusterFeel(PLAYLIST_COUNT, N_INIT, MAX_ITER, TOL, K_SOURCE, L_TARGET, STYLE)
    import FeatureExtractor as FE

    FE.FFILE = FE_SOURCE
    labels, fts = FE.load_features()
    cluster.create_cluster_obj()
    if act == '0':
        cluster.load_centroids()
        cluster.cluster_predict(fts)
        cluster.save_labels()
    elif act == '1':
        cluster.cluster_fit(fts)
        cluster.cluster_predict(fts)
        cluster.save_labels()
    else:
        cluster.cluster_fit(fts)
        cluster.save_centroids()
        cluster.save_labels()
