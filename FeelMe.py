# MAIN IMPORTS
from __future__ import print_function
import FeatureExtractor as fE
import ClusterFeel as cF
import PlaylistFeel as pF
# AUX IMPORTS
from datetime import datetime
import argparse
import numpy as np
import sys
# DEF FUNCTIONS


# PARSE COMMAND LINE

if sys.argv[0] == 'FeelMe.py' and __name__ == '__main__':
    startTime = datetime.now()
    pa = argparse.ArgumentParser()
    pa.add_argument('--verbose', help='Logging')
    # FE FEATURES
    pa.add_argument('--cachepath', help='Path to cache to and look-up from')
    pa.add_argument('--musicroot', help='Root folder with music')
    pa.add_argument('--recursive', help='Parse subfolders')
    pa.add_argument('--cache', help='If y music will be cached to --cachepath')
    pa.add_argument('--ffile', help='Feature file to save features to. Sort of.')
    pa.add_argument('--extractor',
                    help='if none music will be parsed, if filled - parameter will be used as file to try and load features'
                    )
    pa.add_argument('--multi', help='enable multiprocessing with value number of cores')

    # K-MEANS FEATURES
    pa.add_argument('--listnum', help='Number of playlists to generate')
    pa.add_argument('--n_init', help='Number of cluster initializations')
    pa.add_argument('--max_iter', help='Maximum number of iterations')
    pa.add_argument('--tol', help='Relative tolerance to declare conversion')
    pa.add_argument('--ksource', help='Cluster source: filename to load or save centroids')
    pa.add_argument('--style', help='Library to cluster, default: sklearn, avail.: -')
    pa.add_argument('--ltarget', help='Save labels here')
    pa.add_argument('--cfit', help='0 fit (default), 1 load')
    pa.add_argument('--csave', help='0 no save, 1 saves centroids')

    # PLAYLIST FEAUTRES
    pa.add_argument('--ploc', help='Playlist location')

    args = pa.parse_args()
    if getattr(args, 'verbose') is not None:
        if getattr(args, 'verbose') == 'y':
            LOG = True
        else:
            LOG = False
    else:
        LOG = True

    # FE processing
    if getattr(args, 'cachepath') is not None:
        fE.PICKLE_PATH = getattr(args, 'cachepath')
    if getattr(args, 'musicroot'):
        fE.MUSIC_PATH = getattr(args, 'musicroot')
    if getattr(args, 'recursive') == 'y':
        fE.RECURSIVE = True
    else:
        fE.RECURSIVE = False
    if getattr(args, 'cache') == 'y':
        fE.CACHE = True
    else:
        fE.CACHE = False
    if getattr(args, 'ffile') is not None:
        fE.FFILE = getattr(args, 'ffile')
    if getattr(args, 'multi') is not None:
        fE.MULTI = int(getattr(args, 'multi'))
    fE.LOG = True
    act = getattr(args, 'extractor')
    if act is not None:
        try:
            labels, fts = fE.load_features()
        except FileNotFoundError:
            print("> File not found {}".format(fE.FFILE))
            exit(1)
    else:
        if fE.MULTI > 0:
            fts, labels = fE.get_features_from_list_multi(fE.create_list())
        else:
            fts, labels = fE.get_features_from_list(fE.create_list())
        fE.save_features(fts, labels)

    # K-Means processing
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
    if getattr(args, 'ksource') is not None:
        K_SOURCE = getattr(args, 'ksource')
    else:
        K_SOURCE = 'centroids.fmf'
    if getattr(args, 'style') is not None:
        STYLE = getattr(args, 'style')
    else:
        STYLE = 'sklearn'
    if getattr(args, 'cfit') is not None:
        cfit = True
    else:
        cfit = False
    if getattr(args, 'csave') is not None:
        csave = True
    else:
        csave = False
    LTARGET = getattr(args, 'ltarget')

    cluster = cF.ClusterFeel(PLAYLIST_COUNT, N_INIT, MAX_ITER, TOL, K_SOURCE, LTARGET, STYLE, LOG)
    cluster.create_cluster_obj()
    if cfit:
        cluster.load_centroids()
        cluster.cluster_predict(fts)
    else:
        cluster.cluster_fit(fts)
        cluster.cluster_predict(fts)

    if csave:
        cluster.save_centroids()

    # PLAYLIST GEN
    pF.LOG = LOG
    ploc = getattr(args, 'ploc')
    if not ploc:
        ploc = '\\'

    pF.create_playlists_from_clusters(labels, cluster.labels, ploc)

    print("Process {} at: {}".format(__name__, datetime.now() - startTime))
