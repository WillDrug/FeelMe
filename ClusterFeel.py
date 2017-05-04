import numpy as np
from sklearn.cluster import KMeans

PLAYLIST_COUNT = 2
N_INIT = 10
MAX_ITER = 1000
TOL = 1e-4
FE_SOURCE = 'features.fmf'
K_SOURCE = 'centroids.fmc'


#K = KMeans(n_clusters=PLAYLIST_COUNT, init='k-means++', n_init=N_INIT, max_iter=MAX_ITER, tol=TOL, precompute_distances='auto', verbose=0, copy_x=True, n_jobs=-1)
#model = K.fit_predict(data)

import argparse
import sys
if sys.argv[0]=='ClusterFeel.py':
   pa = argparse.ArgumentParser()
   pa.add_argument('--listnum', help='Number of playlists to generate')
   pa.add_argument('--n_init', help='Number of cluster initializations')
   pa.add_argument('--max_iter', help='Maximum number of iterations')
   pa.add_argument('--tol', help='Relative tolerance to declare conversion')
   pa.add_argument('--fesource', help='Feature source: filename to load features')
   pa.add_argument('--ksource', help='Cluster source: filename to load centroids')
   pa.add_argument('--act', help='Action: 0 predicts by loaded centroids, 1 fits and predicts')
   args = pa.parse_args()
   if getattr(args, 'listnum') is not None:
       PLAYLIST_COUNT = getattr(args, 'listnum')
   if getattr(args, 'n_init') is not None:
       N_INIT = getattr(args, 'n_init')
   if getattr(args, 'max_iter') is not None:
       MAX_ITER = getattr(args, 'max_iter')
   if getattr(args, 'tol') is not None:
       TOL = getattr(args, 'tol')
   if getattr(args, 'fesource') is not None:
       FE_SOURCE = getattr(args, 'fesource')
   if getattr(args, 'ksource') is not None:
       K_SOURCE = getattr(args, 'ksource')
   if getattr(args, 'act') is not None:
       act = int(getattr(args, 'act'))
   else:
       print('If called separetely, ClusterFeel.py requires --act parameter')
       exit(1)
   if act == 0:
       print('Stub for prediction')
   else:
       print('Stub for loading')