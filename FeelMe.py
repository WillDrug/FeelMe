# MAIN IMPORTS
from __future__ import print_function


import numpy as np
import pickle
import FeatureExtractor as FE

# END AUX IMPORTS

#DEF FUNCTIONS
from datetime import datetime
startTime = datetime.now()


#PARSE COMMAND LINE
import argparse
pa = argparse.ArgumentParser()
# FE FEATURES
pa.add_argument('--cachepath', help='Path to cache to and look-up from')
pa.add_argument('--musicroot', help='Root folder with music')
pa.add_argument('--recursive', help='Parse subfolders')
pa.add_argument('--cache', help='If y music will be cached to --cachepath')
pa.add_argument('--ffile', help='Feature file to save features to. Sort of.')
pa.add_argument('--extractor', help='if none music will be parsed, if filled - paramter will be used as file to try and load features')


# K-MEANS FEATURES
pa.add_argument('--listnum', help='Number of playlists to generate')
pa.add_argument('--n_init', help='Number of cluster initializations')
pa.add_argument('--max_iter', help='Maximum number of iterations')
pa.add_argument('--tol', help='Relative tolerance to declare conversion')
pa.add_argument('--ksource', help='Cluster source: filename to load centroids')
pa.add_argument('--act', help='Action: 0 predicts by loaded centroids, 1 fits and predicts')

# POST_PROCESSING FEATURES

args = pa.parse_args()

# FE processing
if getattr(args, 'cachepath') is not None:
    FE.PICKLE_PATH = getattr(args, 'cacheto')
if getattr(args, 'musicroot'):
    FE.MUSIC_PATH = getattr(args, 'musicroot')
if getattr(args, 'recursive') == 'y':
    FE.RECURSIVE = True
else:
    FE.RECURSIVE = False
if getattr(args, 'cache') == 'y':
    FE.CACHE = True
else:
    FE.CACHE = False
if getattr(args, 'ffile') is not None:
    FE.FFILE = getattr(args, 'ffile')

act = getattr(args, 'extractor')
if act is not None:
    try:
        labels, fts = FE.load_features()
    except FileNotFoundError:
        print("> File not found {}".format(FE.FFILE))
        exit(1)
else:
    fts, labels = get_features_from_list(create_list())


print("Process {} at: {}".format(__name__, datetime.now() - startTime))