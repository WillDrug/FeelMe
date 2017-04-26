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
act = getattr(args, 'do')
    if act == 'save':
        fts, labels = get_features_from_list(create_list())
        fls = save_features(fts, labels)
    elif act == 'cache':
        for file in create_list():
            load_signal(file, normalize_path(PICKLE_PATH), True)
    elif act == 'load':
        fts, labels = load_features(FFILE)
# K-MEANS FEATURES

# POST_PROCESSING FEATURES

args = pa.parse_args()
if getattr(args, 'cachepath') is not None:
    FE.PICKLE_PATH = getattr(args, 'cacheto')
if getattr(args, 'musicroot'):
    FE.MUSIC_PATH = getattr(args, 'musicroot')
if getattr(args, 'recursive') == 'y':
    FE.RECURSIVE = True
else:
    FE.RECURSIVE = False
if getattr(args, 'cache') == 'n':
    FE.CACHE = False
else:
    FE.CACHE = True

print(FE.create_list())

print("Process {} at: {}".format(__name__, datetime.now() - startTime))