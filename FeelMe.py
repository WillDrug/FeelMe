# MAIN IMPORTS
from __future__ import print_function
import sys
from os import listdir
from os.path import isfile, join

import numpy as np
import pickle
import FeatureExtractor as FE

# END AUX IMPORTS

#DEF FUNCTIONS
from datetime import datetime
startTime = datetime.now()

#BASE VARS
MUSIC_PATH = 'testSet'
PICKLE_PATH = 'pickles'
LOADBY = 'music_plus'
K_PROC = 'trainK'
SAVE_K = 'True'
POST_PROC = 'create_playlists'
REC = False

#PARSE COMMAND LINE
import argparse
pa = argparse.ArgumentParser()
pa.add_argument('--loadby')
pa.add_argument('--musicpath')
pa.add_argument('--picklepath')
pa.add_argument('--Kpath')
pa.add_argument('--cache')
pa.add_argument('--features')
pa.add_argument('--K_proc')
pa.add_argument('--saveK')
pa.add_argument('--post_proc')
pa.add_argument('--r')
args = pa.parse_args()

#POPULATE VARIABLES
loadby = getattr(args, 'loadby')
if loadby is None:
    loadby = LOADBY #music, pickles, music_plus, single_classify, features

musicpath = getattr(args, 'musicpath')
if musicpath is None:
    musicpath = MUSIC_PATH

picklepath = getattr(args, 'picklepath')
if picklepath is None:
    picklepath = PICKLE_PATH

Kpath = getattr(args, 'Kpath')

cache = getattr(args, 'cache')
if cache == 'True':
    cache = True
else:
    cache = False

feature_file = getattr(args, 'features')
if feature_file is None:
    feature_file = 'features.csv'

K_proc = getattr(args, 'K_proc') #preload
if K_proc is None:
    K_proc = K_PROC

save_K = getattr(args, 'saveK')
if save_K is 'True':
    save_K = True
else:
    save_K = False

post_proc = getattr(args, 'post_proc')
if post_proc is None:
    post_proc = POST_PROC

rec = getattr(args, 'r')
if rec == 'True':
    rec = True
else:
    rec = REC

# GET MAIN AUDIO LIST
if loadby in ['music', 'pickles', 'music_plus']:
    inlabels = [f for f in listdir(musicpath) if isfile(join(musicpath, f))]
    inmusic = [musicpath + '\\' + f for f in listdir(musicpath) if isfile(join(musicpath, f))]
    fts = FE.get_features_from_list(musicpath, picklepath, inlabels, loadby, cache)
elif loadby == 'single_classify':
    if Kpath is None:
        print('K-centers need to be specified when classifying one track')
        exit(1)
    print('Stub for single classify')
elif loadby == 'features':
    fts = FE.load_features(musicpath)
else:
    print('Specify load type')
    exit(2)


if K_proc == 'loadK':
    print('Stub for loading K')
elif K_proc == 'trainK':
    print('Stub for training K')
else:
    print('Specify K type')

if save_K is True:
    print('Stub for saving K')

if post_proc == 'create_playlists':
    print('Stub for playlist generation')
elif post_proc == 'save target':
    print('Stub for Y saving')
else:
    print('Stub for command console for classyfication by shit')



print("Process {} at: {}".format(__name__, datetime.now() - startTime))