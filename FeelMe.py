from __future__ import print_function
from os import listdir
from os.path import isfile, join
from datetime import datetime
startTime = datetime.now()

# AUX IMPORTS
import FeatureExtractor as FE

import numpy as np
import matplotlib.pyplot as plt

import librosa

# FOR DEBUG
import pickle
import soundfile as sf

# END AUX IMPORTS

# GET MAIN AUDIO LIST

musicpath = 'testSet'
print(listdir(musicpath))
exit()
inlabels = [f for f in listdir(musicpath) if isfile(join(musicpath, f))]
inmusic = [musicpath+'\\'+f for f in listdir(musicpath) if isfile(join(musicpath, f))]
#fts = FE.getFeaturesFromList(inmusic)

print(inlabels)
exit()
for i, l in zip(inmusic, inlabels):
    try:
        y, sr = pickle.load(open(l+'.pic', 'rb+'))
    except FileNotFoundError:
        y, sr = librosa.load(i)
        pickle._dump((y, sr), open(l+'.pic', 'wb+'))
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    onset_env = librosa.onset.onset_strength(y, sr=sr, aggregate = np.median)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr = sr)
    print(l, tempo)
    # percussive*100/full ??
    # ONSET count / frame count = something like BPM?
    # count of notes > np.mean of chroma cqt


print("Process {} at: {}".format(__name__, datetime.now() - startTime))
plt.show()