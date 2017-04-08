from __future__ import print_function

from datetime import datetime
from os import listdir
from os.path import isfile, join

import numpy as np

startTime = datetime.now()

# AUX IMPORTS
import librosa
from librosa import display
import matplotlib.pyplot as plt


# FOR DEBUG
import pickle

# END AUX IMPORTS

# GET MAIN AUDIO LIST

musicpath = 'testSet'
inlabels = [f for f in listdir(musicpath) if isfile(join(musicpath, f))]
inmusic = [musicpath + '\\' + f for f in listdir(musicpath) if isfile(join(musicpath, f))]
# fts = FE.getFeaturesFromList(inmusic)


X = []
Y = []

for i, l in zip(inmusic, inlabels):
    try:
        y, sr = pickle.load(open('pickles/'+l+'.pic', 'rb+'))
    except FileNotFoundError:
        y, sr = librosa.load(i)
        pickle._dump((y, sr), open('pickles/'+l+'.pic', 'wb+'))

    y_harmonic, y_percussive = librosa.effects.hpss(y, margin=(1.0, 3.0))

    onset_env = librosa.onset.onset_strength(y, sr=sr, aggregate=np.median)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    print(l, tempo)

    # percussive*100/full ??
    # ONSET count / frame count = something like BPM?
    # find a way to compute WUBS
    # something from mean bands - for instance "L, M, H 0/1"

for label, x, y in zip(inlabels, X, Y):
            plt.annotate(
                label,
                xy=(x, y), xytext=(-20, 20),
                textcoords='offset points', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))


print("Process {} at: {}".format(__name__, datetime.now() - startTime))
plt.show()
