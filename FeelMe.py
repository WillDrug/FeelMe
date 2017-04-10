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

    #y_harmonic, y_percussive = librosa.effects.hpss(y, margin=(1.0, 3.0))

    onset_envelope = librosa.onset.onset_strength(y=y, sr=sr, S=None, lag=1, max_size=1, detrend=False, center=True, feature=None, aggregate=None, centering=None)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_envelope, sr=sr)
    times = librosa.frames_to_time(np.arange(len(onset_envelope)), sr=sr)

    plt.figure()
    plt.title(l)
    plt.plot(times, onset_envelope, label='Onset strength')
    plt.vlines(times[onset_frames], 0, onset_envelope.max(), color='r', alpha=0.9, linestyle='--', label='Onsets')
    plt.show()
    exit()
    #X.append(1)
    #Y.append(tempo)
    print(l)


    # percussive*100/full ??
    # ONSET count / frame count = something like BPM?
    # find a way to compute WUBS
    # something from mean bands - for instance "L, M, H 0/1"
"""
plt.scatter(X, Y)
for label, x, y in zip(inlabels, X, Y):
            plt.annotate(
                label,
                xy=(x, y), xytext=(-20, 20),
                textcoords='offset points', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

"""
print("Process {} at: {}".format(__name__, datetime.now() - startTime))
plt.show()