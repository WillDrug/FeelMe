from __future__ import print_function
import FeatureExtractor as FE
import numpy as np
from os import listdir
from os.path import isfile, join
from datetime import datetime
startTime = datetime.now()

# AUX IMPORTS
import librosa

import matplotlib.pyplot as plt
import matplotlib.style as ms
ms.use('seaborn-muted')
import IPython.display
import librosa.display

# FOR DEBUG
import pickle
import soundfile as sf

# END AUX IMPORTS

# GET MAIN AUDIO LIST

musicpath = 'testSet'
inlabels = [f for f in listdir(musicpath) if isfile(join(musicpath, f))]
inmusic = [musicpath+'\\'+f for f in listdir(musicpath) if isfile(join(musicpath, f))]
#fts = FE.getFeaturesFromList(inmusic)


for i, l in zip(inmusic, inlabels):
    y, sr = librosa.load(i)
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    #hop_length = 512
    #tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)

    cp = np.asarray(y_percussive).reshape(-1)
    cp = cp[cp.argsort()]

    c = np.asarray(y).reshape(-1)
    c = c[c.argsort()]

    print("{} : {}".format(l, np.max(c-cp)))

    # percussive*100/full ??
    # ONSET count / frame count = something like BPM?
    # count of notes > np.mean of chroma cqt
    # find a way to compute WUBS
    """
    plt.figure()
    plt.title(l)
    S = librosa.magphase(librosa.stft(y, window=np.ones, center=False))[0]
    rms = librosa.feature.rmse(S=S)
    plt.semilogy(rms.T, label='FULL')
    S = librosa.magphase(librosa.stft(y=y_percussive, window=np.ones, center=False))[0]
    rms = librosa.feature.rmse(S=S)
    plt.semilogy(rms.T, label='Percussive')
    S = librosa.magphase(librosa.stft(y=y_harmonic, window=np.ones, center=False))[0]
    rms = librosa.feature.rmse(S=S)
    plt.semilogy(rms.T, label='Harmonic')
    plt.legend()
    """

"""
#pickle._dump(fts, open('fts.txt', 'wb+'))
fts = pickle.load(open('fts.txt', 'rb+'))


# SCATTER FEATURES
ftss = fts[:, 0:3]
fcnt = ftss.shape[1]

f, axarr = plt.subplots(fcnt, fcnt)
for i in range(0, fcnt):
    for j in range(0, fcnt):
        if i==j: continue
        axarr[i, j].scatter(ftss[:, i], ftss[:, j])
        axarr[i, j].set_title("X: {}, Y: {}".format(i, j))
        for label, x, y in zip(inlabels, ftss[:, i], ftss[:, j]):
            axarr[i, j].annotate(
                label,
                xy=(x, y), xytext=(-20, 20),
                textcoords='offset points', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
f.subplots_adjust(hspace=1.0)
"""
print("Process {} at: {}".format(__name__, datetime.now() - startTime))
plt.show()