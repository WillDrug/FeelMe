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

# END AUX IMPORTS

# GET MAIN AUDIO LIST

musicpath = 'D:\Work\FeelMe\\testSet'
inlabels = [f for f in listdir(musicpath) if isfile(join(musicpath, f))]
inmusic = [musicpath+'\\'+f for f in listdir(musicpath) if isfile(join(musicpath, f))]
#fts = FE.getFeaturesFromList(inmusic)
sclabels = ["o","v","^","<",">"]


for i, l, m in zip(inmusic, inlabels, sclabels):
    y, sr = librosa.load(i)
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    hop_length = 512
    #S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    #log_S = librosa.logamplitude(S, ref_power=np.max)
    #contrast = librosa.feature.spectral_contrast(S=log_S, sr=sr)
    #avg_bands = np.mean(contrast, axis=1)
    onsets_harmonic = librosa.onset.onset_detect(y=y_harmonic,
                                                 sr=sr,
                                                 hop_length=hop_length)
    print(onsets_harmonic.shape)
    #plt.legend()
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