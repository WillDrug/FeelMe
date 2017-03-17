# Librosa for audio
import librosa
# And the display module for visualization
import librosa.display
import numpy as np

#from multiprocessing import Pool, cpu_count

FEATURE_COUNT = int(7)
FEATURE_NAMES = b'BPM, EQ1, EQ2, EQ3, EQ4, EQ5, EQ6, EQ7, TUNE\n'

def getFeaturesFromList(fls):
    ftlist = np.zeros((fls.__len__(), FEATURE_COUNT))
    """
    pool = Pool(processes=np.min((fls.__len__(), cpu_count())))
    result = []
    #for i in range(0, fls.__len__()):
    result.append(pool.map(getFeaturesFromName, fls))
    pool.close()
    pool.join()
    for i in zip(range(0, fls.__len__()), result):
        print(i[1])
        #ftlist[i[0], :] = i[1]
    print(ftlist)"""
    for i in range(0, fls.__len__()):
        ftlist[i, :] = getFeaturesFromName(fls[i])
    return ftlist

def getFeaturesFromName(fl):
    y, sr = librosa.load(fl)
    return getFeatures(y, sr)
def getFeatures(y, sr):
    # base vars
    fts = np.zeros((1, FEATURE_COUNT))
    hop_length = 512  # ~23ms hops

    # Calculate series
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
    # Convert to log scale (dB). Use the peak power as reference.
    log_S = librosa.logamplitude(S, ref_power=np.max)
    C = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)

    # Calculate features
    tempo, beats = librosa.beat.beat_track(y=y_percussive, sr=sr)
    tuning = librosa.estimate_tuning(y=y_harmonic, sr=sr)
    onsets_harmonic = librosa.onset.onset_detect(y=y_harmonic,
                                                 sr=sr,
                                                 hop_length=hop_length)
    contrast = librosa.feature.spectral_contrast(S=log_S, sr=sr)
    avg_bands = np.mean(contrast, axis=1)
    sum_notes = np.sum(C, axis=1) / C.shape[1] # 0.0-1.0 on ~10 notes intensity through the track. Usually very sharp graph at classical and very smooth and techno and trash.

    # Map features
    fts[0, 0] = tempo # calculated BPM
    fts[0, 1] = tuning # calculated tuning
    fts[0, 2] = onsets_harmonic.__len__() #
    fts[0, 3:15] = sum_notes # note intensities

    return fts
def saveFeatures(fts, fn='features.txt'):
    with open('final.csv', 'wb') as f:
        f.write(FEATURE_NAMES)
    # f.write(bytes("SP,"+lists+"\n","UTF-8"))
    # Used this line for a variable list of numbers
        numpy.savetxt(f, a, delimiter=",")