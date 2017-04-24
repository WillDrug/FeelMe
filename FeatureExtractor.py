# Librosa for audio
import librosa
# And the display module for visualization
import librosa.display
import numpy as np

# FEATURE INITIALIZATION

FEATURE_COUNT = int(1070)
FEATURE_NAMES = b'label, est.BPM.perc,est.BPM.harm,est.BPM.clean.perc,est.BPM.clean.harm,har.div.perc.full,'
for i in range(0, 1025):
    FEATURE_NAMES += b'Spectrogram.db.'+bytes("{},".format(i), encoding='utf-8')
for i in range(0, 6):
    FEATURE_NAMES += b'tonnetz.'+bytes("{},".format(i), encoding='utf-8')
for i in range(0, 12):
    FEATURE_NAMES += b'CCQT.'+bytes("{},".format(i), encoding='utf-8')
for i in range(0, 20):
    FEATURE_NAMES += b'MFCC.'+bytes("{},".format(i), encoding='utf-8')
FEATURE_NAMES += b'zerocross.median'
FEATURE_NAMES += b'\n'

def normalize(S):
    return (S-np.min(S))/(np.max(S)-np.min(S))

def fullNorm(S):
    for col in range(S.shape[1]):
        S[:, col] = normalize(S[:, col])
    return S

def get_features_from_list(musicpath, picklepath, labels, mode, cache=False):
    ftlist = np.zeros((labels.__len__(), FEATURE_COUNT))

    for row, label in zip(range(0, labels.__len__()), labels):
        ftlist[row, :] = get_features_from_name(musicpath, picklepath, label, mode, cache)
    return ftlist

def get_features_from_name(musicpath, picklepath, label, mode, cache=False):
    if mode == 'music_plus':
        try:
            y, sr = librosa.load(musicpath + '\\' + label)
        except FileNotFoundError:
            exit(2)
    elif mode == 'picled':
        try:
            y, sr = pickle.load(open(picklepath + '\\' + label + '.pic', 'rb+'))
        except FileNotFoundError:
            exit(2)
    else:
        try:
            y, sr = pickle.load(open(picklepath + '\\' + label + '.pic', 'rb+'))
        except FileNotFoundError:
            y, sr = librosa.load(musicpath + '\\' + label)
            if cache:
                pickle._dump((y, sr), open(picklepath + '\\' + label + '.pic', 'wb+'))

    return getFeatures(y, sr, label)

def getFeatures(y, sr, label):
    # base vars
    fts = np.zeros((1, FEATURE_COUNT))

    # signal processing
    y_h, y_p = librosa.effects.hpss(y)
    y_h_clean, y_p_clean = librosa.effects.hpss(y, margin=(20, 20))


    # feature computation
    onset_p = librosa.onset.onset_detect(y=y_p, sr=sr, units='time')
    onset_p_d = np.diff(onset_p)
    onset_h = librosa.onset.onset_detect(y=y_h, sr=sr, units='time')
    onset_h_d = np.diff(onset_h)

    onset_p_clean = librosa.onset.onset_detect(y=y_p, sr=sr, units='time')
    onset_p_clean_d = np.diff(onset_p_clean)
    onset_h_clean = librosa.onset.onset_detect(y=y_h, sr=sr, units='time')
    onset_h_clean_d = np.diff(onset_h_clean)

    tmp = librosa.power_to_db(np.abs(librosa.stft(y_h_clean)) ** 2, ref=np.min)
    ShCdb = np.sum(tmp, axis=1) / tmp.shape[1]
    tmp = librosa.power_to_db(np.abs(librosa.stft(y_p_clean)) ** 2, ref=np.min)
    SpCdb = np.sum(tmp, axis=1) / tmp.shape[1]

    SfCdb = librosa.power_to_db(np.abs(librosa.stft(y_p_clean + y_h_clean)) ** 2, ref=np.min)

    tonnetz_full = librosa.feature.tonnetz(y=y, sr=sr)

    ccqt = librosa.feature.chroma_cqt(y=y, sr=22050)

    mfcc = librosa.feature.mfcc(y=y, sr=22050, S=None, n_mfcc=20)


    # feature mapping
    fts[0, 0] = label
    fts[0, 1] = 60/np.mean(onset_p_d)
    fts[0, 2] = 60/np.mean(onset_h_d)
    fts[0, 3] = 60/np.mean(onset_p_clean_d)
    fts[0, 4] = 60/np.mean(onset_h_clean_d)
    fts[0, 5] = np.sum(SpCdb)/np.sum(ShCdb)
    fts[0, 6:1031] = np.sum(SfCdb, axis=1) / tmp.shape[1]
    fts[0, 1031:1037] = np.sum(tonnetz_full, axis=1)/tonnetz_full.shape[1]
    fts[0, 1038:1049] = normalize(np.sum(ccqt, axis=1))
    fts[0, 1050:1070] = np.sum(mfcc, axis=1)
    fts[0, 1071] = np.median(librosa.feature.zero_crossing_rate(y=y, frame_length=2048, hop_length=512, center=True))
    return fts

def save_features(fts, fn='features.txt'):
    try:
        with open(fn, 'wb') as f:
            f.write(FEATURE_NAMES)
            np.savetxt(f, fts, delimiter=",")
        return True
    except Exception:
        return False
def load_features(fname):
    print('Stub for loading features from file')