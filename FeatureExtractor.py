# Librosa for audio
import librosa
from audioread import NoBackendError

# And the display module for visualization
import librosa.display
import numpy as np
import pickle
import sys
import os

LOG = True
import progressbar

# FEATURE INITIALIZATION

FEATURE_COUNT = int(1073)
FEATURE_NAMES = b'#label,est.BPM.perc,est.BPM.harm,est.BPM.clean.perc,est.BPM.clean.harm,har.div.perc.full,'
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


# VARIABLE INITIALIZATION
PICKLE_PATH = '\\pickles'
MUSIC_PATH = '\\testSet'
RECURSIVE = False
CACHE = False
FFILE = 'features.fmf'

def normalize_path(PATH):
    if not PATH.__contains__(':'):
        if PATH[0] != '\\' and PATH[0] != '/':
            PATH = '\\' + PATH
        path = os.path.dirname(os.path.realpath('__file__')) + PATH
    else:
        path = PATH
    return path
def create_list():
    if LOG: print('> Building file list...')
    fls = []
    musicpath = normalize_path(MUSIC_PATH)
    if RECURSIVE:
        for root, dirs, files in os.walk(musicpath):
            for file in files:
                files[files.index(file)] = root + '\\' + file
            fls.extend(files)
    else:
        onlyfiles = [f for f in os.listdir(musicpath) if os.path.isfile(os.path.join(musicpath, f))]
        for file in onlyfiles:
            onlyfiles[onlyfiles.index(file)] = musicpath + '\\' + file
        fls.extend(onlyfiles)
    if LOG: print('> File list built!')
    return fls

def normalize(S):
    return (S-np.min(S))/(np.max(S)-np.min(S))

def full_norm(S):
    for col in range(S.shape[1]):
        S[:, col] = normalize(S[:, col])
    return S

def get_features_from_list(fls):
    if LOG: bar = progressbar.ProgressBar(max_value=fls.__len__())
    ftlist = np.ndarray((0, FEATURE_COUNT))
    labels = np.ndarray((0, 1))
    for fl, barcount in zip(fls, range(1, fls.__len__()+1)):
        ft, label = load_and_extract(fl)
        if ft is not None:
            ftlist = np.vstack((ftlist, ft))
            labels = np.vstack((labels, label))
        if LOG: bar.update(barcount)
    return ftlist, labels

def load_signal(f, picklepath, cache=False):
    try:
        y, sr = librosa.load(f)
        if cache:
            pickle._dump((y, sr), open(picklepath + '\\' + os.path.basename(f) + '.pic', 'wb+'))
        return y, sr
    except NoBackendError:
        print("> NoBackEndError for {}".format(os.path.basename(f)))
        pass
    except EOFError:
        print("> EOFError for {}".format(os.path.basename(f)))
        pass
    return False, False
def load_and_extract(fl):
    picklepath = normalize_path(PICKLE_PATH)
    nopickle = False
    try:
        y, sr = pickle.load(open(picklepath + '\\' + os.path.basename(fl) + '.pic', 'rb+'))
    except FileNotFoundError:
        nopickle = True
        pass

    if nopickle:
        if LOG: print("> FileNotFoundError for {}".format(os.path.basename(fl)))
        y, sr = load_signal(fl, picklepath, CACHE)
        if y==False:
            return None, None
        if LOG: print("> Handled, loaded {}".format(os.path.basename(fl)))

    return extract_features(y, sr, fl)

def extract_features(y, sr, label):
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
    fts[0, 0] = 60/np.mean(onset_p_d)
    fts[0, 1] = 60/np.mean(onset_h_d)
    fts[0, 2] = 60/np.mean(onset_p_clean_d)
    fts[0, 3] = 60/np.mean(onset_h_clean_d)
    fts[0, 4] = np.sum(SpCdb)/np.sum(ShCdb)
    fts[0, 5:1030] = np.sum(SfCdb, axis=1) / tmp.shape[1]
    fts[0, 1031:1037] = np.sum(tonnetz_full, axis=1)/tonnetz_full.shape[1]
    fts[0, 1038:1050] = normalize(np.sum(ccqt, axis=1))
    fts[0, 1051:1071] = np.sum(mfcc, axis=1)
    fts[0, 1072] = np.median(librosa.feature.zero_crossing_rate(y=y, frame_length=2048, hop_length=512, center=True))

    return fts, label

def save_features(fts, labels):
    try:
        with open(FFILE, 'wb') as f:
            f.write(FEATURE_NAMES)
            np.savetxt(f, np.c_[labels, fts], delimiter=",", fmt="%s")
        return True
    except Exception as e:
        print('> Exception occurred while saving stuff: {}'.format(e))
        return False
def load_features():
    labels = np.loadtxt(open(FFILE, 'rb'), dtype=str, comments='#', delimiter=',', skiprows=0, usecols=np.arange(0, 1))
    fts = np.atleast_2d(np.loadtxt(open(FFILE, 'rb'), dtype=float, comments='#', delimiter=',', skiprows=0, usecols=np.arange(1, FEATURE_COUNT+1)))
    return labels, fts


# COMM LINE PARMS
import argparse
import sys
if sys.argv[0]=='FeatureExtractor.py':
    pa = argparse.ArgumentParser()
    pa.add_argument('--cachepath', help='Path to cache to and look-up from')
    pa.add_argument('--musicroot', help='Root folder with music')
    pa.add_argument('--recursive', help='Parse subfolders')
    pa.add_argument('--cache', help='If y music will be cached to --cachepath')
    pa.add_argument('--do', help='Specify action. save - parses music and saves features. \ncache - parses music and just pickles it. \n')
    pa.add_argument('--ffile', help='Feature file to save features to. Sort of.')
    args = pa.parse_args()
    if getattr(args, 'cachepath') is not None:
        PICKLE_PATH = getattr(args, 'cacheto')
    if getattr(args, 'musicroot'):
        MUSIC_PATH = getattr(args, 'musicroot')
    if getattr(args, 'recursive') == 'y':
        RECURSIVE = True
    else:
        RECURSIVE = False
    if getattr(args, 'cache') == 'y':
        CACHE = True
    else:
        CACHE = False
    if getattr(args, 'ffile') is not None:
        FFILE = getattr(args, 'ffile')
    act = getattr(args, 'do')
    if act == 'save':
        fts, labels = get_features_from_list(create_list())
        fls = save_features(fts, labels)
    elif act == 'cache':
        for file in create_list():
            load_signal(file, normalize_path(PICKLE_PATH), True)
    elif act == 'load':
        fts, labels = load_features(FFILE)