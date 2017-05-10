import sys
import numpy as np
import os
LOG = True
def create_playlists_from_clusters(filenames, target, saveto):
    if LOG: print('> Creating playlists')
    for i in np.unique(target):
        if LOG: print('> Generating playlist {}'.format(i))
        create_playlist(filenames[target==i], '{}.m3u'.format(i), saveto)

def create_playlist(filenames, playlistname, saveto):
    path = normalize_path(saveto)+'\\'+playlistname
    with open(path, 'w') as f:
        for q in filenames:
            f.write(q)
            f.write('\n')
    if LOG: print('> Playlist {} created'.format(playlistname))

def normalize_path(inpath):
    if not inpath.__contains__(':'):
        if inpath[0] != '\\' and inpath[0] != '/':
            inpath = '\\' + inpath
        path = os.path.dirname(os.path.realpath('__file__')) + inpath
    else:
        path = inpath
    return path




if sys.argv[0] == 'PlaylistFeel.py' and __name__ == '__main__':
    pa = argparse.ArgumentParser()
    pa.add_argument('--fsource', help='Path to features')
    pa.add_argument('--ksource', help='Path to centroids')
    pa.add_argument('--pcount', help='Count of playlists')
    pa.add_argument('--saveto', help='Where to save')
    args = pa.parse_args()
    fsource = getattr(args, 'fsource')
    ksource = getattr(args, 'ksource')
    pcount = getattr(args, 'pcount')
    saveto = getattr(args, 'saveto')
    if not saveto:
        saveto = '\\'
    import FeatureExtractor
    import ClusterFeel
    labels, fts = FeatureExtractor.load_features(fsource)
    cluster = ClusterFeel.ClusterFeel(pcount, 20, 500, 1e-16, ksource, '', 'sklearn')
    cluster.load_centroids()
    cluster.cluster_predict(fts)
    create_playlists_from_clusters(labels, cluster.labels, saveto)
