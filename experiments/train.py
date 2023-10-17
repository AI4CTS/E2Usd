import sys
import time
import numpy as np
import pandas as pd
sys.path.append('./')
from E2USD.e2usd import E2USD
from E2USD.adapers import *
from E2USD.utils import *
from E2USD.clustering import *
from E2USD.params import *

# Define global paths
script_path = os.path.dirname(__file__)
data_path = os.path.join(script_path, '../data/')
output_path = os.path.join(script_path, '../results/output_E2USD')


def create_path(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)

def exp_on_UCR_SEG(win_size, step, verbose=False):
    """Experiment on UCR_SEG dataset."""
    score_list = []
    out_path = os.path.join(output_path, 'UCR-SEG')
    create_path(out_path)
    params['in_channels'] = 1
    params['out_channels'] = 4
    params['compared_length'] = win_size
    params['kernel_size'] = 3


    dataset_path = os.path.join(data_path, 'UCRSEG/')
    for fname in os.listdir(dataset_path):
        info_list = fname[:-4].split('_')
        seg_info = {}
        i = 0
        for seg in info_list[2:]:
            seg_info[int(seg)] = i
            i += 1
        seg_info[len_of_file(dataset_path + fname)] = i
        df = pd.read_csv(dataset_path + fname)
        data = df.to_numpy()
        data = normalize(data)
        t2s = E2USD(win_size, step, E2USD_Adaper(params), DPGMM(None)).fit(data, win_size, step)
        groundtruth = seg_to_label(seg_info)[:-1]
        prediction = t2s.state_seq
        ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
        prediction = np.array(prediction, dtype=int)
        result = np.vstack([groundtruth, prediction])
        np.save(os.path.join(out_path, fname[:-4]), result)
        score_list.append(np.array([ari, anmi, nmi]))
        if verbose:
            print('ID: %s, ARI: %f, ANMI: %f, NMI: %f' % (fname, ari, anmi, nmi))
    score_list = np.vstack(score_list)
    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' % (np.mean(score_list[:, 0]) \
                                                       , np.mean(score_list[:, 1])
                                                   , np.mean(score_list[:, 2])))


def exp_on_MoCap(win_size, step, verbose=False):
    """Experiment on MoCap dataset."""
    base_path = os.path.join(data_path, 'MoCap/4d/')
    out_path = os.path.join(output_path, 'MoCap')
    create_path(out_path)
    score_list = []
    params['in_channels'] = 4
    params['compared_length'] = win_size
    params['out_channels'] = 4
    f_list = os.listdir(base_path)
    f_list.sort()
    for idx, fname in enumerate(f_list):
        dataset_path = base_path + fname
        df = pd.read_csv(dataset_path, sep=' ', usecols=range(0, 4))
        data = df.to_numpy()
        groundtruth = seg_to_label(dataset_info[fname]['label'])[:-1]
        t2s = E2USD(win_size, step, E2USD_Adaper(params), DPGMM(None)).fit(data, win_size, step)

        prediction = t2s.state_seq
        prediction = np.array(prediction, dtype=int)
        result = np.vstack([groundtruth, prediction])
        np.save(os.path.join(out_path, fname), result)
        ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
        score_list.append(np.array([ari, anmi, nmi]))
        if verbose:
            print('ID: %s, ARI: %f, ANMI: %f, NMI: %f' % (fname, ari, anmi, nmi))
    score_list = np.vstack(score_list)
    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' % (np.mean(score_list[:, 0]) \
                                                       , np.mean(score_list[:, 1])
                                                   , np.mean(score_list[:, 2])))


def exp_on_synthetic(win_size=512, step=100, verbose=False):
    """Experiment on Synthetic dataset."""

    out_path = os.path.join(output_path, 'synthetic')
    create_path(out_path)
    params['in_channels'] = 4
    params['compared_length'] = win_size
    params['out_channels'] = 4
    prefix = os.path.join(data_path, 'synthetic/test')

    score_list = []
    for i in range(100):
        df = pd.read_csv(prefix + str(i) + '.csv', usecols=range(4), skiprows=1)
        data = df.to_numpy()
        df = pd.read_csv(prefix + str(i) + '.csv', usecols=[4], skiprows=1)
        groundtruth = df.to_numpy(dtype=int).flatten()
        t2s = E2USD(win_size, step, E2USD_Adaper(params), DPGMM(None)).fit(data, win_size, step)
        prediction = t2s.state_seq
        prediction = np.array(prediction, dtype=int)
        result = np.vstack([groundtruth, prediction])
        np.save(os.path.join(out_path, str(i)), result)
        ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
        score_list.append(np.array([ari, anmi, nmi]))
        if verbose:
            print('ID: %d, ARI: %f, ANMI: %f, NMI: %f' % (i, ari, anmi, nmi))
    score_list = np.vstack(score_list)
    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' % (np.mean(score_list[:, 0]) \
                                                       , np.mean(score_list[:, 1])
                                                   , np.mean(score_list[:, 2])))


def exp_on_ActRecTut(win_size, step, verbose=False):
    """Experiment on ActRecTut dataset."""
    out_path = os.path.join(output_path, 'ActRecTut')
    create_path(out_path)
    params['in_channels'] = 10
    params['compared_length'] = win_size
    params['out_channels'] = 4
    score_list = []
    dir_list = ['subject1_walk', 'subject2_walk']
    for dir_name in dir_list:
        for i in range(10):
            dataset_path = os.path.join(data_path, 'ActRecTut/' + dir_name + '/data.mat')
            data = scipy.io.loadmat(dataset_path)
            groundtruth = data['labels'].flatten()
            groundtruth = reorder_label(groundtruth)
            data = data['data'][:, 0:10]
            data = normalize(data)
            t2s = E2USD(win_size, step, E2USD_Adaper(params), DPGMM(None)).fit(data, win_size, step)
            prediction = t2s.state_seq + 1
            prediction = np.array(prediction, dtype=int)
            result = np.vstack([groundtruth, prediction])
            np.save(os.path.join(out_path, dir_name + str(i)), result)
            ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
            score_list.append(np.array([ari, anmi, nmi]))
            if verbose:
                print('ID: %s, ARI: %f, ANMI: %f, NMI: %f' % (dir_name, ari, anmi, nmi))
    score_list = np.vstack(score_list)
    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' % (np.mean(score_list[:, 0]) \
                                                       , np.mean(score_list[:, 1])
                                                   , np.mean(score_list[:, 2])))





def exp_on_PAMAP2(win_size, step, verbose=False):
    """Experiment on PAMAP2 dataset."""
    out_path = os.path.join(output_path, 'PAMAP2')
    create_path(out_path)
    params['in_channels'] = 9
    params['compared_length'] = win_size
    params['out_channels'] = 9

    dataset_path = os.path.join(data_path, 'PAMAP2/Protocol/subject10' + str(1) + '.dat')
    df = pd.read_csv(dataset_path, sep=' ', header=None)
    data = df.to_numpy()
    hand_acc = data[:, 4:7]
    chest_acc = data[:, 21:24]
    ankle_acc = data[:, 38:41]
    data = np.hstack([hand_acc, chest_acc, ankle_acc])
    data = fill_nan(data)
    data = normalize(data)
    t2s = E2USD(win_size, step, E2USD_Adaper(params), DPGMM(None)).fit(data, win_size, step)
    score_list = []
    for i in range(1, 9):
        dataset_path = os.path.join(data_path, 'PAMAP2/Protocol/subject10' + str(i) + '.dat')
        df = pd.read_csv(dataset_path, sep=' ', header=None)
        data = df.to_numpy()
        groundtruth = np.array(data[:, 1], dtype=int)
        hand_acc = data[:, 4:7]
        chest_acc = data[:, 21:24]
        ankle_acc = data[:, 38:41]
        data = np.hstack([hand_acc, chest_acc, ankle_acc])
        data = fill_nan(data)
        data = normalize(data)
        t2s.predict(data, win_size, step)
        prediction = t2s.state_seq
        prediction = np.array(prediction, dtype=int)
        ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
        score_list.append(np.array([ari, anmi, nmi]))
        if verbose:
            print('ID: %d, ARI: %f, ANMI: %f, NMI: %f' % (i, ari, anmi, nmi))
    score_list = np.vstack(score_list)
    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' % (np.mean(score_list[:, 0]) \
                                                       , np.mean(score_list[:, 1])
                                                   , np.mean(score_list[:, 2])))


def exp_on_USC_HAD(win_size, step, verbose=False):
    """Experiment on USC_HAD dataset."""
    out_path = os.path.join(output_path, 'USC-HAD')
    create_path(out_path)
    score_list = []
    score_list2 = []
    f_list = []
    params['in_channels'] = 6
    params['compared_length'] = win_size
    params['kernel_size'] = 3
    params['nb_steps'] = 40

    train, _ = load_USC_HAD(1, 1, data_path)
    train = normalize(train)
    t2s = E2USD(win_size, step, E2USD_Adaper(params), DPGMM(None)).fit(train, win_size, step)
    for subject in range(1, 15):
        for target in range(1, 6):
            data, groundtruth = load_USC_HAD(subject, target, data_path)
            data = normalize(data)
            t2s.predict(data, win_size, step)
            prediction = t2s.state_seq
            t2s.set_clustering_component(DPGMM(13)).predict_without_encode(data, win_size, step)
            prediction2 = t2s.state_seq
            prediction = np.array(prediction, dtype=int)
            result = np.vstack([groundtruth, prediction])
            np.save(os.path.join(out_path, 's%d_t%d' % (subject, target)), result)
            ari, anmi, nmi = evaluate_clustering(groundtruth, prediction)
            ari2, anmi2, nmi2 = evaluate_clustering(groundtruth, prediction2)
            f1, p, r = evaluate_cut_point(groundtruth, prediction2, 500)
            score_list.append(np.array([ari, anmi, nmi]))
            score_list2.append(np.array([ari2, anmi2, nmi2]))
            f_list.append(np.array([f1, p, r]))
            if verbose:
                print('ID: %s, ARI: %f, ANMI: %f, NMI: %f' % ('s' + str(subject) + 't' + str(target), ari, anmi, nmi))
                print(
                    'ID: %s, ARI: %f, ANMI: %f, NMI: %f' % ('s' + str(subject) + 't' + str(target), ari2, anmi2, nmi2))
    score_list = np.vstack(score_list)
    score_list2 = np.vstack(score_list2)
    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' % (np.mean(score_list[:, 0]) \
                                                       , np.mean(score_list[:, 1])
                                                   , np.mean(score_list[:, 2])))
    print('AVG ---- ARI: %f, ANMI: %f, NMI: %f' % (np.mean(score_list2[:, 0]) \
                                                       , np.mean(score_list2[:, 1])
                                                   , np.mean(score_list2[:, 2])))


if __name__ == '__main__':
    time_start = time.time()
    for i in range(10):
        print('round',i)
        print('exp_on_synthetic')
        exp_on_synthetic(256, 50, verbose=False)
        print('exp_on_MoCap')
        exp_on_MoCap(256, 50, verbose=False)
        print('exp_on_ActRecTut')
        exp_on_ActRecTut(128, 1, verbose=False)
        print('exp_on_PAMAP2')
        exp_on_PAMAP2(512, 100, verbose=False)
        print('exp_on_USC_HAD')
        exp_on_USC_HAD(512, 50, verbose=False)
        print('exp_on_UCR_SEG')
        exp_on_UCR_SEG(512, 50, verbose=False)
    time_end = time.time()
    print('time', time_end - time_start)