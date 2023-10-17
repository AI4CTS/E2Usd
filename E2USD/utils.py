import numpy as np
import torch.utils.data
import math
from sklearn import metrics

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return np.shape(self.dataset)[0]

    def __getitem__(self, index):
        return self.dataset[index]


class LabelledDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels

    def __len__(self):
        return np.shape(self.dataset)[0]

    def __getitem__(self, index):
        return self.dataset[index], self.labels[index]

def all_normalize(data_tensor):
    mean = np.mean(data_tensor)
    var = np.var(data_tensor)
    i = 0
    for channel in data_tensor[0]:
        data_tensor[0][i] = (channel - mean)/math.sqrt(var)
        i += 1
    return data_tensor

def compact(series):
    compacted = []
    pre = series[0]
    compacted.append(pre)
    for e in series[1:]:
        if e != pre:
            pre = e
            compacted.append(e)
    return compacted

def remove_duplication(series):

    result = []
    for e in series:
        if e not in result:
            result.append(e)
    return result

def reorder_label(label):
    label = np.array(label)
    ordered_label_set = remove_duplication(compact(label))
    idx_list = [np.argwhere(label==e) for e in ordered_label_set]
    for i, idx in enumerate(idx_list):
        label[idx] = i
    return label


def __adjusted_macro_score(groundtruth, prediction, score_type):
    if len(prediction) != len(groundtruth):
        print('prediction and groundtruth must be of the same length')
        return

    length = len(groundtruth)

    # convert to numpy array.
    groundtruth = np.array(groundtruth, dtype=int)
    prediction = np.array(prediction, dtype=int)

    groundtruth_set = set(groundtruth)
    prediction_set = set(prediction)
    used_label_set = set()
    n = len(groundtruth_set)

    total = 0
    for i in groundtruth_set:
        used_j = 0
        max_score = 0
        for j in prediction_set:
            if j in used_label_set:
                continue

            # convert to binary array of positive label.
            true = np.zeros(length, dtype=int)
            pred = np.zeros(length, dtype=int)
            true[np.argwhere(groundtruth == i)] = 1
            pred[np.argwhere(prediction == j)] = 1

            if score_type == 'f1':
                score = metrics.f1_score(true, pred, average='binary', pos_label=1, zero_division=0)
            elif score_type == 'precision':
                score = metrics.precision_score(true, pred, average='binary', pos_label=1, zero_division=0)
            elif score_type == 'recall':
                score = metrics.recall_score(true, pred, average='binary', pos_label=1, zero_division=0)
            else:
                print('Error: Score type does not exists.')

            if score > max_score:
                max_score = score
                used_j = j

        used_label_set.add(used_j)
        total += max_score
    return total / n


def adjusted_macro_recall(groundtruth, prediction):
    return __adjusted_macro_score(groundtruth, prediction, score_type='recall')


def adjusted_macro_precision(groundtruth, prediction):
    return __adjusted_macro_score(groundtruth, prediction, score_type='precision')


def adjusted_macro_f1score(groundtruth, prediction):
    return __adjusted_macro_score(groundtruth, prediction, score_type='f1')


def macro_recall(groundtruth, prediction, if_reorder_label=False):
    '''
    calculate macro precision
    '''
    groundtruth = np.array(groundtruth, dtype=int)
    prediction = np.array(prediction, dtype=int)
    if len(prediction) != len(groundtruth):
        print('prediction and groundtruth must be of the same length')
    else:
        if if_reorder_label:
            groundtruth = reorder_label(groundtruth)
            prediction = reorder_label(prediction)
        return metrics.recall_score(groundtruth, prediction, average='macro', zero_division=0)


def macro_precision(groundtruth, prediction, if_reorder_label=False):
    '''
    calculate macro precision
    '''
    groundtruth = np.array(groundtruth, dtype=int)
    prediction = np.array(prediction, dtype=int)
    if len(prediction) != len(groundtruth):
        print('prediction and groundtruth must be of the same length')
    else:
        if if_reorder_label:
            groundtruth = reorder_label(groundtruth)
            prediction = reorder_label(prediction)
        return metrics.precision_score(groundtruth, prediction, average='macro', zero_division=0)


def macro_f1score(groundtruth, prediction, if_reorder_label=False):
    '''
    calculate macro f1 score
    '''
    groundtruth = np.array(groundtruth, dtype=int)
    prediction = np.array(prediction, dtype=int)
    if len(prediction) != len(groundtruth):
        print('prediction and groundtruth must be of the same length')
    else:
        if if_reorder_label:
            groundtruth = reorder_label(groundtruth)
            prediction = reorder_label(prediction)
        return metrics.f1_score(groundtruth, prediction, average='macro', zero_division=0)


def find_cut_points_from_label(label):
    pre = 0
    current = 0
    length = len(list(label))
    result = []
    while current < length:
        if label[current] == label[pre]:
            current += 1
            continue
        else:
            result.append(current)
            pre = current
    return result


def evaluate_cut_point(groundtruth, prediction, d):
    list_true_pos_cut = find_cut_points_from_label(groundtruth)
    list_pred_pos_cut = find_cut_points_from_label(prediction)
    # print(list_true_pos_cut, list_pred_pos_cut)
    tp = 0
    fn = 0
    for pos_true in list_true_pos_cut:
        flag = False
        list_elem_to_be_removed = []
        for pos_pred in list_pred_pos_cut:
            if pos_pred >= pos_true - d and pos_pred <= pos_true + d - 1:
                # print('current elem %d, internal[%d,%d],pop%d'%(pos_pred, pos_true-d, pos_true+d-1, pos_pred))
                # list_pred_pos_cut.remove(pos_pred)
                list_elem_to_be_removed.append(pos_pred)
                flag = True
        if not flag:
            fn += 1
        else:
            tp += 1
        # remove
        for e in list_elem_to_be_removed:
            list_pred_pos_cut.remove(e)
        # print(list_pred_pos_cut)

    fp = len(list_pred_pos_cut)

    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)

    if precision + recall == 0:
        fscore = 0
    else:
        fscore = 2 * precision * recall / (precision + recall)
    return fscore, precision, recall


# groundtruth = [0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,1,1,1,2,2,5,5,5,5,5]
# prediction =  [1,1,1,1,0,1,0,0,0,1,1,1,1,1,2,2,2,2,2,2,5,5,5,3,3]
# print(evaluate_cut_point(groundtruth, prediction, 2))

def ARI(prediction, groundtruth):
    return metrics.adjusted_rand_score(groundtruth, prediction)


def ANMI(prediction, groundtruth):
    return metrics.adjusted_mutual_info_score(groundtruth, prediction)


def NMI(groundtruth, prediction):
    return metrics.normalized_mutual_info_score(groundtruth, prediction)


def evaluate_clustering(groundtruth, prediction):
    ari = ARI(groundtruth, prediction)
    anmi = ANMI(groundtruth, prediction)
    nmi = NMI(groundtruth, prediction)
    return ari, anmi, nmi


def evaluation(groundtruth, prediction):
    ari = ARI(groundtruth, prediction)
    ami = ANMI(groundtruth, prediction)
    f1 = adjusted_macro_f1score(groundtruth, prediction)
    precision = adjusted_macro_precision(groundtruth, prediction)
    recall = adjusted_macro_recall(groundtruth, prediction)
    return f1, precision, recall, ari, ami


def adjusted_macro_F_measure(groundtruth, prediction):
    f1 = adjusted_macro_f1score(groundtruth, prediction)
    precision = adjusted_macro_precision(groundtruth, prediction)
    recall = adjusted_macro_recall(groundtruth, prediction)
    return f1, precision, recall


def remove_constant_col(df):
    data = df.to_numpy()
    sum = np.sum(data, axis=0)
    data = np.squeeze(data[:, np.argwhere(sum != 0)], 2)
    return pd.DataFrame(data)


def len_of_file(path):
    return len(open(path, 'rU').readlines())


def batch_z_normalize(data_tensor):
    result = np.empty(shape=data_tensor.shape)
    num_batch, _, _ = data_tensor.shape
    for i in range(num_batch):
        w = data_tensor[i, :, :]
        _range = np.max(w) - np.min(w)
        w = (w - np.min(w)) / _range
        result[i] = w
    return result


def normalize(X, mode='channel'):
    if mode == 'channel':
        for i in range(X.shape[1]):
            max = np.max(X[:, i])
            min = np.min(X[:, i])
            X[:, i] = (X[:, i] - min) / (max - min)
    elif mode == 'all':
        max = np.max(X)
        min = np.min(X)
        X = (X - min) / (max - min)

    return X


def all_normalize(data_tensor):
    mean = np.mean(data_tensor)
    var = np.var(data_tensor)
    i = 0
    for channel in data_tensor[0]:
        data_tensor[0][i] = (channel - mean) / math.sqrt(var)
        i += 1
    return data_tensor


def z_normalize(array):
    _range = np.max(array) - np.min(array)
    return (array - np.min(array)) / _range


# return the index of elems inside the interval (start,end]
def find(array, start, end):
    pos_min = array > start
    pos_max = array <= end
    return np.argwhere(pos_min & pos_max == True)


def calculate_density_matrix(feature_list, n=100):
    # convert to np array
    feature_list = np.array(feature_list)
    x = feature_list[:, 0]
    y = feature_list[:, 1]

    h_start = np.min(y)
    h_end = np.max(y)
    h_step = (h_end - h_start) / n
    w_start = np.min(x)
    w_end = np.max(x)
    w_step = (w_end - w_start) / n

    row_partition = []
    for i in range(n):
        row_partition.append(find(y, h_start + i * h_step, h_start + (i + 1) * h_step))
    # print(len(row_partition))

    row_partition = list(reversed(row_partition))

    density_matrix = []

    for row_idx in row_partition:
        row = x[row_idx]
        row_densities = []
        for i in range(n):
            density = len(find(row, w_start + i * w_step, w_start + (i + 1) * w_step))
            row_densities.append(density)
        density_matrix.append(row_densities)
    density_matrix = np.array(density_matrix)
    return density_matrix, w_start, w_end, h_start, h_end,


def calculate_scalar_velocity_list(feature_list, interval=1):
    velocity_list = []

    for pre_pos, pos in zip(feature_list[:-interval], feature_list[interval:]):
        # calculate displacement
        velocity_list.append(np.linalg.norm(pos - pre_pos))
    for i in range(interval):
        velocity_list.append(0)
    velocity_list = np.array(velocity_list)

    return velocity_list


def calculate_velocity_list(feature_list, interval=500):
    velocity_list_x = []
    velocity_list_y = []

    for pre_pos, pos in zip(feature_list[:-interval], feature_list[interval:]):
        # calculate displacement for x, y, respectively
        displacement_x = (pos[0] - pre_pos[0]) / interval
        displacement_y = (pos[1] - pre_pos[1]) / interval
        velocity_list_x.append(displacement_x)
        velocity_list_y.append(displacement_y)
    for i in range(interval):
        velocity_list_y.append(0)
        velocity_list_x.append(0)
    velocity_list_x = np.array(velocity_list_x)
    velocity_list_y = np.array(velocity_list_y)

    return velocity_list_x, velocity_list_y

def __adjusted_macro_score(groundtruth, prediction, score_type):
    if len(prediction) != len(groundtruth):
        print('prediction and groundtruth must be of the same length')
        return

    length = len(groundtruth)

    # convert to numpy array.
    groundtruth = np.array(groundtruth, dtype=int)
    prediction = np.array(prediction, dtype=int)

    groundtruth_set = set(groundtruth)
    prediction_set = set(prediction)
    used_label_set = set()
    n = len(groundtruth_set)

    total = 0
    for i in groundtruth_set:
        used_j = 0
        max_score = 0
        for j in prediction_set:
            if j in used_label_set:
                continue

            # convert to binary array of positive label.
            true = np.zeros(length, dtype=int)
            pred = np.zeros(length, dtype=int)
            true[np.argwhere(groundtruth == i)] = 1
            pred[np.argwhere(prediction == j)] = 1

            if score_type == 'f1':
                score = metrics.f1_score(true, pred, average='binary', pos_label=1, zero_division=0)
            elif score_type == 'precision':
                score = metrics.precision_score(true, pred, average='binary', pos_label=1, zero_division=0)
            elif score_type == 'recall':
                score = metrics.recall_score(true, pred, average='binary', pos_label=1, zero_division=0)
            else:
                print('Error: Score type does not exists.')

            if score > max_score:
                max_score = score
                used_j = j

        used_label_set.add(used_j)
        total += max_score
    return total / n


def adjusted_macro_recall(groundtruth, prediction):
    return __adjusted_macro_score(groundtruth, prediction, score_type='recall')


def adjusted_macro_precision(groundtruth, prediction):
    return __adjusted_macro_score(groundtruth, prediction, score_type='precision')


def adjusted_macro_f1score(groundtruth, prediction):
    return __adjusted_macro_score(groundtruth, prediction, score_type='f1')


def macro_recall(groundtruth, prediction, if_reorder_label=False):
    groundtruth = np.array(groundtruth, dtype=int)
    prediction = np.array(prediction, dtype=int)
    if len(prediction) != len(groundtruth):
        print('prediction and groundtruth must be of the same length')
    else:
        if if_reorder_label:
            groundtruth = reorder_label(groundtruth)
            prediction = reorder_label(prediction)
        return metrics.recall_score(groundtruth, prediction, average='macro', zero_division=0)


def macro_precision(groundtruth, prediction, if_reorder_label=False):
    groundtruth = np.array(groundtruth, dtype=int)
    prediction = np.array(prediction, dtype=int)
    if len(prediction) != len(groundtruth):
        print('prediction and groundtruth must be of the same length')
    else:
        if if_reorder_label:
            groundtruth = reorder_label(groundtruth)
            prediction = reorder_label(prediction)
        return metrics.precision_score(groundtruth, prediction, average='macro', zero_division=0)


def macro_f1score(groundtruth, prediction, if_reorder_label=False):
    groundtruth = np.array(groundtruth, dtype=int)
    prediction = np.array(prediction, dtype=int)
    if len(prediction) != len(groundtruth):
        print('prediction and groundtruth must be of the same length')
    else:
        if if_reorder_label:
            groundtruth = reorder_label(groundtruth)
            prediction = reorder_label(prediction)
        return metrics.f1_score(groundtruth, prediction, average='macro', zero_division=0)


def find_cut_points_from_label(label):
    pre = 0
    current = 0
    length = len(list(label))
    result = []
    while current < length:
        if label[current] == label[pre]:
            current += 1
            continue
        else:
            result.append(current)
            pre = current
    return result


def evaluate_cut_point(groundtruth, prediction, d):
    list_true_pos_cut = find_cut_points_from_label(groundtruth)
    list_pred_pos_cut = find_cut_points_from_label(prediction)
    # print(list_true_pos_cut, list_pred_pos_cut)
    tp = 0
    fn = 0
    for pos_true in list_true_pos_cut:
        flag = False
        list_elem_to_be_removed = []
        for pos_pred in list_pred_pos_cut:
            if pos_pred >= pos_true - d and pos_pred <= pos_true + d - 1:
                list_elem_to_be_removed.append(pos_pred)
                flag = True
        if not flag:
            fn += 1
        else:
            tp += 1
        for e in list_elem_to_be_removed:
            list_pred_pos_cut.remove(e)

    fp = len(list_pred_pos_cut)

    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)

    if precision + recall == 0:
        fscore = 0
    else:
        fscore = 2 * precision * recall / (precision + recall)
    return fscore, precision, recall

def ARI(prediction, groundtruth):
    return metrics.adjusted_rand_score(groundtruth, prediction)


def ANMI(prediction, groundtruth):
    return metrics.adjusted_mutual_info_score(groundtruth, prediction)


def NMI(groundtruth, prediction):
    return metrics.normalized_mutual_info_score(groundtruth, prediction)


def evaluate_clustering(groundtruth, prediction):
    ari = ARI(groundtruth, prediction)
    anmi = ANMI(groundtruth, prediction)
    nmi = NMI(groundtruth, prediction)
    return ari, anmi, nmi


def evaluation(groundtruth, prediction):
    ari = ARI(groundtruth, prediction)
    ami = ANMI(groundtruth, prediction)
    f1 = adjusted_macro_f1score(groundtruth, prediction)
    precision = adjusted_macro_precision(groundtruth, prediction)
    recall = adjusted_macro_recall(groundtruth, prediction)
    return f1, precision, recall, ari, ami


def adjusted_macro_F_measure(groundtruth, prediction):
    f1 = adjusted_macro_f1score(groundtruth, prediction)
    precision = adjusted_macro_precision(groundtruth, prediction)
    recall = adjusted_macro_recall(groundtruth, prediction)
    return f1, precision, recall

def fill_nan(data):
    x_len, y_len = data.shape
    for x in range(x_len):
        for y in range(y_len):
            if np.isnan(data[x, y]):
                data[x, y] = data[x - 1, y]
    return data


def compact(series):
    '''
    Compact Time Series.
    '''
    compacted = []
    pre = series[0]
    compacted.append(pre)
    for e in series[1:]:
        if e != pre:
            pre = e
            compacted.append(e)
    return compacted


def remove_duplication(series):
    '''
    Remove duplication.
    '''
    result = []
    for e in series:
        if e not in result:
            result.append(e)
    return result


def seg_to_label(label):
    pre = 0
    seg = []
    for l in label:
        seg.append(np.ones(l - pre, dtype=int) * label[l])
        pre = l
    result = np.concatenate(seg)
    return result


def reorder_label(label):
    # Start from 0.
    label = np.array(label)
    ordered_label_set = remove_duplication(compact(label))
    idx_list = [np.argwhere(label == e) for e in ordered_label_set]
    for i, idx in enumerate(idx_list):
        label[idx] = i
    return label


def adjust_label(label):
    '''
    Adjust label order.
    '''
    label = np.array(label)
    compacted_label = compact(label)
    ordered_label_set = remove_duplication(compacted_label)
    label_set = set(label)
    idx_list = [np.argwhere(label == e) for e in ordered_label_set]
    for idx, elem in zip(idx_list, label_set):
        label[idx] = elem
    return label


def bucket_vote(bucket):
    '''
    The bucket vote algorithm.
    @return: element of the largest amount.
    @Param bucket: the bucket of data, array like, one dim.
    '''
    vote_vector = np.zeros(len(set(bucket)), dtype=int)

    # create symbol table
    symbol_table = {}
    symbol_list = []
    for i, s in enumerate(set(bucket)):
        symbol_table[s] = i
        symbol_list.append(s)

    # do vote
    for e in bucket:
        vote_vector[symbol_table[e]] += 1

    symbol_idx = np.argmax(vote_vector)
    return symbol_list[symbol_idx]


def smooth(X, bucket_size):
    for i in range(0, len(X), bucket_size):
        s = bucket_vote(X[i:i + bucket_size])
        true_size = len(X[i:i + bucket_size])
        X[i:i + bucket_size] = s * np.ones(true_size, dtype=int)
    return X


def dilate_label(label, f, max_len):
    slice_list = []
    for e in label:
        slice_list.append(e * np.ones(f, dtype=int))
    return np.concatenate(slice_list)[:max_len]


def str_list_to_label(label):
    label_set = remove_duplication(label)
    label = np.array(label)
    new_label = np.array(np.ones(len(label)))
    for i, l in enumerate(label_set):
        idx = np.argwhere(label == l)
        new_label[idx] = i
    return new_label.astype(int)