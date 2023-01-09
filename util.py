import math
import math
import os
import random
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn
from joblib import dump, load
from sklearn.utils import murmurhash3_32

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from GRUModel import GRUModel
from sklearn.neural_network import MLPClassifier
from sklearn import svm


def to_str_df(key_df):
    data_str = key_df.astype(str).apply(lambda xs: ','.join(xs), axis=1)
    return data_str


def to_series_key(key_df):
    if type(key_df) == pd.DataFrame and key_df.shape[1] == 1:
        col_str = key_df.columns[0]
        new_keys = key_df[col_str]
        return new_keys
    else:
        warnings.warn("converting to series failed")
        exit(0)


def to_binary_arr(key_df, cur_level, cur_range, cur_seed=1):
    data_str = key_df.astype(str).apply(lambda xs: ','.join(xs), axis=1)
    # lever_factor = int(cur_level) ^ int(cur_range)
    lever_factor = int(cur_range) ** int(cur_level)
    idx = data_str.apply(lambda x: int(murmurhash3_32(x, seed=cur_seed) / lever_factor) % cur_range).values
    return idx


def get_size(clf):
    if type(clf) is DecisionTreeClassifier:
        return len(clf.tree_.children_left) * 6 * 8
    elif type(clf) is RandomForestClassifier:
        ss = 0
        for tree in clf.estimators_:
            ss += get_size(tree) * 8
        return ss
    elif type(clf) is GRUModel:
        ss = 0
        for arr in clf.model.weights:
            cur_dim = arr.shape.dims
            for dimension in cur_dim:
                ss += dimension.value
        return ss * 32
    elif type(clf) is MLPClassifier:
        return sum(clf.hidden_layer_sizes) * 8
    elif type(clf) is svm.NuSVC or type(clf) is svm.NuSVC:
        return 400
    # elif type(clf) is XGBClassifier:
    #    return 400 # TODO
    else:
        return get_model_size_by_dump(clf, "r")


def get_model_size_by_dump(clf, loc):
    dump(clf, "model/" + loc + 'filename.npy')  # TODO
    model_size = os.path.getsize("model/" + loc + 'filename.npy')
    #os.remove("model/" + loc + 'filename.npy')
    return model_size


'''
def get_train_x(keys, scores):
    # Note this method reserves the index in the key but ignores that in the scores
    if scores is None:
        train_x = keys
    else:
        idx = keys.index
        scores.index = idx
        train_x = pd.concat([keys, scores], axis=1)
        train_x.index = idx
    return train_x
'''


def max_min_scaler(x): return (x - np.min(x)) / (np.max(x) - np.min(x))


def interval_scaler(arr, low, high): return (arr - low) / (high - low)


def get_exp_bf_size(key_num, fpr):
    if fpr == 0:
        print("warning: fpr 0")
        return key_num * 128
    elif fpr == 1:
        print("warning: fpr 1")
        return 0
    else:
        return int(key_num * (-1.44) * math.log2(fpr))


def analysis(pos_scores, neg_scores, thres=None):
    pos_std = np.std(pos_scores, ddof=1)
    neg_std = np.std(neg_scores, ddof=1)
    pos_mean = np.mean(pos_scores)
    neg_mean = np.mean(neg_scores)
    print("pos_std %s, neg_std %s, pos_mean %s, neg_mean %s" % (pos_std, neg_std, pos_mean, neg_mean))
    if thres:
        for i in range(len(thres) - 1):
            pos_num = sum([thres[i] < x < thres[i + 1] for x in pos_scores])
            neg_num = sum([thres[i] < x < thres[i + 1] for x in neg_scores])
            print("range %s: pos key %s neg key %s ratio %s" % (i, pos_num, neg_num, pos_num * 1.0 / (1 + neg_num)))


def update_weight(pred, label, weights):
    miss = [int(x) for x in (pred != label)]
    err_m = np.dot(weights, miss) / sum(weights)
    if err_m == 0:
        alpha = 5
    else:
        alpha = 0.5 * np.log((1 - err_m) / float(err_m))
    miss2 = [x if x == 1 else -1 for x in miss]  # -1 * y_i * G(x_i): 1 / -1
    weights = np.multiply(weights, np.exp([float(x) * alpha for x in miss2]))
    return alpha, weights


def update_score(alpha_sum, alpha_local, score_sum, score_local):
    alpha_new = alpha_local + alpha_sum
    score_new = (score_sum * alpha_sum + score_local * alpha_local) / alpha_new
    return alpha_new, score_new


def shuffle_for_training(negatives, positives):
    if type(negatives) == pd.DataFrame or type(positives) == pd.DataFrame:
        assert (negatives.shape[1] == 1 and positives.shape[1] == 1)
        # a quick dirty fix, it supposed to accept multiple cols
        a = [(i, 0) for i in negatives[negatives.columns[0]]]
        b = [(i, 1) for i in positives[negatives.columns[0]]]
    else:
        a = [(i, 0) for i in negatives]
        b = [(i, 1) for i in positives]
    combined = a + b
    random.shuffle(combined)
    return list(zip(*combined))


def set_rand_seed(seed):
    random.seed(seed)  # 24
    np.random.seed(seed)
    tf.random.set_seed(seed)


def fetch_pos_score(raw_score):
    if len(raw_score.shape) == 1 or raw_score.shape[1] == 1 or raw_score.size == 0:
        return raw_score
    else:
        return raw_score[:, 1]


def load_data_from_csv(dataset_name, col_num):
    if "facebook" == dataset_name:
        data = pd.read_csv('dataset/train.csv', nrows=col_num, index_col='row_id')
        key_cols = ["x", "y"]  # "accuracy", "time" #,"place_id"
        data["label"] = data["x"].values + data["y"].values
        data["label"] = data["label"].apply(lambda x: (int(x / 2)) % 2 == 0)  # 2
    elif "url" == dataset_name:
        data = pd.read_csv('dataset/URLdata9.csv', nrows=col_num)  #
        key_cols = ["url"]  # "accuracy", "time" #,"place_id"
    elif "higgs" == dataset_name:
        data = pd.read_csv('dataset/higgs3.csv', nrows=col_num)
        key_cols = data.columns[:-1]
    elif "letor" == dataset_name:
        data = pd.read_csv('dataset/letor.csv', nrows=col_num)
        key_cols = data.columns[:-1]
    elif "amazon" == dataset_name:
        data = pd.read_csv('dataset/amazon.csv', nrows=col_num)
        key_cols = data.columns[1:]
    elif "zipf" == dataset_name:
        data = pd.read_csv('dataset/zipf_105.csv', nrows=col_num)
        key_cols = ["key"]
    elif "train1000" == dataset_name:
        data = pd.read_csv('dataset/1000_train.csv', nrows=col_num)
        data = data.reset_index()
        data.rename(columns={'index': 'label'}, inplace=True)
        key_cols = data.columns[1:]
        data["label"] = data["label"].apply(lambda x: x > 0.5)

    else:
        print("unknown dataset")
        exit(0)
    return key_cols, data


def zipf_generator(cur_C, cur_a, cur_range, path):
    raw = np.array(range(1, cur_range + 1))
    raw = cur_C / (raw ** cur_a)
    label = np.array(list(map(lambda x: x > random.random(), raw)))
    df = pd.DataFrame({"key": list(range(1, cur_range + 1)), "label": label})
    df.to_csv("dataset/" + path + ".csv", index=False)
    return label


class ZipfClassifier:
    def __init__(self, zipf_c=200000, zipf_alpha=1.05):
        self.alpha = zipf_alpha
        self.c = zipf_c

    def fit(self, x, y, sample_weight=None):
        pass

    def predict_proba(self, keys):
        cur_res = np.minimum(1, self.c / keys ** self.alpha)
        return np.array(cur_res)


if __name__ == '__main__':
    res = zipf_generator(200000, 1.05, 1000000, "zipf_105")
    # print(res)
    print(len(res))
    print(sum(res))
    '''
    data = pd.read_csv('./train.csv', nrows=20000, index_col='row_id')
    data_y = data["place_id"].apply(lambda x: x > 5501003147)
    data_x = data[["x", "y", "accuracy", "time"]]  # , "accuracy","place_id"
    '''

    '''
    clf = sklearn.ensemble.RandomForestClassifier(n_estimators=10,max_depth=20)
    clf.fit(X=data_x, y=data_y)
    y_pred = clf.predict(data_x)
    y_score = clf.predict_proba(data_x)
    ss = 0
    '''

    '''
    format_df = pd.DataFrame()
    format_df["key"] = data_x.values.tolist()
    format_df["label"] = data_y

    clf = sklearn.ensemble.RandomForestClassifier(n_estimators=10, max_depth=20)
    x = pd.DataFrame(format_df["key"].values.tolist())
    clf.fit(X=x, y=format_df["label"])

    print(x[0:10])
    '''
# TODO:把所有东西都打包成key score和label列似乎不错
'''
def to_format_df(df, key_cols, label_col, score_cols=[]):
    format_df = pd.DataFrame()
    format_df["key"] = data[key_cols].values.tolist()
    format_df["label"] = data_y
    if score_cols is not []:
        format_df["score"] = data[key_cols].values.tolist()
    return format_df

def from_format_df(format_df):
'''
