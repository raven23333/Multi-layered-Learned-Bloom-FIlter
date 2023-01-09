import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from BF import BloomFilter
from util import *
import os

DATA_PATH = 'dataset/URLdata.csv'
c = 1.9
num_group = 8


def R_size(count_key, count_nonkey, R0):
    R = [0] * len(count_key)
    R[0] = max(R0, 1)
    for k in range(1, len(count_key)):
        R[k] = max(
            int(count_key[k] * (np.log(count_nonkey[0] / count_nonkey[k]) / np.log(0.618) + R[0] / count_key[0])), 1)
    return R


class dABF():

    def __init(self):
        self.thresholds = []
        self.nonempty = 0
        self.filters = []

    def build_with_paras(self, keys, labels, scores, paras):

        # preprocess
        thresholds = np.zeros(num_group + 1)
        thresholds[0] = -0.1
        thresholds[-1] = 1.1
        num_negative = len(labels) - sum(labels)
        tau = sum(c ** np.arange(0, num_group, 1))
        num_piece = int(num_negative / tau)
        scores.index = labels.index  # TODO 虽然不知道为啥好使但是有用
        print("???")
        score = np.sort(np.array(list(scores.loc[labels == False])))

        # Determine the thresholds array
        for i in range(1, num_group):
            if thresholds[-i] > 0:
                score_1 = score[score < thresholds[-i]]
                if int(num_piece * c ** (i - 1)) <= len(score_1):
                    thresholds[-(i + 1)] = score_1[-int(num_piece * c ** (i - 1))]
                else:
                    thresholds[-(i + 1)] = 0
            else:
                thresholds[-(i + 1)] = 1
        # calculate count_nonkey and some other parameters
        count_nonkey = np.zeros(num_group)
        for j in range(num_group):
            count_nonkey[j] = sum((score >= thresholds[j]) & (score < thresholds[j + 1]))
        num_nonempty_groups = sum(count_nonkey > 0)  # group number of (nonempty) groups
        count_nonkey = count_nonkey[count_nonkey > 0]  # key number in nonempty groups
        thresholds = thresholds[-(num_nonempty_groups + 1):]  # threshold without negative value
        self.thresholds = thresholds

        # Count the keys of each group
        used_key = keys[labels]  # TODO: rename to key
        score = np.array(list(scores.loc[labels == True]))

        count_key = np.zeros(num_nonempty_groups)
        key_group = []
        for j in range(num_nonempty_groups):
            count_key[j] = sum((score >= thresholds[j]) & (score < thresholds[j + 1]))
            key_group.append(
                used_key[(score >= thresholds[j]) & (score < thresholds[j + 1])])  # group urls by score interval
        init_res = [count_key, count_nonkey, num_nonempty_groups, key_group]

        non_empty_ix = min(np.where(count_key > 0)[0])
        self.non_empty_ix = non_empty_ix
        if "space" in paras:
            self.build_BFs_with_space(init_res, paras["space"])
        elif "fpr" in paras:
            self.build_BFs_with_FPR(init_res, paras["fpr"])
        # TODO: 如果出错，处理第一个区间左端点可能小于0的问题
        # if self.non_empty_ix - 0 > 1e-4:
        #    self.non_empty_ix=0

    def build_BFs_with_space(self, init_paras, space_budget):
        [count_key, count_nonkey, num_nonempty_groups, key_group] = init_paras
        # Search the Bloom filters' size
        R = np.zeros(num_nonempty_groups)
        R[:] = 0.5 * space_budget
        non_empty_ix = self.non_empty_ix
        if non_empty_ix > 0:
            R[0:non_empty_ix] = 0
        kk = 1
        while abs(sum(R) - space_budget) > 200:
            if sum(R) > space_budget:
                R[non_empty_ix] = R[non_empty_ix] - int((0.5 * space_budget) * 0.5 ** kk + 1)
            else:
                R[non_empty_ix] = R[non_empty_ix] + int((0.5 * space_budget) * 0.5 ** kk + 1)
            R[non_empty_ix:] = R_size(count_key, count_nonkey, R[non_empty_ix])
            if int((0.5 * space_budget) * (0.5) ** kk + 1) == 1:
                break
            kk += 1

        # threshold的最后一位是1.1，然后(1.1的前一个)~1.1这个区间没有BF
        # actually build BF
        Bloom_Filters = []
        for j in range(int(num_nonempty_groups)):
            if j < self.non_empty_ix:
                Bloom_Filters.append(BloomFilter(1, 1))
            else:
                Bloom_Filters.append(BloomFilter(to_str_df(key_group[j]).values, R[j]))
                Bloom_Filters[j].insert(key_group[j])
        self.filters = Bloom_Filters

    def build_BFs_with_FPR(self, init_paras, fpr):
        [count_key, count_nonkey, num_nonempty_groups, key_group] = init_paras
        bloom_filter_list = []
        total_nonkey = sum(count_nonkey)
        for j in range(int(num_nonempty_groups)):
            if j < self.non_empty_ix:
                bloom_filter_list.append(BloomFilter(1, 1))  # TODO
            else:
                curBF = BloomFilter(0, 0)
                curFPR1 = (fpr * total_nonkey) / (len(key_group) * count_nonkey[j])
                curFPR = max(0, min(1, curFPR1))
                print("group %s, ratio= %s, expFPR=%s" % (j,count_nonkey[j]/total_nonkey,curFPR1))  # TODO
                curBF.build_with_fpr(to_str_df(key_group[j]).values, curFPR)
                bloom_filter_list.append(curBF)
        self.filters = bloom_filter_list

    def getelem(self):
        return {'thres': self.thresholds, 'bfs': self.filters, 'non': self.non_empty_ix}

    def test_single(self, key, score):
        loc = 0
        for i in range(len(self.thresholds)):
            if score < self.thresholds[i]:
                loc = i - 1
                break
        return (self.filters[loc]).test_single(key)

    def test_group(self, keys, scores):
        result = []
        for key, score in zip(keys, scores):
            result.append(self.test_single(key, score))
        return np.array(result)


if __name__ == '__main__':
    data = pd.read_csv(DATA_PATH)
    data = data.rename(columns={'url': 'key'})
    data["label"] = data["label"].apply(lambda x: x == 1)
    negative_sample = data.loc[(data['label'] == False)]
    positive_sample = data.loc[data['label']]
    train_negative = negative_sample.sample(frac=0.3)
    # train_negative = train_negative.loc[(train_negative['score'] >= 0.3) & (train_negative['score'] <= 0.6)]

    dbf = dABF()
    dbf.build_with_paras(data["key"], data["label"], data["score"], {"fpr": 0.01})
    res = dbf.getelem()
    thresholds = res['thres']
    non_empty_ix = res['non']
    Bloom_Filters = res['bfs']
    print(dbf.non_empty_ix)
    print(thresholds)

    '''
    ML_positive = train_negative.loc[(train_negative['score'] >= thresholds[-2]), 'key']
    url_negative = train_negative.loc[(train_negative['score'] < thresholds[-2]), 'key']
    score_negative = train_negative.loc[(train_negative['score'] < thresholds[-2]), 'score']
    test_result = np.zeros(len(url_negative))
    ss = 0
    
    for score_s, url_s in zip(score_negative, url_negative):
        ix = min(np.where(score_s < thresholds)[0]) - 1
        if ix >= non_empty_ix:
            test_result[ss] = Bloom_Filters[ix].test_single(url_s)
        else:
            test_result[ss] = 0
        ss += 1
    FP_items = sum(test_result) + len(ML_positive)
    print('False positive items: %f, Number of groups: %d, c = %f' % (FP_items / len(train_negative), 8, round(1.9, 2)))
    '''
    FP_items = sum(dbf.test_group(train_negative['key'], train_negative['score']))
    print('False positive items: %f, Number of groups: %d, c = %f' % (FP_items / len(train_negative), 8, round(1.9, 2)))
