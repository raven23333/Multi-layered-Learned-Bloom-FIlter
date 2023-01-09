import hashlib
import math
import sys

import numpy as np
import pandas as pd
from sklearn.utils import murmurhash3_32
from random import randint
import os


def get_hash_function(m, ss=0):
    if ss <= 0:
        ss = randint(1, 99999999)

    def hash_m(x):
        return murmurhash3_32(x, seed=ss) % m

    return hash_m


class BloomFilter():
    """
    hash_len is the length of hash table, k is the number of hash functions
    h is the list of hash functions, n is expected element number？
    """

    def build_with_paras(self, df, paras):
        keys = df["key"]
        if "fpr" in paras:
            self.build_with_fpr(keys, paras["fpr"])
            return
        if "space" in paras:
            self.build_with_space(keys, paras["space"])

    def build_with_space(self, keys, hash_len):
        self.n = len(keys)
        self.hash_len = int(hash_len)
        if (self.n > 0) & (self.hash_len > 0):
            self.k = max(1, int(self.hash_len / self.n * 0.6931472))
        elif self.n == 0:
            self.k = 1
        self.h = []
        for i in range(self.k):
            self.h.append(get_hash_function(self.hash_len))
        self.table = np.zeros(self.hash_len, dtype=int)
        for key in keys:
            for j in range(self.k):
                t = self.h[j](key)
                self.table[t] = 1

    def build_with_fpr(self, keys, fpr):
        if fpr == 0:
            self.noTable = False
            self.insert(keys)
            return
        elif fpr == 1:
            self.noTable = True
            return
        hash_len = (-1 * len(keys) * math.log(fpr)) / (math.log(2) ** 2)
        hash_len = max(hash_len, 1) + 1
        self.build_with_space(keys, hash_len)

    def __init__(self, n, hash_len):
        self.n = n
        self.hash_len = int(hash_len)
        if (self.n > 0) & (self.hash_len > 0):
            self.k = max(1, int(self.hash_len / n * 0.6931472))
        elif self.n == 0:
            self.k = 1
        self.h = []
        for i in range(self.k):
            self.h.append(get_hash_function(self.hash_len))
        self.table = np.zeros(self.hash_len, dtype=int)
        self.noTable = None
        self.hash_set = set()

    def insert(self, key):
        if self.noTable is False:
            for item in key:
                self.hash_set.add(hashlib.md5(item.encode('utf-8')).hexdigest()[:4])
            return
        elif self.noTable is True:
            return
        if self.hash_len == 0:
            raise SyntaxError('cannot insert to an empty hash table')
        for i in key:
            for j in range(self.k):
                t = self.h[j](i)
                self.table[t] = 1

    def test_single(self, key):
        if self.noTable is False:
            return hashlib.md5(key.encode('utf-8')).hexdigest() in self.hash_set
        elif self.noTable is True:
            return True
        test_result = 0
        match = 0
        if self.hash_len > 0:
            for j in range(self.k):
                t = self.h[j](key)
                match += 1 * (self.table[t] == 1)
            if match == self.k:
                test_result = 1
        return test_result

    def test_group(self, keys):
        if self.noTable is True:
            return np.ones(len(keys))
        elif self.noTable is False:
            test_result = np.zeros(len(keys))
            ss = 0
            for item in keys:
                test_result[ss] = hashlib.md5(item.encode('utf-8')).hexdigest() in self.hash_set
                ss += 1
            return test_result
        else:
            test_result = np.zeros(len(keys))
            ss = 0
            if self.hash_len > 0:
                for key in keys:
                    match = 0
                    for j in range(self.k):
                        t = self.h[j](key)
                        match += 1 * (self.table[t] == 1)
                    if match == self.k:
                        test_result[ss] = 1
                    ss += 1
            return test_result

    def get_size(self):
        if self.noTable is True:
            return 0
        elif self.noTable is False:
            return len(self.hash_set)*16
        return len(self.table) + self.k


if __name__ == '__main__':
    DATA_PATH = 'dataset/URLdata.csv'
    data = pd.read_csv(DATA_PATH)
    data = data.rename(columns={'url': 'key'})
    negative_sample = data.loc[(data['label'] == -1)]
    positive_sample = data.loc[(data['label'] == 1)]
    train_negative = negative_sample.sample(frac=0.3)
    train_keys = negative_sample['key']
    test_keys = positive_sample['key']
    # train_negative = train_negative.loc[(train_negative['score'] >= 0.3) & (train_negative['score'] <= 0.6)]
    bf = BloomFilter(1, 1)
    # bf = BloomFilter(len(train_keys),4800000)
    # bf.insert(train_keys)
    # bf.build_with_space(train_keys, 4800000)
    bf.build_with_fpr(train_keys, 0.3)
    # res = bf.test_group(train_keys)
    # print("FPR=" + str(sum(res) / len(train_keys)))

    res = bf.test_group(test_keys)
    print("siz=" + str(bf.get_size()))
    print("FPR=" + str(sum(res) / len(test_keys)))
