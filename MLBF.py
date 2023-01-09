import math
import warnings

import numpy as np

from PLBF import get_parameter_vals
from dABF import *
from xgboost import XGBClassifier
from sklearn import svm
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
import time
from tqdm import *

# TODO:双 minimum size
# NOTE: link模式的partition只有uniform没有ada-BF
node_size = 10000  # 50000
batch_size = 5000  # 20000
MIN_INNER_SIZE = 2000

MLBF_para = {"fpr": 0.001, "partition": 6, "boost": True, "split": "ada-BF", "split_num": 4,
             "max-layer-num": 500,  # math.inf #uniform
             "cheat": False, "multiple_candidates": False, "warm-start": False,  # TODO
             "advanced-dynamic-fpr": True
             }  # ada-BF uniform

rnn_para = {"learning_rate": 0.001, "maxlen": 50, "pca_embedding_dim": None,
            "gru_size": 16, "batch_size": 1024, "hidden_size": None, "second_gru_size": None, "decay": 0.0001,
            "epochs": 30}


# boost_dic:包含score, alpha-sum, weight(后面俩可能有) # model_dic:包含模型参数
# para_dic: 包含杂项 #multi-dic: 包含多个其他dic

class node:
    def __init__(self, loc):
        self.leaves = []
        self.fpr = MLBF_para["fpr"]
        self.loc = loc
        self.element_numb = 0
        self.pos_numb = 0

    def insert(self, keys, label, model_dic=None):
        pass

    def test_group(self, keys, score_dic=None):
        return None

    def insert_with_clean(self, keys, labels):
        if keys.empty:
            return
        test_res = self.test_group(keys)
        insert_idx = np.logical_not(np.logical_and(test_res, labels))
        self.insert(keys[insert_idx], labels[insert_idx])

    def get_size(self, verbose=False):
        pass

    def get_cache_size(self):
        return 0

    def finalize_insert(self):
        if self.leaves:
            for j in range(len(self.leaves)):
                if type(self.leaves[j]) is hash_node:
                    tmp_keys, tmp_label, tmp_multi_dic = self.leaves[j].output()
                    self.leaves[j] = inner_node.to_bf(tmp_keys, tmp_label, tmp_multi_dic)
                else:
                    self.leaves[j].finalize_insert()


class bf_node(node):
    def __init__(self, loc, fpr=MLBF_para["fpr"]):
        super().__init__(loc)
        self.fpr = fpr
        self.bf = BloomFilter(1, 1)

    def insert(self, keys, label=None, multi_dic=None):
        if self.bf.hash_len == 1:
            if label is None:
                self.element_numb = keys.shape[0]
                self.pos_numb = keys.shape[0]
                self.bf.build_with_fpr(to_str_df(keys).values, self.fpr)
            else:
                self.element_numb = sum(label)
                self.pos_numb = self.element_numb
                self.bf.build_with_fpr(to_str_df(keys[label]).values, self.fpr)
            return True
        else:
            warnings.warn("attempting to insert into BF")
            if not self.leaves:
                self.leaves.append(hash_node(self.loc + '#'))
            self.leaves[0].insert(keys, label, multi_dic)
            self.leaves[0] = hash_node.to_inner_maybe(self.leaves[0])

    def local_insert(self, keys, label=None, multi_dic=None):
        self.element_numb += keys.shape[0]
        if label is None:
            self.pos_numb += keys.shape[0]
            insert_keys = to_str_df(keys).values
        else:
            self.pos_numb += sum(label)
            insert_keys = to_str_df(keys[label]).values

        if self.bf.hash_len == 1:
            self.bf.build_with_fpr(insert_keys, self.fpr)
            return
        else:
            self.bf.insert(insert_keys)

    def test_group(self, keys, boost_dic=None):
        if self.element_numb == 0:
            return np.zeros(len(keys))
        local_truths = self.bf.test_group(to_str_df(keys).values)
        if self.leaves:
            retest_idx = np.logical_not(local_truths)
            local_truths[retest_idx] = inner_node.test_child(self.leaves[0], keys, retest_idx, boost_dic)
        return local_truths

    def get_size(self, verbose=False):
        bf_size = self.bf.get_size()
        leaf_size = 0
        if self.leaves:
            for child in self.leaves:
                leaf_size += child.get_size(verbose)  # False
        sum_size = bf_size + leaf_size
        if verbose:
            print("bf_node, bf_size = " + str(bf_size) + " ,leaf_size = " + str(leaf_size) +
                  " ,total_size = " + str(sum_size))
        return bf_size + leaf_size

    def get_cache_size(self):
        leaf_size = 0
        if self.leaves:
            for child in self.leaves:
                leaf_size += child.get_cache_size()
        return leaf_size

    def describe(self):
        pass


class hash_node(node):
    def __init__(self, loc):
        super().__init__(loc)
        self.hash_table = set()
        self.df_list = None
        self.multi_dic = None

    def insert(self, keys, label, new_multi_dic=None):
        if self.multi_dic is None:
            self.multi_dic = new_multi_dic
        elif MLBF_para["boost"]:
            self.multi_dic["boost_dic"]["weights"] = np.concatenate(
                (self.multi_dic["boost_dic"]["weights"], new_multi_dic["boost_dic"]["weights"]))
            self.multi_dic["boost_dic"]["scores"] = np.concatenate(
                (self.multi_dic["boost_dic"]["scores"], new_multi_dic["boost_dic"]["scores"]))

        tmp_df_list = [keys, label]
        if self.df_list is None:
            self.df_list = tmp_df_list
        else:
            self.df_list = [pd.concat([x, y]) if (x is not None and y is not None) else None
                            for x, y in zip(self.df_list, tmp_df_list)]

        pos_key = to_str_df(keys[label]).values
        for row in pos_key:
            self.hash_table.add(str(row))

        self.element_numb += len(keys)
        self.pos_numb += sum(label)

    def test_group(self, keys, score=None):
        # print("here leaf " + self.loc)
        test_result = np.zeros(len(keys))
        ss = 0
        str_keys = to_str_df(keys).values
        for entry in str_keys:
            if str(entry) in self.hash_table:
                test_result[ss] = 1
            ss += 1
        return test_result

    def clear(self):
        self.hash_table.clear()
        self.df_list = None
        self.multi_dic = None
        self.element_numb = 0

    def get_size(self, verbose=False):
        if self.element_numb == 0:
            return 0
        exp_bf_size = get_exp_bf_size(len(self.hash_table), MLBF_para["fpr"])
        if verbose:
            print("exp_bf_size=" + str(exp_bf_size))
        # return exp_bf_size TODO
        res = 0
        if MLBF_para["default_model"]["type"] == "GRU_RNN":
            tmp = self.df_list[0]["url"].values
            for item in tmp:
                res += len(item) * 8
        else:
            res += self.df_list[0].shape[0] * self.df_list[0].shape[1] * 32
        if self.multi_dic is not None and "boost_dic" in self.multi_dic and "scores" in self.multi_dic["boost_dic"]:
            res += 2 * 32 * len(self.multi_dic["boost_dic"]["scores"])
        return res

    def get_cache_size(self):
        return self.df_list[0].shape[0] * self.df_list[0].shape[1] * 32

    def output(self):
        if self.df_list is not None:
            [cur_keys, cur_label] = self.df_list
        else:
            cur_keys, cur_label = None, None
        cur_multi_dic = {"para_dic": {"name": self.loc, "static": False, "fpr": self.fpr}, "boost_dic": {}}
        if self.multi_dic is not None:
            cur_multi_dic.update(self.multi_dic)
        cur_multi_dic["para_dic"]["from_hash_node"] = True
        return cur_keys, cur_label, cur_multi_dic

    @staticmethod
    def to_inner_maybe(target):
        if type(target) is hash_node and (
                target.element_numb >= node_size or (MLBF_para["cheat"] and target.pos_numb >= node_size)):
            tmp_keys, tmp_label, tmp_multi_dic = target.output()
            return inner_node.to_bf(tmp_keys, tmp_label, tmp_multi_dic)
        else:
            return target


class inner_node(node):
    def __init__(self, loc):
        super().__init__(loc)
        self.inner_thresholds = None
        self.leaves_thresholds = None
        self.bf_list = []
        self.clf = None
        self.model_dic = None
        self.alpha = None
        self.fpr_list = []  # may not use

    def insert(self, keys, labels, multi_dic=None):
        # initialization
        # print("inserting " + self.loc)
        self.element_numb += keys.shape[0]
        if multi_dic is None:
            multi_dic = {"model_dic": MLBF_para["default_model"].copy(), "boost_dic": {}}

        if self.clf is None:
            # initial insert
            self.train_model(keys, labels, multi_dic)

            new_boost_dic = self.predict_score(keys, multi_dic["boost_dic"], labels)
            scores = new_boost_dic["scores"]
            [self.inner_thresholds, fpr_list] = get_parameter_vals(list(scores[labels]),
                                                                   list(scores[np.logical_not(labels)]),
                                                                   self.fpr, MLBF_para["partition"])
            if MLBF_para["split_num"] == MLBF_para["partition"]:
                self.fpr_list = fpr_list
            self.inner_thresholds[0] -= 1e-6
            self.inner_thresholds[-1] += 1e-6
            if len(fpr_list) == 1:  # build failed
                warnings.warn("build inner failed")
                return False

            new_multi_dic = {"para_dic": {"fpr": fpr_list}, "boost_dic": {}}
            if MLBF_para["boost"]:
                new_multi_dic["boost_dic"] = new_boost_dic
            self.build_local_backup_filters(keys, labels, scores, new_multi_dic)
            return True
        else:
            # append insert
            new_boost_dic = self.predict_score(keys, multi_dic["boost_dic"], labels)
            if not self.leaves:
                assert (MLBF_para["split"] == "uniform" or MLBF_para["split"] == "ada-BF")
                partition_num = MLBF_para["split_num"]
                cur_fpr_list = []
                if MLBF_para["split"] == "ada-BF" and partition_num > 1:
                    if MLBF_para["split_num"] == MLBF_para["partition"]:
                        self.leaves_thresholds, cur_fpr_list = self.inner_thresholds, self.fpr_list
                    else:
                        scores = new_boost_dic["scores"]
                        [self.leaves_thresholds, cur_fpr_list] = get_parameter_vals(list(scores[labels]),
                                                                                    list(
                                                                                        scores[np.logical_not(labels)]),
                                                                                    self.fpr, MLBF_para["split_num"])
                        self.leaves_thresholds[0] -= 1e-6
                        self.leaves_thresholds[-1] += 1e-6
                    partition_num = len(self.leaves_thresholds) - 1

                if MLBF_para["split"] == "uniform" or (not MLBF_para["advanced-dynamic-fpr"]):
                    cur_fpr_list = list(np.full(partition_num, MLBF_para["fpr"]))
                for j in range(partition_num):
                    cur_hash_node = hash_node(self.loc + str(j))
                    cur_hash_node.fpr = cur_fpr_list[j]
                    self.leaves.append(cur_hash_node)

            self.append_insert(keys, labels, new_boost_dic)
            return True

    def test_group(self, keys, boost_dic=None):
        # seems the model_dic here is used only for boosting
        if boost_dic is None:
            boost_dic = {}
        if not self.clf:
            warnings.warn("testing node without initialization")
            return np.zeros(len(keys))
        if keys.empty:
            warnings.warn("empty dataframe as input")
            return np.zeros(0)
        new_boost_dic = self.predict_score(keys, boost_dic, None)
        truths = self.test_local(keys, new_boost_dic)
        if self.leaves:
            retest_idx = np.logical_not(truths)
            indexes = self.get_leaves_idx(
                {"keys": keys, "scores": np.array(new_boost_dic["scores"]), "split": MLBF_para["split"],
                 "cur_thresholds": "leaves"})
            for cur_leaf, cur_idx in zip(self.leaves, indexes):
                cur_test_idx = np.logical_and(cur_idx, retest_idx)
                if sum(cur_test_idx) == 0:
                    continue
                truths[cur_test_idx] = inner_node.test_child(cur_leaf, keys, cur_test_idx, new_boost_dic)
        return truths

    def get_size(self, verbose=False):
        model_size = get_size(self.clf)
        leaf_size = 0
        bf_node_size = 0
        if self.bf_list:
            for bf in self.bf_list:
                bf_node_size += bf.get_size(verbose)  # False
        if self.leaves:
            for child in self.leaves:
                leaf_size += child.get_size(verbose)  # False
        sum_size = model_size + bf_node_size + leaf_size
        if verbose:
            print(self.loc + ' ' + "model_size=" + str(model_size) + ",bf_node_size=" + str(bf_node_size) +
                  ",leaf_size=" + str(leaf_size) + ",total_size=" + str(sum_size))
        return sum_size

    def get_cache_size(self):
        leaf_size = 0
        if self.leaves:
            for child in self.leaves:
                leaf_size += child.get_cache_size()
        return leaf_size

    '''
    "private functions" below
    '''

    def test_local(self, keys, boost_dic):
        result = np.zeros(len(keys))
        cur_score = boost_dic["scores"]
        if self.bf_list:
            indexes = self.get_leaves_idx(
                {"keys": keys, "scores": np.array(cur_score), "split": "ada-BF", "cur_thresholds": "inner"})
            for cur_bf, cur_idx in zip(self.bf_list, indexes):
                result[cur_idx] = inner_node.test_child(cur_bf, keys, cur_idx, boost_dic)
        return result

    def train_model(self, keys, labels, multi_dic):
        if MLBF_para["boost"]:
            weights = multi_dic["boost_dic"].get("weights", np.ones(len(keys)))
        else:
            weights = None
        model_dic = multi_dic["model_dic"]
        self.model_dic = model_dic
        self.clf = inner_node.train_model_core(keys, labels, self.model_dic, weights)

    @staticmethod
    def train_model_core(keys, labels, model_dic, weights=None):
        if model_dic["type"] == "tree":
            clf = DecisionTreeClassifier(max_depth=model_dic["max-depth"], random_state=1)
            clf.fit(X=keys, y=labels, sample_weight=weights)
        elif model_dic["type"] == "GRU_RNN":
            clf = GRUModel('dataset/glove.6B.50d-char2.txt', 50, learning_rate=rnn_para["learning_rate"],
                           pca_embedding_dim=rnn_para["pca_embedding_dim"],
                           maxlen=rnn_para["maxlen"],
                           gru_size=rnn_para["gru_size"], batch_size=rnn_para["batch_size"],
                           hidden_size=rnn_para["hidden_size"], second_gru_size=rnn_para["second_gru_size"],
                           decay=rnn_para["decay"], epochs=rnn_para["epochs"])

            neg_idx = np.logical_not(labels.values)

            new_keys = to_series_key(keys)
            shuffled = shuffle_for_training(new_keys[neg_idx], new_keys[labels])
            clf.fit(shuffled[0], shuffled[1], cur_weight=weights)
        elif model_dic["type"] == "xgboost":
            clf = XGBClassifier(n_estimators=model_dic["n-estimators"], max_depth=model_dic["max-depth"],
                                random_state=1)
            clf.fit(X=keys, y=labels, sample_weight=weights)  # , eval_metric='l1', early_stopping_rounds=5
        elif model_dic["type"] == "lgbm":
            clf = LGBMClassifier(num_leaves=model_dic["num_leaves"], learning_rate=0.05,
                                 n_estimators=model_dic["n-estimators"], n_jobs=4)
            clf.fit(X=keys, y=labels, sample_weight=weights)
        elif model_dic["type"] == "mlp":
            clf = MLPClassifier(hidden_layer_sizes=model_dic["hidden-layer-sizes"], activation="relu",
                                max_iter=1000)
            clf.fit(X=keys, y=labels)
        elif model_dic["type"] == "zipf":
            clf = ZipfClassifier()
        elif model_dic["type"] == "svm":
            # self.clf = svm.SVC(kernel="rbf", probability=True, class_weight="balanced")
            clf = LogisticRegression()
            clf.fit(X=keys, y=labels)
        else:
            print("unexpected model type")
            exit(0)

        return clf

    def predict_score(self, keys, boost_dic, labels=None):
        # NOTE: the alpha in inner node will also be initialized in this function
        cur_score = inner_node.get_score_with_clf(self.clf, keys, self.model_dic["type"])
        if MLBF_para["boost"]:
            binary_predict_res = [1 if s >= 0.5 else 0 for s in cur_score]

            if labels is not None:  # using in inserting process
                weights = boost_dic.get("weights", np.ones(len(keys)))
                tmp_alpha, new_weights = update_weight(binary_predict_res, labels, weights)
                if self.alpha is None:
                    self.alpha = tmp_alpha

            if "alpha_sum" in boost_dic:
                new_alpha, new_score = update_score(boost_dic["alpha_sum"], self.alpha, boost_dic["scores"], cur_score)
            else:
                new_alpha, new_score = self.alpha, cur_score

            new_boost_dic = {"scores": new_score, "alpha_sum": new_alpha}
            if labels is not None:
                new_boost_dic["weights"] = new_weights
        else:
            new_boost_dic = {"scores": cur_score}
        return new_boost_dic

    @staticmethod
    def get_score_with_clf(clf, keys, model_type):
        cur_score = None
        if model_type in ["tree", "xgboost", "mlp", "zipf", "svm", "lgbm"]:
            cur_score = fetch_pos_score(clf.predict_proba(keys))
        elif model_type == "GRU_RNN":
            new_keys = to_series_key(keys)
            cur_score = clf.predicts(new_keys)
        else:
            print("unexpected model type")
            exit(0)
        assert (cur_score is not None)
        cur_score = np.array(cur_score)
        return cur_score

    def build_local_backup_filters(self, keys, labels, scores, multi_dic):
        fpr_list = multi_dic["para_dic"]["fpr"]
        boost_dic = multi_dic["boost_dic"]
        indexes = self.get_leaves_idx({"keys": keys, "scores": scores, "split": "ada-BF", "cur_thresholds": "inner"})
        for cur_idx, cur_fpr in zip(indexes, fpr_list):
            cur_child_name = self.loc + str(chr(ord('a') + len(self.bf_list)))
            if MLBF_para["inner_multi_layer"]:
                cur_para_dic = {"name": cur_child_name, "fpr": cur_fpr, "static": True}
                cur_boost_dic = inner_node.split_boost_dic(boost_dic, cur_idx) if MLBF_para["boost"] else {}
                cur_node = inner_node.to_bf(keys[cur_idx], labels[cur_idx],
                                            {"para_dic": cur_para_dic, "boost_dic": cur_boost_dic})
            else:
                cur_pos_idx = np.logical_and(labels.values, cur_idx)
                cur_node = inner_node.build_bf_node(cur_child_name, keys[cur_pos_idx], cur_fpr)
            self.bf_list.append(cur_node)

    def append_insert(self, keys, labels, boost_dic):
        cur_score = boost_dic["scores"]
        # generate index
        indexes = self.get_leaves_idx(
            {"keys": keys, "scores": cur_score, "split": MLBF_para["split"], "cur_thresholds": "leaves"})
        for j in range(len(indexes)):
            cur_child_multi_dic = {"model_dic": MLBF_para["default_model"].copy(),
                                   "boost_dic": inner_node.split_boost_dic(boost_dic, indexes[j]) if MLBF_para[
                                       "boost"] else {}}
            self.leaves[j].insert(keys[indexes[j]], labels[indexes[j]], cur_child_multi_dic)
            self.leaves[j] = hash_node.to_inner_maybe(self.leaves[j])

    '''
    "utils functions" below
    '''

    @staticmethod
    def to_bf(keys, labels, multi_dic):
        # para_dic: {"name":str name of node, weights, scores, alpha_sum,
        #           "static":True/False}
        if labels is None or len(labels) == 0:
            cur_node = bf_node(multi_dic["para_dic"]["name"])
            return cur_node
        para_dic = multi_dic["para_dic"]
        cur_pos_key_num = sum(labels)
        cur_fpr = para_dic["fpr"]
        cur_pos_ratio = cur_pos_key_num / len(labels)
        extreme_pos_ratio_flag = MLBF_para["avoid_extreme_pos_ratio"] and para_dic["static"] and (
                cur_pos_ratio > 0.95 or cur_pos_ratio < 0.05)
        if (("from_hash_node" not in multi_dic["para_dic"]) and len(multi_dic["para_dic"]["name"]) > MLBF_para["max-layer-num"]) or len(
                labels) <= MIN_INNER_SIZE or extreme_pos_ratio_flag:  # TODO
            cur_node = inner_node.build_bf_node(para_dic["name"], keys[labels], cur_fpr)
        else:
            exp_bf_size = get_exp_bf_size(cur_pos_key_num, cur_fpr)
            candidate_model_list = MLBF_para["candidate_models"].copy() \
                if para_dic["static"] else [MLBF_para["default_model"].copy()]
            cur_node = None
            cur_size = exp_bf_size
            for candidate_model in candidate_model_list:
                tmp_node = inner_node(para_dic["name"])
                tmp_node.fpr = cur_fpr
                cur_multi_dic = {"model_dic": candidate_model.copy(), "boost_dic": {}}
                if MLBF_para["boost"]:
                    cur_multi_dic["boost_dic"] = multi_dic["boost_dic"]
                if not tmp_node.insert(keys, labels, cur_multi_dic):
                    continue
                tmp_size = tmp_node.get_size(False)
                if tmp_size <= cur_size or (MLBF_para["optimize_space"]):  # TODO
                    cur_node, cur_size = tmp_node, tmp_size
            if cur_node is None or (cur_size >= exp_bf_size and para_dic["static"]):
                cur_node = inner_node.build_bf_node(para_dic["name"], keys[labels], cur_fpr)
        return cur_node

    def get_leaves_idx(self, dic):
        indexes = []
        if dic["split"] == "ada-BF":
            assert (dic["cur_thresholds"] in ["inner", "leaves"])
            cur_score = dic["scores"]
            if dic["cur_thresholds"] == "inner":
                for j in range(len(self.inner_thresholds) - 1):
                    indexes.append((self.inner_thresholds[j] <= cur_score) & (cur_score < self.inner_thresholds[j + 1]))
            else:
                for j in range(len(self.leaves_thresholds) - 1):
                    indexes.append(
                        (self.leaves_thresholds[j] <= cur_score) & (cur_score < self.leaves_thresholds[j + 1]))
        elif dic["split"] == "uniform":
            keys = dic["keys"]
            if MLBF_para["split_num"] == 1:
                indexes = [np.full(len(dic["scores"]), True)]
            else:
                split_arr = to_binary_arr(keys, len(self.loc), MLBF_para["split_num"])
                for j in range(MLBF_para["split_num"]):
                    indexes.append(split_arr == j)
        else:
            print("unknown partition method")

        return indexes

    @staticmethod
    def build_bf_node(name_str, cur_keys, fpr):
        cur_bf_node = bf_node(name_str, fpr=fpr)
        cur_bf_node.insert(keys=cur_keys)
        return cur_bf_node

    @staticmethod
    def test_child(child, keys, retest_idx, boost_dic):
        new_boost_dic = inner_node.split_boost_dic(boost_dic, retest_idx) if MLBF_para["boost"] else None
        leaf_truths = child.test_group(keys[retest_idx], new_boost_dic)
        return leaf_truths

    @staticmethod
    def split_boost_dic(raw_dic, idx=None):
        res_dic = {}
        if "scores" in raw_dic:
            res_dic["scores"] = raw_dic["scores"][idx] if idx is not None else raw_dic["scores"]
        if "alpha_sum" in raw_dic:
            res_dic["alpha_sum"] = raw_dic["alpha_sum"]
        if "weights" in raw_dic:
            res_dic["weights"] = raw_dic["weights"][idx] if idx is not None else raw_dic["weights"]
        return res_dic


class ca_lbf(node):
    def __init__(self, loc):
        super().__init__(loc)
        self.model_list = []
        self.threshold_list = []
        self.bf = None
        self.hash_node = None

    def insert(self, keys, labels, multi_dic=None):
        # initialization
        # print("inserting " + self.loc)
        self.element_numb += keys.shape[0]
        self.pos_numb += sum(labels)
        if self.bf is None:
            self.bf = bf_node("r", MLBF_para["fpr"] / 2)
            self.hash_node = hash_node("tmp")

        if len(self.model_list) > 0:
            self.hash_node.insert(keys, labels, multi_dic)
            if self.hash_node.element_numb > node_size:
                keys, labels, multi_dic = self.hash_node.output()
                self.hash_node.clear()
                self.insert_core(keys, labels, multi_dic)
        else:
            self.insert_core(keys, labels, multi_dic)

    def insert_core(self, keys, labels, multi_dic):
        weights = np.ones(len(keys)) if MLBF_para["boost"] else None
        cur_clf = inner_node.train_model_core(keys, labels, MLBF_para["default_model"].copy(), weights)
        cur_scores = inner_node.get_score_with_clf(cur_clf, keys, MLBF_para["default_model"]["type"])
        cur_threshold = ca_lbf.get_lbf_threshold(labels, cur_scores, MLBF_para["fpr"] / 2)
        cur_idx = np.logical_and(labels, cur_scores <= cur_threshold)
        # cur_idx = np.logical_and(labels, cur_scores < cur_threshold)
        self.bf.local_insert(keys, cur_idx)
        self.threshold_list.append(cur_threshold)
        self.model_list.append(cur_clf)

    def test_group(self, keys, boost_dic=None):

        cur_res = np.zeros(keys.shape[0])
        for cur_clf, cur_thr in zip(self.model_list, self.threshold_list):
            cur_score = inner_node.get_score_with_clf(cur_clf, keys, MLBF_para["default_model"]["type"])
            cur_truth_idx = cur_score >= cur_thr
            cur_res = np.logical_or(cur_res, cur_truth_idx)

        retest_idx = np.logical_not(cur_res)
        cur_res[retest_idx] = self.bf.test_group(keys[retest_idx])

        retest_idx = np.logical_not(cur_res)
        if not (self.hash_node.pos_numb == 0 and self.hash_node.element_numb == 0 and (not self.hash_node.df_list)):
            cur_res[retest_idx] = self.hash_node.test_group(keys[retest_idx])

        return cur_res

    def get_size(self, verbose=False):
        model_size = 0
        bf_size = 0
        hash_size = 0
        for cur_clf in self.model_list:
            model_size += get_size(cur_clf)
        bf_size = self.bf.get_size(False)
        hash_size = self.hash_node.get_size()
        total_size = model_size + bf_size + hash_size
        if verbose:
            print("model_size: " + str(model_size) + " bf_size: " + str(bf_size) + " hash_size: " + str(hash_size) +
                  " total: " + str(total_size))
        return total_size

    @staticmethod
    def get_lbf_threshold(labels, scores, target_fpr):
        non_key_idx = np.logical_not(labels)
        neg_scores = scores[non_key_idx].tolist()
        neg_scores.sort()
        cur_threshold_idx = max(1, int(len(neg_scores) * target_fpr))
        locate_score = neg_scores[len(neg_scores) - cur_threshold_idx]
        # return locate_score

        neg_scores_deduplicated = list(set(neg_scores))
        neg_scores_deduplicated.sort()
        deduplicated_threshold_idx = neg_scores_deduplicated.index(locate_score)
        if deduplicated_threshold_idx == len(neg_scores_deduplicated) - 1:
            # warnings.warn("no negatives are ignored")
            return 1.0
        else:
            return neg_scores_deduplicated[deduplicated_threshold_idx + 1]


########################################################################################################################

def initialize_clf_and_shape_paras(mode, dataset_name):
    global MLBF_para
    assert (mode in ["static", "dynamic", "link"])
    MLBF_para["inner_multi_layer"] = mode == "static"
    MLBF_para["optimize_space"] = mode == "static"
    MLBF_para["avoid_extreme_pos_ratio"] = mode == "static"
    if mode == "link":
        MLBF_para["split_num"] = 1
    xgb_para_dic_1 = {"type": "xgboost", "n-estimators": 3, "max-depth": 1}
    xgb_para_dic_2 = {"type": "xgboost", "n-estimators": 35, "max-depth": 5}
    xgb_para_dic_3 = {"type": "xgboost", "n-estimators": 10, "max-depth": 3}
    xgb_para_dic_4 = {"type": "xgboost", "n-estimators": 7, "max-depth": 1}
    xgb_para_dic_5 = {"type": "xgboost", "n-estimators": 5, "max-depth": 1}
    lgbm_para_dic = {"type": "lgbm", "n-estimators": 5, "num_leaves": 8}
    tree_para_dic_1 = {"type": "tree", "max-depth": 7}
    tree_para_dic_2 = {"type": "tree", "max-depth": 13}
    tree_para_dic_3 = {"type": "tree", "max-depth": 11}
    gru_para_dic = {"type": "GRU_RNN"}
    svm_para_dic = {"type": "svm"}
    mlp_para_dic = {"type": "mlp", "hidden-layer-sizes": (50, 50)}
    mlp_para_dic_2 = {"type": "mlp", "hidden-layer-sizes": (10, 10)}
    possible_candidate_list = None
    if dataset_name == "facebook":
        # MLBF_para["default_model"] = tree_para_dic_1
        MLBF_para["default_model"] = tree_para_dic_3
        possible_candidate_list = [tree_para_dic_1, tree_para_dic_2, xgb_para_dic_2]
    elif dataset_name == "url":
        MLBF_para["default_model"] = gru_para_dic
        possible_candidate_list = MLBF_para["default_model"]
    elif dataset_name == "higgs":
        MLBF_para["default_model"] = xgb_para_dic_3
        possible_candidate_list = [xgb_para_dic_2, tree_para_dic_2]
    elif dataset_name == "train1000":
        # MLBF_para["default_model"] = xgb_para_dic_1  # lgbm_para_dic #
        MLBF_para["default_model"] = xgb_para_dic_1
        # MLBF_para["default_model"] = lgbm_para_dic
        # xgb_para_dic_4  # tree_para_dic_2#xgb_para_dic_1 xgb_para_dic_3 tree_para_dic_1
        possible_candidate_list = [xgb_para_dic_2, tree_para_dic_2]  # xgb_para_dic_4   #tree_para_dic_2
    elif dataset_name == "letor":
        MLBF_para["default_model"] = xgb_para_dic_2
        possible_candidate_list = [xgb_para_dic_1, xgb_para_dic_2, tree_para_dic_1]
    elif dataset_name == "amazon":
        MLBF_para["default_model"] = xgb_para_dic_2
        possible_candidate_list = [xgb_para_dic_1, xgb_para_dic_2, tree_para_dic_1]
    elif dataset_name == "zipf":
        MLBF_para["default_model"] = {"type": "zipf"}
        possible_candidate_list = [{"type": "zipf"}]
        # TODO
    else:
        warnings.warn("unexpected dataset")
        exit(0)

    MLBF_para["candidate_models"] = possible_candidate_list if MLBF_para["multiple_candidates"] else \
        [MLBF_para["default_model"]]


def test_single(cur_node, cur_train, cur_test, key_cols, cur_res):
    time_start = time.perf_counter()
    cur_node.insert(cur_train[key_cols], cur_train["label"])
    time_end = time.perf_counter()
    cur_res["insert_time"].append(time_end - time_start)

    time_start = time.perf_counter()
    query_res = cur_node.test_group(cur_test[key_cols])
    time_end = time.perf_counter()
    cur_res["query_time"].append(time_end - time_start)
    cur_res["fpr"].append((sum(query_res) / len(query_res)))

    cur_res["space"].append(cur_node.get_size(False))
    cur_res["cache_size"].append(cur_node.get_cache_size())


def static_test(cur_train, cur_test, key_cols, use_boost=True):
    global MLBF_para
    # template_dic = {"insert_time": [], "query_time": [], "fpr": [], "space": []}
    res_dics = {"boost": {"insert_time": [], "query_time": [], "fpr": [], "space": [], "cache_size": []},
                "noboost": {"insert_time": [], "query_time": [], "fpr": [], "space": [], "cache_size": []},
                "plbf": {"insert_time": [], "query_time": [], "fpr": [], "space": [], "cache_size": []},
                "bf": {"insert_time": [], "query_time": [], "fpr": [], "space": [], "cache_size": []},
                "LBF": {"insert_time": [], "query_time": [], "fpr": [], "space": [], "cache_size": []}
                }
    for cur_data_idx in trange(len(cur_train)):
        cur_node = ca_lbf('r')
        test_single(cur_node, cur_train[cur_data_idx], cur_test, key_cols, res_dics["LBF"])

        if use_boost:
            MLBF_para["boost"] = False
            MLBF_para["max-layer-num"] = 3
            cur_node = inner_node("r")
            test_single(cur_node, cur_train[cur_data_idx], cur_test, key_cols, res_dics["boost"])

        MLBF_para["max-layer-num"] = 3
        MLBF_para["boost"] = False
        cur_node = inner_node("r")
        test_single(cur_node, cur_train[cur_data_idx], cur_test, key_cols, res_dics["noboost"])

        MLBF_para["max-layer-num"] = 1
        cur_node = inner_node("r")
        test_single(cur_node, cur_train[cur_data_idx], cur_test, key_cols, res_dics["plbf"])

        cur_node = bf_node('r')
        test_single(cur_node, cur_train[cur_data_idx], cur_test, key_cols, res_dics["bf"])

    return res_dics


def dynamic_test2(cur_train, cur_test, key_cols, use_boost=True):
    global MLBF_para
    res_dics = {
        "ca-lbf": {"insert_time": [], "query_time": [], "fpr": [], "space": []}
    }
    MLBF_para["cheat"] = False
    for cur_data_idx in trange(len(cur_train)):
        cur_node = ca_lbf("r")
        test_single(cur_node, cur_train[cur_data_idx], cur_test, key_cols, res_dics["ca-lbf"])
    return res_dics


def dynamic_test(cur_train, cur_test, key_cols, use_boost=True):
    global MLBF_para
    # template_dic = {"insert_time": [], "query_time": [], "fpr": [], "space": []}
    res_dics = {
        "boost4": {"insert_time": [], "query_time": [], "fpr": [], "space": [], "cache_size": []},
        "noboost4": {"insert_time": [], "query_time": [], "fpr": [], "space": [], "cache_size": []},
        "boost2": {"insert_time": [], "query_time": [], "fpr": [], "space": [], "cache_size": []},
        "noboost2": {"insert_time": [], "query_time": [], "fpr": [], "space": [], "cache_size": []},
        "uniboost4": {"insert_time": [], "query_time": [], "fpr": [], "space": [], "cache_size": []},
        "uniboost2": {"insert_time": [], "query_time": [], "fpr": [], "space": [], "cache_size": []},
        "noboostlink": {"insert_time": [], "query_time": [], "fpr": [], "space": [], "cache_size": []},
        "boostlink": {"insert_time": [], "query_time": [], "fpr": [], "space": [], "cache_size": []}
    }

    MLBF_para["max-layer-num"] = 1
    # boost cheat partition-method link
    MLBF_para["cheat"] = False

    MLBF_para["split_num"] = 4
    MLBF_para["split"] = "ada-BF"#
    if use_boost:
        MLBF_para["boost"] = True
        cur_node = inner_node("r")
        for cur_data_idx in trange(len(cur_train)):
            test_single(cur_node, cur_train[cur_data_idx], cur_test, key_cols, res_dics["boost4"])
    '''
    MLBF_para["boost"] = False
    cur_node = inner_node("r")
    for cur_data_idx in trange(len(cur_train)):
        test_single(cur_node, cur_train[cur_data_idx], cur_test, key_cols, res_dics["noboost4"])
    '''

    MLBF_para["split_num"] = 2
    MLBF_para["split"] = "ada-BF"#
    if use_boost:
        MLBF_para["boost"] = True
        cur_node = inner_node("r")
        for cur_data_idx in trange(len(cur_train)):
            test_single(cur_node, cur_train[cur_data_idx], cur_test, key_cols, res_dics["boost2"])

    '''
    MLBF_para["boost"] = False
    cur_node = inner_node("r")
    for cur_data_idx in trange(len(cur_train)):
        test_single(cur_node, cur_train[cur_data_idx], cur_test, key_cols, res_dics["noboost2"])
    '''
    MLBF_para["split"] = "uniform"
    MLBF_para["split_num"] = 1
    if use_boost:
        MLBF_para["boost"] = True
        cur_node = inner_node("r")
        for cur_data_idx in trange(len(cur_train)):
            test_single(cur_node, cur_train[cur_data_idx], cur_test, key_cols, res_dics["boostlink"])
    '''
    MLBF_para["boost"] = False
    cur_node = inner_node("r")
    for cur_data_idx in trange(len(cur_train)):
        test_single(cur_node, cur_train[cur_data_idx], cur_test, key_cols, res_dics["noboostlink"])

    
    MLBF_para["split_num"] = 4
    MLBF_para["split"] = "uniform"
    if use_boost:
        MLBF_para["boost"] = True
        cur_node = inner_node("r")
        for cur_data_idx in trange(len(cur_train)):
            test_single(cur_node, cur_train[cur_data_idx], cur_test, key_cols, res_dics["uniboost4"])

    MLBF_para["split_num"] = 2
    MLBF_para["boost"] = False
    cur_node = inner_node('r')
    for cur_data_idx in trange(len(cur_train)):
        test_single(cur_node, cur_train[cur_data_idx], cur_test, key_cols, res_dics["uniboost2"])
    '''
    return res_dics


def uni_test(cur_train, cur_test, key_cols, use_boost=True):
    global MLBF_para
    res_dics = {
        "boost4": {"insert_time": [], "query_time": [], "fpr": [], "space": [], "cache_size": []},
        "boost2": {"insert_time": [], "query_time": [], "fpr": [], "space": [], "cache_size": []}}
    MLBF_para["split"] = "ada-BF"
    MLBF_para["split_num"] = 2
    if use_boost:
        MLBF_para["boost"] = True
        cur_node = inner_node("r")
        for cur_data_idx in trange(len(cur_train)):
            test_single(cur_node, cur_train[cur_data_idx], cur_test, key_cols, res_dics["boost2"])

    MLBF_para["split"] = "ada-BF"
    MLBF_para["split_num"] = 4
    if use_boost:
        MLBF_para["boost"] = True
        cur_node = inner_node("r")
        for cur_data_idx in trange(len(cur_train)):
            test_single(cur_node, cur_train[cur_data_idx], cur_test, key_cols, res_dics["boost4"])
    return res_dics


def partition_test(cur_train, cur_test, key_cols, use_boost):
    global MLBF_para
    res_dics = {"noboost": {"insert_time": [], "query_time": [], "fpr": [], "space": []},
                "boost": {"insert_time": [], "query_time": [], "fpr": [], "space": []}}
    cur_paras = []
    for i in tqdm(range(2, 10)):
        cur_paras.append(i)
        MLBF_para["partition"] = i
        MLBF_para["boost"] = False
        cur_node = inner_node("r")
        test_single(cur_node, cur_train, cur_test, key_cols, res_dics["noboost"])
        if MLBF_para["default_model"]["type"] == "GRU_RNN":
            res_dics["boost"] = res_dics["noboost"]
        else:
            MLBF_para["boost"] = True
            cur_node = inner_node("r")
            test_single(cur_node, cur_train, cur_test, key_cols, res_dics["boost"])
    res_dics["para"] = cur_paras
    return res_dics


def fpr_test(cur_train, cur_test, key_cols, use_boost):
    global MLBF_para
    res_dics = {"SMLBF": {"insert_time": [], "query_time": [], "fpr": [], "space": [], "cache_size": []},
                "PLBF": {"insert_time": [], "query_time": [], "fpr": [], "space": [], "cache_size": []},
                "LBF": {"insert_time": [], "query_time": [], "fpr": [], "space": [], "cache_size": []},
                "BF": {"insert_time": [], "query_time": [], "fpr": [], "space": [], "cache_size": []}}
    point_num = 100
    cur_paras = np.linspace(0.0001, 0.01, point_num).tolist()
    MLBF_para["boost"] = False
    MLBF_para["max-layer-num"] = 3
    MLBF_para["partition"] = 6

    for i in tqdm(range(0, point_num)):
        MLBF_para["fpr"] = cur_paras[i]
        cur_node = inner_node("r")
        test_single(cur_node, cur_train, cur_test, key_cols, res_dics["SMLBF"])
    MLBF_para["max-layer-num"] = 1
    for i in tqdm(range(0, point_num)):
        MLBF_para["fpr"] = cur_paras[i]
        cur_node = inner_node("r")
        test_single(cur_node, cur_train, cur_test, key_cols, res_dics["PLBF"])
    for i in tqdm(range(0, point_num)):
        MLBF_para["fpr"] = cur_paras[i]
        cur_node = ca_lbf("r")
        test_single(cur_node, cur_train, cur_test, key_cols, res_dics["LBF"])

    for i in tqdm(range(0, point_num)):
        MLBF_para["fpr"] = cur_paras[i]
        cur_node = bf_node("r")
        cur_node.fpr = cur_paras[i]
        test_single(cur_node, cur_train, cur_test, key_cols, res_dics["BF"])
    res_dics["para"] = cur_paras
    return res_dics


def max_layer_test(cur_train, cur_test, key_cols, use_boost):
    global MLBF_para
    res_dics = {"noboost": {"insert_time": [], "query_time": [], "fpr": [], "space": []},
                "boost": {"insert_time": [], "query_time": [], "fpr": [], "space": []}}
    cur_paras = []
    for i in tqdm(range(1, 8)):
        cur_paras.append(i)
        MLBF_para["max-layer-num"] = i
        MLBF_para["boost"] = False
        cur_node = inner_node("r")
        test_single(cur_node, cur_train, cur_test, key_cols, res_dics["noboost"])
        if MLBF_para["default_model"]["type"] == "GRU_RNN":
            res_dics["boost"] = res_dics["noboost"]
        else:
            MLBF_para["boost"] = True
            cur_node = inner_node("r")
            test_single(cur_node, cur_train, cur_test, key_cols, res_dics["boost"])
    res_dics["para"] = cur_paras
    return res_dics


def test_fpr(dataset_name, data_len, mode, test_pos=False, draw_line=False):
    initialize_clf_and_shape_paras(mode, dataset_name)
    half_len = int(data_len / 2)
    key_cols, data = load_data_from_csv(dataset_name, data_len)
    train_data = data[0:half_len]
    tst_data = data[half_len:data_len]
    pos_data = train_data.loc[data["label"]]
    neg_data = train_data.loc[data["label"] == False]
    tst_pos = tst_data.loc[data["label"]]
    tst_neg = tst_data.loc[data["label"] == False]
    dbf = inner_node("r")
    if draw_line:
        train_key_list = []
        pos_key_num_list = []
        key_num_list = []
        for i in range(int(train_data.shape[0] / batch_size)):
            cur_start = 0 if mode == "static" else batch_size * i
            train_key_list.append(train_data[cur_start:batch_size * (i + 1)])
            pos_key_num_list.append(sum(train_data["label"][cur_start:batch_size * (i + 1)]))
            key_num_list.append(batch_size * (i + 1))
        if mode == "static":
            res_dics = static_test(train_key_list, tst_neg, key_cols, True)
            # res_dics = max_layer_test(train_key_list[-1], neg_data, key_cols, True)
            # res_dics = fpr_test(train_key_list[-1], tst_neg, key_cols, True)
        elif mode == "dynamic":
            res_dics = dynamic_test(train_key_list, tst_neg, key_cols, True)
            # res_dics = uni_test(train_key_list, neg_data, key_cols, True)
        else:
            res_dics = {}
        res_dics.update({"pos_key_num_list": pos_key_num_list, "key_num_list": key_num_list})
        res_dics.update({"Paras": MLBF_para})
        res_dics.update({"node_size": node_size, "batch_size": batch_size, "MIN_Innersize": MIN_INNER_SIZE})
        f = open("resultN/" + "new_multi-test-boost" + mode + '_' +
                 ("" if mode == "static" else ("no" if not MLBF_para["cheat"] else "") + "cheat")
                 + '_' + MLBF_para["default_model"]["type"] + '_' + dataset_name + '_' + str(data_len) + ".txt", "a")
        f.write(str(res_dics))
        f.close()
    else:
        dbf = ca_lbf("r")  # ~TODO
        # MLBF_para["fpr"]=0.020
        # MLBF_para["max-layer-num"] = 1
        # dbf = bf_node("r")
        # dbf.fpr = 0.011
        if mode == "static":
            dbf.insert(train_data[key_cols], train_data["label"])
        else:
            train_key_list = []
            for i in range(int(train_data.shape[0] / batch_size)):
                train_key_list.append(train_data[batch_size * i:batch_size * (i + 1)])
            for new_data in train_key_list:
                dbf.insert(new_data[key_cols], new_data["label"])
            dbf.finalize_insert()
        print("####################################insert finished##############################################")
        dbf.get_size(True)

        # print("exp plbf size:" + str(get_exp_plbf_size(dbf, True)))
        print("exp bf size:" + str(get_exp_bf_size(sum(train_data["label"]), MLBF_para["fpr"])))
        print("key num: " + str(len(pos_data[key_cols])))
        if test_pos:
            res0 = dbf.test_group(pos_data[key_cols])
            print("pos test fpr: " + str(sum(res0) / len(res0)))
            res3 = dbf.test_group(tst_pos[key_cols])
            print("new pos test fpr: " + str(sum(res3) / len(res3)))
        res2 = dbf.test_group(neg_data[key_cols])
        print("neg test fpr: " + str(sum(res2) / len(res2)))
        res4 = dbf.test_group(tst_neg[key_cols])
        print("new neg test fpr: " + str(sum(res4) / len(res4)))


def get_exp_plbf_size(target, verbose=False):
    sizes = 0
    for cur_bf_node in target.bf_list:
        cur_bf_size = get_exp_bf_size(cur_bf_node.pos_numb, cur_bf_node.fpr)
        sizes += cur_bf_size
        if verbose:
            print("node: " + cur_bf_node.loc + ' ' + str(cur_bf_size))
    sizes += get_size(target.clf)
    return sizes


if __name__ == '__main__':
    # global MLBF_para
    set_rand_seed(24)  # 24 [42] [13]

    # test_fpr("url", 400000, "static", test_pos=False, draw_line=False)
    # test_fpr("higgs", 10500000, "static")
    # test_fpr("higgs", 800000, "static")

    # test_fpr("facebook", 800000, "static", test_pos=False, draw_line=True)
    # test_fpr("train1000", 800000, "static", test_pos=False, draw_line=True)
    # test_fpr("url", 400000, "static", test_pos=True, draw_line=True)

    # test_fpr("facebook", 800000, "dynamic", test_pos=False, draw_line=True)
    # test_fpr("train1000", 20000, "dynamic", test_pos=False, draw_line=True)
    test_fpr("url", 400000, "dynamic", test_pos=True, draw_line=True)

    # test_fpr("url", 400000, "static", test_pos=False, draw_line=True)
    # test_fpr("train1000", 800000, "static", test_pos=False, draw_line=False)
    #
    # MLBF_para["partition"]=4
    # test_fpr("url", 400000, "static", test_pos=False, draw_line=True)
    # test_fpr("facebook", 800000, "static", test_pos=False, draw_line=False)
    # test_fpr("train1000", 800000, "static", test_pos=False, draw_line=False)
    # test_fpr("train1000", 800000, "static", test_pos=False, draw_line=True)

    # test_fpr("url", 800000, "static")
# train1000[10,8,6,6,4,4,4]
# {rabbbbbbbb,rbbabbbbbb/rccbbccc/reeade/rfda/reda/reda}
# facebook[7,2,2,3,2,3,2]
# url[2,2,2,2,2,2,2]
# {raaaaaaa,ra}
'''
boost
total_size=317731
79971
pos test:1.0
^&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
neg test:0.003748641117594872
^&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
new neg test:0.008087774294670846
'''

'''
no boost
total_size=276219
79971
pos test:1.0
^&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
neg test:0.0030988766572117607
^&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
new neg test:0.007962382445141065

'''

'''
dynamic no boost
total_size=2547074
key num: 200313
pos test fpr: 1.0
^&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
neg test fpr: 0.029516192841797413
^&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
new pos test fpr: 0.38397417582692994
^&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
new neg test fpr: 0.0359406873918313

'''

'''
dynamic boost 2765095

key num: 200313
pos test fpr: 1.0
^&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
neg test fpr: 0.026922133138361535
^&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
new pos test fpr: 0.34601985935048596
^&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
new neg test fpr: 0.02776101388050694


dynamic no boost 2682463
key num: 200313
pos test fpr: 1.0
^&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
neg test fpr: 0.025554993564929115
^&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
new pos test fpr: 0.330977479035403
^&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
new neg test fpr: 0.027277216117946902


ada-bf no boost 1174570
pos test fpr: 1.0
^&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
neg test fpr: 0.05292282421990415
^&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
new pos test fpr: 0.8029302817501491
^&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
new neg test fpr: 0.053666638403567134

824294
>>>>test1 finished
>>>>test2 finished
>>>>test3 finished
key num: 200313
pos test fpr: 1.0
^&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
neg test fpr: 0.05446523809762278
^&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
new pos test fpr: 0.8532052149591736
^&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
new neg test fpr: 0.05550207733781553

'''

'''
new boost link
total_size=719516
key num: 200313
neg test fpr: 0.06565775438561348
new neg test fpr: 0.06589126021835738


new no boost link 
model_size=7152,bf_node_size=21097,leaf_size=2046522,total_size=2074771
key num: 200313
neg test fpr: 0.1868824710672202
new neg test fpr: 0.1865713701451892
'''

'''
boost link nocheat
insert time: [0.043336299999999994, 0.0948032000000012, 0.08409130000000076, 0.08494029999999952, 0.09070199999999318, 0.11323740000000271, 0.12706489999999349, 0.13422790000001328, 0.13514539999999897, 0.12496000000004415, 0.11116900000001806, 0.11874030000001312, 0.17786999999998443, 0.15674609999996392, 0.14512179999996988, 0.1396158000000014, 0.16064129999995203, 0.15390240000010635, 0.14820609999992485, 0.19449600000007194, 0.20751520000021628, 0.16838510000002316, 0.16894589999992604, 0.17245679999996355, 0.1936955000001035, 0.2071049999999559, 0.18626020000010612, 0.2008082000002105, 0.2044301999999334, 0.19739389999995183, 0.20710929999995642, 0.2637686000002759, 0.22824900000023263, 0.23107209999943734, 0.22302870000021358, 0.25838800000019546, 0.25851539999985107, 0.31088349999936327, 0.28165459999945597, 0.2700972000002366, 0.26360939999995026, 0.2675651999998081, 0.3067991999996593, 0.2699764999997569, 0.3068189000005077, 0.3057695000006788, 0.272368899999492, 0.3316511000011815, 0.3242352999986906, 0.29899719999957597, 0.2858747000009316, 0.33474630000091565, 0.31207309999990684, 0.315311099999235, 0.3863473999990674, 0.32481860000007146, 0.31039330000021437, 0.34970139999859384, 0.3390087999996467, 0.33133619999898656, 0.34032209999895713, 0.4238451000001078, 0.38695089999964694, 0.38050920000023325, 0.37497330000041984, 0.3982226999978593, 0.368726600001537, 0.403530799998407, 0.42204950000086683, 0.43465629999991506, 0.4370494999966468, 0.4069452000003366, 0.4791454000005615, 0.43370770000183256, 0.4277541999981622, 0.44838999999774387, 0.4338921000016853, 0.5013107000013406, 0.4554205999993428, 0.4305400999983249]
query time: [5.6126173999999995, 12.4005069, 19.1112016, 26.689820899999994, 34.3025868, 42.0679313, 49.0231091, 57.413870599999996, 65.12244529999998, 71.47423509999999, 79.4145517, 86.53513120000002, 94.43689589999997, 101.34273519999999, 112.49477640000009, 128.6114422999999, 135.67125669999996, 145.3636535999999, 153.98672760000022, 159.67328739999994, 169.57628439999985, 175.50047489999997, 178.07286850000014, 199.20551239999986, 202.67551349999985, 205.14646519999997, 203.56354850000025, 209.92439660000036, 215.38070219999963, 223.82890849999967, 231.63232400000015, 253.08123290000003, 274.6960462000002, 267.33902450000005, 284.65838859999985, 289.81059710000045, 287.80084389999956, 305.36965640000017, 318.6439560000008, 325.7479043000003, 317.08866560000024, 323.78449589999946, 328.52902579999954, 331.9218354000004, 346.37334199999987, 363.4760349999997, 366.31106680000084, 379.43786490000093, 380.92505249999886, 385.34915569999976, 391.38164549999965, 401.5613768000003, 413.1907749000002, 413.9181427000003, 421.6258804000008, 428.09847759999866, 434.5449421000012, 442.0456928000003, 457.6765228000004, 481.4486909999996, 485.3242255999994, 513.3398797999998, 531.3072255999996, 533.5217416999985, 521.8528061999987, 530.8493431999996, 546.9511351000001, 609.0251283999969, 571.5106276000006, 573.6436598, 575.1949633000004, 591.6008927000003, 588.833274200002, 600.5095789000006, 604.6555305000002, 606.1613204999994, 614.0329343000012, 628.4769661999999, 635.2728870000028, 638.7449796000001]
fpr: [0.013825643276458003, 0.021461667755627266, 0.024045247559813864, 0.030449333406484886, 0.03517259609869474, 0.03768634942168711, 0.03923749482535898, 0.040165189504082356, 0.04121258672199584, 0.0434769597550088, 0.04438969161633341, 0.046304932243375216, 0.04690843254512536, 0.04836980104440466, 0.04899325176935316, 0.04996583490027282, 0.050743901405008555, 0.05127258761976489, 0.05191598876791174, 0.052833708235035935, 0.0538062913659556, 0.055536990578412646, 0.05656942497892736, 0.05746719402285321, 0.05791607854481613, 0.05948218676588677, 0.06032010454021756, 0.06134755133493269, 0.062230357561459775, 0.06274906856461693, 0.06319296548078027, 0.06343735816496007, 0.06398599480291475, 0.06429522636248922, 0.0649286522990369, 0.06547230133119199, 0.0657565948617685, 0.06651969854910547, 0.06711821124505604, 0.06744739322782885, 0.06777158760480206, 0.06835014987755428, 0.06861449298493244, 0.06925290652727971, 0.07019556402340185, 0.07062449812216641, 0.070883853623745, 0.07114320912532357, 0.07135268856890627, 0.07241504860421852, 0.07314323905095838, 0.07348239624533036, 0.07395621879629122, 0.07444500416465084, 0.07463952079083477, 0.07481907459961995, 0.07505847967800017, 0.07533279799697751, 0.07551733941156227, 0.07565200476815115, 0.0766295755048704, 0.07761213384738924, 0.07797622907075916, 0.07839020035212498, 0.07885404769148666, 0.07951739926283186, 0.07970194067741662, 0.07994633336159643, 0.08017576322837748, 0.08036529224876182, 0.08117328438829509, 0.08199125173942752, 0.08246008668458879, 0.08262966528177479, 0.08321321516032659, 0.08356733517209734, 0.08375686419248168, 0.08426061237824008, 0.08472445971760176, 0.08492895155538487]
'''

'''
noboost link nocheat 
insert time: [0.04410430000000076, 0.022402100000000758, 0.04323559999999915, 0.022733500000001072, 0.06074040000000025, 0.06987409999999983, 0.058487700000000586, 0.039127300000004084, 0.09086560000000077, 0.04422289999999407, 0.05386670000000038, 0.10264079999998899, 0.11170620000001463, 0.04377539999998703, 0.047064699999992854, 0.04336340000000405, 0.04907510000001025, 0.15757110000001262, 0.08460270000000492, 0.155709800000011, 0.09059560000002875, 0.060433400000022175, 0.061203599999998914, 0.05695509999998194, 0.08461410000001024, 0.05619910000001482, 0.0568794000000139, 0.10619830000001684, 0.1983148000000483, 0.08390940000003866, 0.08978830000000926, 0.09255570000004809, 0.06901189999996404, 0.07012839999993048, 0.1336178999999902, 0.0729122999999845, 0.09908139999993182, 0.10486949999994977, 0.07838430000003882, 0.0748059000000012, 0.0718750999999429, 0.13820339999995213, 0.14892670000006092, 0.07753350000007231, 0.08274930000004588, 0.07722790000002533, 0.24660990000006677, 0.086091099999976, 0.1268463999999767, 0.13922580000007656, 0.09585619999984374, 0.11140110000019376, 0.09699049999994713, 0.14417570000000524, 0.1505331999999271, 0.11837839999998323, 0.1531472000001486, 0.09803370000008726, 0.15027770000006058, 0.11081130000002304, 0.13350609999997687, 0.157859199999848, 0.11790730000006988, 0.1404080999998314, 0.2012649999999212, 0.14170190000004368, 0.1462258000001384, 0.16878109999993285, 0.1558684000001449, 0.12794129999997494, 0.11691080000036891, 0.18179049999980634, 0.14968009999984133, 0.11709120000023177, 0.22372409999979936, 0.1209830000002512, 0.12078810000002704, 0.20236690000001545, 0.1764632999997957, 0.14569019999999]
query time: [5.9106306, 9.6867532, 10.779433699999998, 9.471324799999998, 10.896093399999998, 12.733359299999996, 14.515742399999993, 14.958998199999996, 15.359650299999998, 15.702512800000008, 16.087785900000014, 16.071630500000026, 18.49730790000001, 18.78744739999999, 18.68421429999998, 19.127796100000012, 19.608202500000004, 20.452598600000016, 21.27607310000002, 22.291360399999974, 23.2524358, 23.29159040000002, 23.054104100000018, 23.259464100000002, 24.66098249999999, 24.961566199999993, 25.264128699999958, 24.867373799999996, 25.55466880000006, 27.158444499999973, 29.155415199999993, 29.352530199999933, 29.15926239999999, 29.740001399999983, 29.62735109999994, 29.289521700000023, 31.331348299999945, 32.34801719999996, 32.906678000000056, 32.34514920000004, 32.58934309999995, 33.324707099999955, 34.24827620000008, 34.6268988999999, 34.57729369999993, 34.62805490000005, 37.72921419999989, 35.70485750000012, 37.06947020000007, 38.29553299999998, 37.767801899999995, 37.47068800000011, 37.757266299999856, 38.268639399999984, 40.19759739999995, 41.625607800000125, 41.45530280000003, 42.63064839999993, 41.72573349999993, 41.79594400000019, 42.94950730000005, 43.37055510000005, 43.038926500000116, 44.41667540000003, 44.37235480000004, 45.186039500000106, 46.61648779999996, 47.167665599999964, 47.23141220000002, 47.43640379999988, 47.89920220000022, 48.32118270000001, 49.65975649999973, 50.45587669999986, 50.17427120000002, 50.03756659999999, 50.00961529999995, 51.22143879999976, 53.2350892999998, 54.788556399999834]
fpr: [0.013825643276458003, 0.013825643276458003, 0.013825643276458003, 0.013825643276458003, 0.01993546038095333, 0.02485323969934712, 0.025785921983870082, 0.025785921983870082, 0.028020369382085516, 0.028020369382085516, 0.028020369382085516, 0.028020369382085516, 0.029950572826526082, 0.029950572826526082, 0.029950572826526082, 0.029950572826526082, 0.029950572826526082, 0.03050918467607994, 0.03153663147079507, 0.03153663147079507, 0.03235459882192751, 0.03235459882192751, 0.03235459882192751, 0.03235459882192751, 0.03338703322244223, 0.03338703322244223, 0.03338703322244223, 0.03338703322244223, 0.03436460395916149, 0.03496810426091163, 0.036120241200616465, 0.036933220945949315, 0.036933220945949315, 0.036933220945949315, 0.0370229978503419, 0.0370229978503419, 0.038100320703052915, 0.03889335002518741, 0.03889335002518741, 0.03889335002518741, 0.03889335002518741, 0.03949685032693756, 0.040444495428859285, 0.040444495428859285, 0.040444495428859285, 0.040444495428859285, 0.04150685546417154, 0.04150685546417154, 0.042539289864686254, 0.042539289864686254, 0.042539289864686254, 0.042539289864686254, 0.042539289864686254, 0.04303805044464506, 0.04416524935535195, 0.044793687686100044, 0.044793687686100044, 0.044793687686100044, 0.044978229100684795, 0.044978229100684795, 0.045975750260602403, 0.04607550237659416, 0.04607550237659416, 0.04690344493932577, 0.04690344493932577, 0.047402205519284576, 0.048294986957410836, 0.04846456555459683, 0.048489503583594766, 0.048489503583594766, 0.048489503583594766, 0.04919275600133668, 0.05020025237285346, 0.05020025237285346, 0.05020025237285346, 0.05020025237285346, 0.05020025237285346, 0.05170650932432904, 0.05267410484944912, 0.053362394449792266]
'''

'''
    def build_clf_and_predict_score(self, keys, labels=None):
        alpha_sum, raw_score, weights = self.boost_previous_step(keys, labels)
        cur_clf = inner_node.train_model_core(keys, labels, MLBF_para["default_model"].copy(), weights)
        tmp_alpha, _, final_score, _ = ca_lbf.boost_one_step(keys, cur_clf, None, alpha_sum, raw_score, weights, labels,
                                                             True)
        self.model_list.append(cur_clf)
        self.alpha_list.append(tmp_alpha)
        return final_score

    def boost_previous_step(self, keys, labels=None):
        alpha_sum = 0
        weights = np.ones(len(keys))
        final_score = np.zeros(len(keys))
        for cur_clf, cur_alpha in zip(self.model_list, self.alpha_list):
            tmp_alpha, alpha_sum, final_score, weights = ca_lbf.boost_one_step(keys, cur_clf, cur_alpha, alpha_sum,
                                                                               final_score, weights, labels)
        return alpha_sum, final_score, weights

    @staticmethod
    def boost_one_step(keys, cur_clf, cur_alpha, alpha_sum, final_score, weights=None, labels=None, is_current=False):
        cur_score = inner_node.get_score_with_clf(cur_clf, keys, MLBF_para["default_model"]["type"])
        binary_predict_res = [1 if s >= 0.5 else 0 for s in cur_score]
        if labels is not None:  # using in inserting process
            tmp_alpha, weights = update_weight(binary_predict_res, labels, weights)
        if is_current:
            cur_alpha = tmp_alpha
        alpha_sum, final_score = update_score(alpha_sum, cur_alpha, final_score, cur_score)
        return tmp_alpha, alpha_sum, final_score, weights
'''
