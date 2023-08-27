"""
Created on Aug 8, 2016
Processing datasets. 
@author: Xiangnan He (xiangnanhe@gmail.com)
"""

import os
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from util.tool import csr_to_user_dict_bytime, csr_to_user_dict
from .utils import check_md5
from util.logger import Logger
from util import randint_choice
import numpy as np
from .utils import filter_data, split_by_ratio, split_by_loo
import collections
import tensorflow as tf


class Dataset(object):
    def __init__(self, conf):
        """Constructor
        """
        self.train_matrix = None
        self.test_matrix = None
        self.time_matrix = None
        self.negative_matrix = None
        self.userids = None
        self.itemids = None
        self.num_users = None
        self.num_items = None
        self.num_entities = None
        self.num_relations = None
        self.dataset_name = conf["data.input.dataset"]
        self.sparsity = conf["sparsity"]
        self.sparsity_ratio = conf["sparsity_ratio"]
        self.mask = conf["mask"]
        self.mask_ratio = conf["mask_ratio"]
        self.path_length = conf["pathlength"]
        self.noisy = conf["noisy"]
        self.noisy_ratio = conf["noisy_ratio"]

        # self._split_data(conf)
        self._load_data(conf)

    def _get_data_path(self, config):
        data_path = config["data.input.path"]
        ori_prefix = os.path.join(data_path, self.dataset_name)

        saved_path = os.path.join(data_path, "_tmp_"+self.dataset_name)
        saved_prefix = "%s_%s_u%d_i%d" % (self.dataset_name, config["splitter"], config["user_min"], config["item_min"])
        if "by_time" in config and config["by_time"] is True:
            saved_prefix += "_by_time"

        saved_prefix = os.path.join(saved_path, saved_prefix)

        return ori_prefix, saved_prefix

    def _check_saved_data(self, splitter, ori_prefix, saved_prefix):
        check_state = False
        # get md5
        if splitter in ("loo", "ratio"):
            rating_file = ori_prefix + ".rating"
            ori_file_md5 = [check_md5(rating_file)]
        elif splitter == "given":
            train_file = ori_prefix + ".train"
            test_file = ori_prefix + self.path_length + ".test"
            ori_file_md5 = [check_md5(file) for file in [train_file, test_file]]
        else:
            raise ValueError("'%s' is an invalid splitter!" % splitter)

        # check md5
        if os.path.isfile(saved_prefix + ".md5"):
            with open(saved_prefix + ".md5", 'r') as md5_fin:
                saved_md5 = [line.strip() for line in md5_fin.readlines()]
            if ori_file_md5 == saved_md5:
                check_state = True

        # check saved files
        for postfix in [".train", self.path_length + ".test", ".user2id", ".item2id"]:
            if not os.path.isfile(saved_prefix + postfix):
                check_state = False

        return check_state

    def _load_data(self, config):
        format_dict = {"UIRT": ["user", "item", "rating", "time"],
                       "UIR": ["user", "item", "rating"],
                       "UI": ["user", "item"],
                       "UIT": ["user", "item", "time"]
                       }
        file_format = config["data.column.format"]
        if file_format not in format_dict:
            raise ValueError("'%s' is an invalid data column format!" % file_format)

        ori_prefix, saved_prefix = self._get_data_path(config)
        splitter = config["splitter"]
        sep = config["data.convert.separator"]
        columns = format_dict[file_format]
        train_file = saved_prefix + ".train"
        test_file = saved_prefix + self.path_length + ".test"
        user_map_file = saved_prefix + ".user2id"
        item_map_file = saved_prefix + ".item2id"

        if self._check_saved_data(splitter, ori_prefix, saved_prefix):
            print("load saved data...")
            # load saved data
            train_data = pd.read_csv(train_file, sep=sep, header=None, names=columns)
            test_data = pd.read_csv(test_file, sep=sep, header=None, names=columns)

            user_map = pd.read_csv(user_map_file, sep=sep, header=None, names=["user", "id"])
            item_map = pd.read_csv(item_map_file, sep=sep, header=None, names=["item", "id"])
            self.userids = {user: uid for user, uid in zip(user_map["user"], user_map["id"])}
            self.itemids = {item: iid for item, iid in zip(item_map["item"], item_map["id"])}
        else:  # split and save data
            print("split and save data...")
            by_time = config["by_time"] if file_format in {"UIRT", "UIT"} else False
            train_data, test_data = self._split_data(ori_prefix, saved_prefix, columns, by_time, config)

        all_data = pd.concat([train_data, test_data])
        self.train_data = train_data
        self.num_users = max(all_data["user"]) + 1
        self.num_items = max(all_data["item"]) + 1
        self.num_ratings = len(all_data)

        if file_format in {"UI", "UIT"}:
            train_ratings = [1.0] * len(train_data["user"])
            test_ratings = [1.0] * len(test_data["user"])
        else:
            train_ratings = train_data["rating"]
            test_ratings = test_data["rating"]

        self.train_matrix = csr_matrix((train_ratings, (train_data["user"], train_data["item"])),
                                       shape=(self.num_users, self.num_items))
        self.test_matrix = csr_matrix((test_ratings, (test_data["user"], test_data["item"])),
                                      shape=(self.num_users, self.num_items))


        if file_format in {"UIRT", "UIT"}:
            self.time_matrix = csr_matrix((train_data["time"], (train_data["user"], train_data["item"])),
                                          shape=(self.num_users, self.num_items))

        self.negative_matrix = self._load_test_neg_items(all_data, config, saved_prefix)

    def _split_data(self, ori_prefix, saved_prefix, columns, by_time, config):
        splitter = config["splitter"]
        user_min = config["user_min"]
        item_min = config["item_min"]
        sep = config["data.convert.separator"]

        dir_name = os.path.dirname(saved_prefix)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        if splitter in ("loo", "ratio"):
            rating_file = ori_prefix + ".rating"
            all_data = pd.read_csv(rating_file, sep=sep, header=None, names=columns)
            filtered_data = filter_data(all_data, user_min=user_min, item_min=item_min)
            if splitter == "ratio":
                ratio = config["ratio"]
                train_data, test_data = split_by_ratio(filtered_data, ratio=ratio, by_time=by_time)
            elif splitter == "loo":
                train_data, test_data = split_by_loo(filtered_data, by_time=by_time)
            else:
                raise ValueError("There is not splitter '%s'" % splitter)
            with open(saved_prefix+".md5", "w") as md5_out:
                md5_out.writelines(check_md5(rating_file))
        elif splitter == "given":
            train_file = ori_prefix + ".train"
            test_file = ori_prefix + self.path_length + ".test"
            train_data = pd.read_csv(train_file, sep=sep, header=None, names=columns)
            test_data = pd.read_csv(test_file, sep=sep, header=None, names=columns)
            with open(saved_prefix+".md5", "w") as md5_out:
                md5_out.writelines('\n'.join([check_md5(train_file), check_md5(test_file)]))
                # md5_out.writelines(check_md5(test_file))
        else:
            raise ValueError("'%s' is an invalid splitter!" % splitter)

        # remap id
        if config['remap']:
            all_data = pd.concat([train_data, test_data])
            unique_user = all_data["user"].unique()
            self.userids = pd.Series(data=range(len(unique_user)), index=unique_user).to_dict()
            train_data["user"] = train_data["user"].map(self.userids)
            test_data["user"] = test_data["user"].map(self.userids)

            unique_item = all_data["item"].unique()
            self.itemids = pd.Series(data=range(len(unique_item)), index=unique_item).to_dict()
            train_data["item"] = train_data["item"].map(self.itemids)
            test_data["item"] = test_data["item"].map(self.itemids)
            user2id = [[user, id] for user, id in self.userids.items()]
            item2id = [[item, id] for item, id in self.itemids.items()]
            np.savetxt(saved_prefix + ".user2id", user2id, fmt='%s', delimiter=sep)
            np.savetxt(saved_prefix + ".item2id", item2id, fmt='%s', delimiter=sep)

        # save files
        np.savetxt(saved_prefix+".train", train_data, fmt='%d', delimiter=sep)
        np.savetxt(saved_prefix+self.path_length + ".test", test_data, fmt='%d', delimiter=sep)

        # remap test negative items and save to a file
        neg_item_file = ori_prefix + ".neg"
        if os.path.isfile(neg_item_file):
            neg_item_list = []
            with open(neg_item_file, 'r') as fin:
                for line in fin.readlines():
                    line = line.strip().split(sep)
                    user_items = [self.userids[line[0]]]
                    user_items.extend([self.itemids[i] for i in line[1:]])
                    neg_item_list.append(user_items)

            test_neg = len(neg_item_list[0]) - 1
            np.savetxt("%s.neg%d" % (saved_prefix, test_neg), neg_item_list, fmt='%d', delimiter=sep)

        all_remapped_data = pd.concat([train_data, test_data])
        self.num_users = max(all_remapped_data["user"]) + 1
        self.num_items = max(all_remapped_data["item"]) + 1
        self.num_ratings = len(all_remapped_data)

        logger = Logger(saved_prefix+".info")
        logger.info(os.path.basename(saved_prefix))
        logger.info(self.__str__())

        return train_data, test_data

    def _load_test_neg_items(self, all_data, config, saved_prefix):
        number_neg = config["rec.evaluate.neg"]
        sep = config["data.convert.separator"]
        neg_matrix = None
        if number_neg > 0:
            neg_items_file = "%s.neg%d" % (saved_prefix, number_neg)
            if not os.path.isfile(neg_items_file):
                # sampling
                neg_items = []
                grouped_user = all_data.groupby(["user"])
                for user, u_data in grouped_user:
                    line = [user]
                    line.extend(randint_choice(self.num_items, size=number_neg,
                                               replace=False, exclusion=u_data["item"].tolist()))
                    neg_items.append(line)

                neg_items = pd.DataFrame(neg_items)
                np.savetxt("%s.neg%d" % (saved_prefix, number_neg), neg_items, fmt='%d', delimiter=sep)
            else:
                # load file
                neg_items = pd.read_csv(neg_items_file, sep=sep, header=None)

            user_list, item_list = [], []
            for line in neg_items.values:
                user_list.extend([line[0]] * (len(line) - 1))
                item_list.extend(line[1:])

            neg_matrix = csr_matrix(([1] * len(user_list), (user_list, item_list)),
                                    shape=(self.num_users, self.num_items))

        return neg_matrix

    def __str__(self):
        num_users, num_items = self.num_users, self.num_items
        num_ratings = self.num_ratings
        sparsity = 1 - 1.0*num_ratings/(num_users*num_items)
        data_info = ["Dataset name: %s" % self.dataset_name,
                     "The number of users: %d" % num_users,
                     "The number of items: %d" % num_items,
                     "The number of ratings: %d" % num_ratings,
                     "Average actions of users: %.2f" % (1.0*num_ratings/num_users),
                     "Average actions of items: %.2f" % (1.0*num_ratings/num_items),
                     "The sparsity of the dataset: %.6f%%" % (sparsity * 100)]
        data_info = "\n".join(data_info)
        return data_info

    def __repr__(self):
        return self.__str__()

    def get_user_train_dict(self, by_time=False):
        if by_time:
            train_dict = csr_to_user_dict_bytime(self.time_matrix, self.train_matrix)
        else:
            train_dict = csr_to_user_dict(self.train_matrix)

        return train_dict

    def get_user_test_dict(self):
        test_dict = csr_to_user_dict(self.test_matrix)
        return test_dict

    def get_user_test_neg_dict(self):
        test_neg_dict = None
        if self.negative_matrix is not None:
            test_neg_dict = csr_to_user_dict(self.negative_matrix)
        return test_neg_dict

    def get_train_interactions(self):
        dok_matrix = self.train_matrix.todok()
        users_list, items_list = [], []
        for (user, item), value in dok_matrix.items():
            users_list.append(user)
            items_list.append(item)

        return users_list, items_list

    def to_csr_matrix(self):
        return self.train_matrix.copy()

    def get_kg_ori(self, config):
        self.kg = collections.defaultdict(list)
        self.rd = collections.defaultdict(list)
        kg_file = os.path.join(config["data.input.path"], self.dataset_name + '.kg_final')
        self.kg_np = np.loadtxt(kg_file, dtype=np.int64)
        self.num_entities = len(set(self.kg_np[:, 0]) | set(self.kg_np[:, 2]))
        self.num_relations = len(set(self.kg_np[:, 1]))

        for h, r, t in self.kg_np:
            if h not in self.kg:
                self.kg[h] = []
            self.kg[h].append((t, r))
            self.rd[r].append((h, t))

        return self.num_entities, self.num_relations, self.kg

    def get_kg(self, config, train_dict, add_user=False, reverse=False):
        # 如果加入user，并且对更新后的kg进行KGE辅助训练，应该更新kg_np

        self.kg = dict()
        num_users = 0
        kg_file = os.path.join(config["data.input.path"], self.dataset_name + '.kg_final')
        self.kg_np = np.loadtxt(kg_file, dtype=np.int64)
        self.kg_np = np.unique(self.kg_np, axis=0)
        self.num_entities = len(set(self.kg_np[:, 0]) | set(self.kg_np[:, 2]))
        self.num_relations = len(set(self.kg_np[:, 1]))
        add_triple = []

        if add_user:
            num_users = self.num_users
            for user, items in train_dict.items():
                self.kg[user] = []
                for item in items:
                    item += num_users
                    if item not in self.kg:
                        self.kg[item] = []
                    self.kg[user].append((item, self.num_relations))
                    add_triple.append([user, self.num_relations, item])
                    if reverse:
                        self.kg[item].append((user, self.num_relations * 2 + 1))
                    else:
                        self.kg[item].append((user, self.num_relations))

        for h, r, t in self.kg_np:
            h += num_users
            t += num_users
            if h not in self.kg:
                self.kg[h] = []
            self.kg[h].append((t, r))
            if t not in self.kg:
                self.kg[t] = []
            if reverse:
                self.kg[t].append((h, r + self.num_relations + 1))
            else:
                self.kg[t].append((h, r))
                add_triple.append([t, r, h])
        self.kg_np[:, 0] += self.num_users
        self.kg_np[:, 2] += self.num_users
        if add_triple:
            self.kg_np = np.vstack((self.kg_np, add_triple))
        return self.num_entities, self.num_relations, self.kg

    def get_adj(self, kg, neighbor_sample_size, n_entity, add_user=False):
        print('constructing adjacency matrix ...')
        # each line of adj_entity stores the sampled neighbor entities for a given entity
        # each line of adj_relation stores the corresponding sampled neighbor relations
        if add_user:
            all_num = n_entity + self.num_users  # 将user与item的知识图谱融合，因此新知识图谱的数量为n_entity + n_user
        else:
            all_num = n_entity
        adj_entity = np.zeros([all_num, neighbor_sample_size], dtype=np.int64)
        adj_relation = np.zeros([all_num, neighbor_sample_size], dtype=np.int64)

        for entity in range(all_num):
            neighbors = kg[entity]
            n_neighbors = len(neighbors)
            if n_neighbors >= neighbor_sample_size:
                sampled_indices = np.random.choice(list(range(n_neighbors)), size=neighbor_sample_size, replace=False)
            else:
                sampled_indices = np.random.choice(list(range(n_neighbors)), size=neighbor_sample_size, replace=True)

            adj_entity[entity] = np.array([neighbors[i][0] for i in sampled_indices])
            adj_relation[entity] = np.array([neighbors[i][1] for i in sampled_indices])

        return adj_entity, adj_relation
