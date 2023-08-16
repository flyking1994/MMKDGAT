"""
Paper: LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
Author: Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, and Meng Wang
Reference: https://github.com/hexiangnan/LightGCN
"""
import time

import scipy.sparse as sp
import tensorflow as tf
import numpy as np
from model.AbstractRecommender import AbstractRecommender
from util import timer
from util import l2_loss, inner_product, log_loss
from util import get_info
from data import PairwiseSampler, TripleSampler, TripleSamplerV2
from sklearn.metrics import f1_score, roc_auc_score
import collections


class MMKDGAT(AbstractRecommender):
    def __init__(self, sess, dataset, config):
        super(MMKDGAT, self).__init__(dataset, config)
        self.lr = config['lr']
        self.reg = config['reg']
        self.emb_dim = config['embed_size']
        self.batch_size = config['batch_size']
        self.kg_batch_size = config['kg_batch_size']
        self.epochs = config["epochs"]
        self.n_layers = config['n_layers']
        self.adj_type = config['adj_type']
        self.alpha = config['alpha']
        self.head_nums = config['head_nums']
        self.kge = config['kge']
        self.lr_kge = config['lr_kge']
        self.reg_kge = config['reg_kge']
        self.merge = config['merge']
        self.ent_ratio = config['ent_ratio']
        self.img_ratio = config['img_ratio']
        self.stop_cnt = config['stop_cnt']

        self.dataset = dataset
        self.n_users, self.n_items = self.dataset.num_users, self.dataset.num_items
        self.user_pos_train = self.dataset.get_user_train_dict(by_time=False)
        self.user_pos_test = self.dataset.get_user_test_dict()

        self.num_entities, self.num_relations, self.kg = self.dataset.get_kg(config, self.user_pos_train, add_user=True, reverse=True)
        self.n_nodes = self.n_users + self.num_entities

        self.img_features = np.load('./features/' + self.dataset.dataset_name + '_re_features.npy')
        noimg_idx = np.array([idx for idx in range(len(self.img_features)) if all(self.img_features[idx] == 0)])  # 没有影像特征的实体
        self.img_idx = np.array([idx for idx in range(len(self.img_features)) if any(self.img_features[idx] != 0)])  # 有影像特征的实体
        self.img_matrix = np.ones([self.num_entities, self.emb_dim], dtype=np.float32)
        self.emp_matrix = np.ones([self.num_entities, self.emb_dim], dtype=np.float32)
        self.img_matrix[noimg_idx] = [0.] * self.emb_dim  # 没有影像特征的实体为0
        self.emp_matrix[self.img_idx] = [0.] * self.emb_dim  # 有影像特征的实体为0

        self.all_users = list(self.user_pos_train.keys())
        test_users = list(self.user_pos_test.keys())
        self.test_users = test_users

        self.norm_adj, self.rel_adj = self.create_adj_mat(adj_type=self.adj_type)
        self.all_h_list, self.all_r_list, self.all_t_list = self._get_all_kg_data(self.rel_adj)
        self.sess = sess

    @timer
    def create_adj_mat(self, adj_type):
        head_list = []
        tail_list = []
        rel_list = []
        for head, tail_rels in self.kg.items():
            head_list.extend([head] * len(tail_rels))
            tails, rels = zip(*tail_rels)
            tail_list.extend(tails)
            rel_list.extend(rels)

        heads_np = np.array(head_list)
        tails_np = np.array(tail_list)
        rels_np = np.array(rel_list)
        ratings = np.ones_like(heads_np, dtype=np.float32)

        tmp_adj = sp.csr_matrix((ratings, (heads_np, tails_np)), shape=(self.n_nodes, self.n_nodes))
        rel_adj = sp.csr_matrix((rels_np, (heads_np, tails_np)), shape=(self.n_nodes, self.n_nodes))
        adj_mat = tmp_adj + tmp_adj.T

        def normalized_adj_left(adj):
            rowsum = np.array(adj.sum(1))
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        def normalized_adj_right(adj):
            rowsum = np.array(adj.sum(1))
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj = adj.dot(d_mat_inv)
            return norm_adj.tocoo()

        def normalized_adj_sys(adj):
            rowsum = np.array(adj.sum(1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj_tmp = d_mat_inv.dot(adj)
            adj_matrix = norm_adj_tmp.dot(d_mat_inv)
            print('use the pre adjcency matrix')
            return adj_matrix

        if adj_type == 'plain':
            adj_matrix = adj_mat
            print('use the plain adjacency matrix')
        elif adj_type == 'norm':
            adj_matrix = normalized_adj_left(adj_mat + sp.eye(adj_mat.shape[0]))
            print('use the normalized adjacency matrix')
        elif adj_type == 'gcmc':
            # left normalization
            adj_matrix = normalized_adj_left(adj_mat)
            print('use the gcmc adjacency matrix')
        elif adj_type == 'right':
            adj_matrix = normalized_adj_right(adj_mat)
        elif adj_type == 'pre':
            # pre adjcency matrix
            adj_matrix = normalized_adj_sys(adj_mat)
        elif adj_type == 'pre_sl':
            # pre adjcency matrix
            adj_mat = adj_mat + sp.eye(adj_mat.shape[0])
            rowsum = np.array(adj_mat.sum(1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj_tmp = d_mat_inv.dot(adj_mat)
            adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        elif adj_type == 'lr':
            adj_matrix = 0.5 * normalized_adj_right(adj_mat) + 0.5 * normalized_adj_left(adj_mat)
        elif adj_type == 'ls':
            adj_matrix = 0.5 * normalized_adj_sys(adj_mat) + 0.5 * normalized_adj_left(adj_mat)
        elif adj_type == 'lsr':
            adj_matrix = 1/3 * normalized_adj_sys(adj_mat) + 1/3 * normalized_adj_left(adj_mat) + 1/3 * normalized_adj_right(adj_mat)
        else:
            mean_adj = normalized_adj_left(adj_mat)
            adj_matrix = mean_adj + sp.eye(mean_adj.shape[0])
            print('use the mean adjacency matrix')

        return adj_matrix, rel_adj.tocoo()

    def _get_all_kg_data(self, rel_mat):
        def _reorder_list(org_list, order):
            new_list = np.array(org_list)
            new_list = new_list[order]
            return new_list

        all_h_list = list(rel_mat.row)
        all_t_list = list(rel_mat.col)
        all_r_list = list(rel_mat.data)

        assert len(all_h_list) == len(all_r_list)

        # resort the all_h/t/r_list,
        # ... since tensorflow.sparse.softmax requires indices sorted in the canonical lexicographic order
        org_h_dict = dict()

        for idx, h in enumerate(all_h_list):
            if h not in org_h_dict.keys():
                org_h_dict[h] = [[], []]

            org_h_dict[h][0].append(all_t_list[idx])
            org_h_dict[h][1].append(all_r_list[idx])

        sorted_h_dict = dict()
        for h in org_h_dict.keys():
            org_t_list, org_r_list = org_h_dict[h]
            sort_t_list = np.array(org_t_list)
            sort_order = np.argsort(sort_t_list)

            sort_t_list = _reorder_list(org_t_list, sort_order)
            sort_r_list = _reorder_list(org_r_list, sort_order)

            sorted_h_dict[h] = [sort_t_list, sort_r_list]

        od = collections.OrderedDict(sorted(sorted_h_dict.items()))
        new_h_list, new_t_list, new_r_list = [], [], []

        for h, vals in od.items():
            new_h_list += [h] * len(vals[0])
            new_t_list += list(vals[0])
            new_r_list += list(vals[1])

        assert sum(new_h_list) == sum(all_h_list)
        assert sum(new_t_list) == sum(all_t_list)
        assert sum(new_r_list) == sum(all_r_list)

        return new_h_list, new_r_list, new_t_list

    def _create_variable(self):

        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))

        self.sub_mat = dict()
        self.sub_mat['adj_values'] = tf.placeholder(tf.float32)
        self.sub_mat['adj_indices'] = tf.placeholder(tf.int64)
        self.sub_mat['adj_shape'] = tf.placeholder(tf.int64)

        # 导致结果细微变化的原因是占位符的增加
        self.h = tf.placeholder(tf.int32, shape=[None])
        self.r = tf.placeholder(tf.int32, shape=[None])
        self.pos_t = tf.placeholder(tf.int32, shape=[None])
        self.neg_t = tf.placeholder(tf.int32, shape=[None])
        self.att_scores = tf.placeholder(tf.float32, shape=[len(self.all_h_list)])

        self.weights = dict()
        initializer = tf.contrib.layers.xavier_initializer()
        self.weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name='user_embedding')
        self.weights['entity_embedding'] = tf.Variable(initializer([self.num_entities, self.emb_dim]), name='entity_embedding')
        self.weights['relation_embedding'] = tf.Variable(
            initializer([self.num_relations * 2 + 2, self.emb_dim]), name='relation_embedding')
        self.weights['image_embedding'] = tf.constant(self.img_features, name='image_embedding')

        '''
        self.ent_ratio = self.img_ratio = 1.0, ratio == add
        self.ent_ratio = self.img_ratio = 0.5, ratio == mean (无影像特征不除以2)
        '''
        self.ent_ratio = tf.constant(self.ent_ratio, name='ent_ratio')
        self.img_ratio = tf.constant(self.img_ratio, name='img_ratio')

        ent_with_img_embedding = self.weights['entity_embedding'] * self.img_matrix
        ent_without_img_embedding = self.weights['entity_embedding'] * self.emp_matrix
        self.weights['entity_embedding'] = ent_without_img_embedding + self.ent_ratio * ent_with_img_embedding + self.img_ratio * self.weights['image_embedding']

        for k in range(self.head_nums):
            self.weights['head_W_%d' % k] = tf.Variable(
                initializer([self.emb_dim * 3, 1]), name='head_W_%d' % k)
            self.weights['head_b_%d' % k] = tf.Variable(
                initializer([1, ]), name='head_b_%d' % k)

    def _recommendation(self):
        self.ua_embeddings, self.ia_embeddings = self._create_lightgcn_embed_ir()

        """
        *********************************************************
        Establish the final representations for user-item pairs in batch.
        """
        self.u_g_embeddings = tf.nn.embedding_lookup(self.ua_embeddings, self.users)
        self.pos_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.pos_items)
        self.neg_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.neg_items)
        self.u_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['user_embedding'], self.users)
        self.pos_i_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['entity_embedding'], self.pos_items)
        self.neg_i_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['entity_embedding'], self.neg_items)

        """
        *********************************************************
        Inference for the testing phase.
        """
        self.item_embeddings_final = tf.Variable(tf.zeros([self.n_items, self.emb_dim]),
                                                 dtype=tf.float32, name="item_embeddings_final", trainable=False)
        self.user_embeddings_final = tf.Variable(tf.zeros([self.n_users, self.emb_dim]),
                                                 dtype=tf.float32, name="user_embeddings_final", trainable=False)

        self.assign_opt = [tf.assign(self.user_embeddings_final, self.ua_embeddings),
                           tf.assign(self.item_embeddings_final, self.ia_embeddings)]

        u_embed = tf.nn.embedding_lookup(self.user_embeddings_final, self.users)
        self.batch_ratings = tf.matmul(u_embed, self.item_embeddings_final, transpose_a=False, transpose_b=True)

        """
        *********************************************************
        Generate Predictions & Optimize via BPR loss.
        """
        self.mf_loss, self.emb_loss = self.create_bpr_loss(self.u_g_embeddings,
                                                           self.pos_i_g_embeddings,
                                                           self.neg_i_g_embeddings)
        self.loss = self.mf_loss + self.emb_loss

        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def _get_kge_embedding(self):
        embeddings = tf.concat([self.weights['user_embedding'], self.weights['entity_embedding']], axis=0)
        # head & tail entity embeddings: batch_size * emb_dim

        self.h_e = tf.nn.embedding_lookup(embeddings, self.h)
        self.pos_t_e = tf.nn.embedding_lookup(embeddings, self.pos_t)
        self.neg_t_e = tf.nn.embedding_lookup(embeddings, self.neg_t)

        # relation embeddings: batch_size * kge_dim
        self.r_e = tf.nn.embedding_lookup(self.weights['relation_embedding'], self.r)

    def _attention(self):
        self.scores = self.get_att_scores_multihead()
        self.attention = self.normalize_att_scores()

    def _mmkge(self):
        h_e, r_e, pos_t_e, neg_t_e = self._get_kg_inference()

        def _get_kg_score(h_e, r_e, t_e):
            kg_score = tf.reduce_sum(tf.square((h_e + r_e - t_e)), 1, keepdims=True)
            return kg_score

        pos_kg_score = _get_kg_score(h_e, r_e, pos_t_e)
        neg_kg_score = _get_kg_score(h_e, r_e, neg_t_e)

        # Using the softplus as BPR loss to avoid the nan error.
        kg_loss = tf.reduce_mean(tf.nn.softplus(-(neg_kg_score - pos_kg_score)))

        kg_reg_loss = l2_loss(h_e, r_e, pos_t_e, neg_t_e)
        kg_reg_loss = kg_reg_loss / tf.cast(tf.shape(pos_kg_score)[0], tf.float32)

        self.kge_loss_kge = kg_loss
        self.reg_loss_kge = self.reg_kge * kg_reg_loss
        self.loss_kge = self.kge_loss_kge + self.reg_loss_kge

        # Optimization process.
        self.opt_kge = tf.train.AdamOptimizer(learning_rate=self.lr_kge).minimize(self.loss_kge)

    def _get_kg_inference(self):
        h_e = self.h_e
        r_e = self.r_e
        pos_t_e = self.pos_t_e
        neg_t_e = self.neg_t_e

        return h_e, r_e, pos_t_e, neg_t_e

    def build_graph(self):
        self._create_variable()
        self._recommendation()
        self._get_kge_embedding()
        self._attention()
        self._mmkge()

    def _create_lightgcn_embed_ir(self):
        adj_mat = tf.SparseTensor(
            self.sub_mat['adj_indices'],
            self.sub_mat['adj_values'],
            self.sub_mat['adj_shape'])

        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['entity_embedding']], axis=0)
        ego_embeddings_init = tf.concat([self.weights['user_embedding'], self.weights['entity_embedding']], axis=0)

        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):
            side_embeddings = tf.sparse_tensor_dense_matmul(adj_mat, ego_embeddings, name="sparse_dense")

            # transformed sum messages of neighbors.
            ego_embeddings = (1 - self.alpha) * side_embeddings + self.alpha * ego_embeddings_init
            all_embeddings += [ego_embeddings]

        all_embeddings = tf.stack(all_embeddings, 1)
        all_embeddings = tf.reduce_mean(all_embeddings, axis=1, keepdims=False)
        u_g_embeddings, i_g_embeddings, _ = tf.split(all_embeddings, [self.n_users, self.n_items, self.num_entities-self.n_items], 0)
        return u_g_embeddings, i_g_embeddings

    @staticmethod
    def _convert_sp_mat_to_sp_tensor(X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    @staticmethod
    def _convert_csr_to_sparse_tensor_inputs(X):
        coo = X.tocoo()
        indices = np.mat([coo.row, coo.col]).transpose()
        return indices, coo.data, coo.shape

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = inner_product(users, pos_items)
        neg_scores = inner_product(users, neg_items)

        self.pos_score_normalized = tf.sigmoid(pos_scores)
        self.neg_score_normalized = tf.sigmoid(neg_scores)

        regularizer = l2_loss(self.u_g_embeddings_pre, self.pos_i_g_embeddings_pre, self.neg_i_g_embeddings_pre)

        mf_loss = tf.reduce_sum(log_loss(pos_scores - neg_scores))

        emb_loss = self.reg * regularizer

        return mf_loss, emb_loss

    def get_att_scores_multihead(self):

        # head||rel||tail
        h_r_t = tf.concat([self.h_e, self.r_e, self.pos_t_e], 1)

        all_att_scores = []
        for k in range(self.head_nums):
            att_scores = tf.nn.leaky_relu(
                tf.matmul(h_r_t, self.weights['head_W_%d' % k]) + self.weights['head_b_%d' % k])
            all_att_scores += [att_scores]
        all_att_scores = tf.stack(all_att_scores, 1)
        scores = tf.reduce_mean(all_att_scores, axis=1, keepdims=False)
        scores = tf.squeeze(scores, axis=1)
        return scores

    def normalize_att_scores(self):
        indices = np.mat([self.all_h_list, self.all_t_list]).transpose()  # 生成一个索引矩阵
        attention = tf.sparse.softmax(tf.SparseTensor(indices, self.att_scores, (self.n_nodes, self.n_nodes)))
        return attention

    def update_attention(self, sess):
        kg_len = len(self.all_h_list)
        att_scores = []

        for start in range(0, kg_len, self.batch_size):
            end = start + self.batch_size

            feed_dict = {
                self.h: self.all_h_list[start:end],
                self.r: self.all_r_list[start:end],
                self.pos_t: self.all_t_list[start:end]
            }
            scores = sess.run(self.scores, feed_dict=feed_dict)

            att_scores += list(scores)

        att_scores = sp.coo_matrix((att_scores, (self.all_h_list, self.all_t_list)), shape=(self.n_nodes, self.n_nodes))
        att_scores = att_scores.data * self.norm_adj.data
        attention = sess.run(self.attention, feed_dict={self.att_scores: att_scores})
        new_attention_values = attention.values
        new_attention_indices = attention.indices

        rows = new_attention_indices[:, 0]
        cols = new_attention_indices[:, 1]
        attention_in = sp.coo_matrix((new_attention_values, (rows, cols)), shape=(self.n_nodes, self.n_nodes))

        return attention_in

    def train_model(self):
        data_iter = PairwiseSampler(self.dataset, neg_num=1, batch_size=self.batch_size, shuffle=True, user_pos_dict=self.user_pos_train)
        test_iter = PairwiseSampler(self.dataset, neg_num=1, batch_size=self.batch_size, shuffle=True, user_pos_dict=self.user_pos_test)
        triple_iter = TripleSampler(self.dataset, neg_num=1, batch_size=self.kg_batch_size, shuffle=True, kg=self.kg)
        best_result, cur_epoch, cur_auc, cur_f1 = None, None, None, None
        best_user_embeddings, best_entity_embeddings = [], []
        step = 0

        for epoch in range(self.epochs):
            attention_scores = self.update_attention(self.sess)
            # attention_scores = self.norm_adj

            sub_mat = dict()
            sub_mat['adj_indices'], sub_mat['adj_values'], sub_mat['adj_shape'] = \
                self._convert_csr_to_sparse_tensor_inputs(attention_scores)

            for bat_users, bat_pos_items, bat_neg_items in data_iter:
                feed = {self.users: bat_users,
                        self.pos_items: bat_pos_items,
                        self.neg_items: bat_neg_items}

                feed.update({
                    self.sub_mat['adj_values']: sub_mat['adj_values'],
                    self.sub_mat['adj_indices']: sub_mat['adj_indices'],
                    self.sub_mat['adj_shape']: sub_mat['adj_shape']})

                self.sess.run([self.opt], feed_dict=feed)

            # print(1)
            # for bat_heads, bat_relations, bat_pos_tails, bat_neg_tails in triple_iter:
            #     feed = {self.h: bat_heads,
            #             self.r: bat_relations,
            #             self.pos_t: bat_pos_tails,
            #             self.neg_t: bat_neg_tails}
            #     user_embeddings, entity_embeddings, _ = self.sess.run([self.weights['user_embedding'],self.weights['entity_embedding'],self.opt_kge], feed_dict=feed)
            # print(2)
            # auc_list, f1_list = [], []
            # for bat_users, bat_pos_items, bat_neg_items in test_iter:
            #     feed = {self.users: bat_users,
            #             self.pos_items: bat_pos_items,
            #             self.neg_items: bat_neg_items}
            #
            #     feed.update({
            #         self.sub_mat['adj_values']: sub_mat['adj_values'],
            #         self.sub_mat['adj_indices']: sub_mat['adj_indices'],
            #         self.sub_mat['adj_shape']: sub_mat['adj_shape']})
            #
            #     auc, f1 = self.ctr_eval(feed)
            #     auc_list.append(auc)
            #     f1_list.append(f1)
            # final_auc, final_f1 = float(np.mean(auc_list)), float(np.mean(f1_list))
            final_auc, final_f1 = 1, 1
            feed = {
                self.sub_mat['adj_values']: sub_mat['adj_values'],
                self.sub_mat['adj_indices']: sub_mat['adj_indices'],
                self.sub_mat['adj_shape']: sub_mat['adj_shape']}
            result = self.evaluate_model(self.test_users, feed)
            info = get_info(result, epoch, final_auc, final_f1)
            self.logger.info(info)
            if not best_result or result['NDCG'][4] > best_result['NDCG'][4]:
                best_result, cur_epoch, cur_auc, cur_f1 = result, epoch, final_auc, final_f1
                # best_user_embeddings, best_entity_embeddings = user_embeddings, entity_embeddings
                step = 0
            else:
                step += 1

            if step >= self.stop_cnt or epoch == self.epochs - 1:
                info = get_info(best_result, cur_epoch, cur_auc, cur_f1)
                self.logger.info('-' * 27 + ' BEST RESULT ' + '-' * 27)
                self.logger.info(info)
                break
        # np.save('user_embedding_noVI.npy', best_user_embeddings)
        # np.save('entity_embedding_noVI.npy', best_entity_embeddings)

    # @timer
    def evaluate_model(self, users, feed):
        self.sess.run(self.assign_opt, feed_dict=feed)
        return self.evaluator.evaluate(self, users)

    def predict(self, users, candidate_items=None):
        feed_dict = {self.users: users}
        ratings = self.sess.run(self.batch_ratings, feed_dict=feed_dict)
        if candidate_items is not None:
            ratings = [ratings[idx][u_item] for idx, u_item in enumerate(candidate_items)]
        # print("ratings:", ratings)
        return ratings

    def ctr_eval(self, feed_dict):
        pos_scores, neg_scores = self.sess.run([self.pos_score_normalized, self.neg_score_normalized], feed_dict)
        scores = np.concatenate((pos_scores, neg_scores), axis=0)
        assert len(pos_scores) == len(neg_scores)
        labels = np.array(len(pos_scores) * [1] + len(neg_scores) * [0])
        auc = roc_auc_score(y_true=labels, y_score=scores)
        scores[scores >= 0.5] = 1
        scores[scores < 0.5] = 0
        f1 = f1_score(y_true=labels, y_pred=scores)
        return auc, f1
