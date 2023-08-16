import tensorflow as tf
from abc import abstractmethod

LAYER_IDS = {}


def get_layer_id(layer_name=''):
    if layer_name not in LAYER_IDS:
        LAYER_IDS[layer_name] = 0
        return 0
    else:
        LAYER_IDS[layer_name] += 1
        return LAYER_IDS[layer_name]


class Aggregator(object):
    def __init__(self, n_neighbor, batch_size, dim, dropout, act, feature_transform, activation, att_type, name, agg):
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_id(layer))
        self.name = name
        self.dropout = dropout
        self.act = act
        self.batch_size = batch_size
        self.dim = dim
        self.n_neighbors = n_neighbor
        self.top = self.n_neighbors // 2
        self.feature_transform = feature_transform
        self.activation = activation
        self.att_type = att_type
        self.agg = agg

    def __call__(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, masks):
        outputs = self._call(self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, masks)
        return outputs

    @abstractmethod
    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, masks):
        # dimension:
        # self_vectors: [batch_size, -1, dim]
        # neighbor_vectors: [batch_size, -1, n_neighbor, dim]
        # neighbor_relations: [batch_size, -1, n_neighbor, dim]
        # user_embeddings: [batch_size, dim]
        pass

    def _mix_neighbor_vectors(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings):
        avg = False
        if not avg:
            # [batch_size, 1, 1, dim]
            user_embeddings = tf.reshape(user_embeddings, [self.batch_size, 1, 1, self.dim])
            self_vectors = tf.reshape(self_vectors, [self.batch_size, -1, 1, self.dim])

            # [batch_size, -1, n_neighbor]
            user_relation_scores = tf.reduce_sum(user_embeddings * neighbor_relations, axis=-1)
            self_relation_scores = tf.reduce_sum(self_vectors * neighbor_relations, axis=-1)
            self_neighbor_scores = tf.reduce_sum(self_vectors * neighbor_vectors, axis=-1)
            if self.att_type == 'ur':
                user_relation_scores_normalized = tf.nn.softmax(user_relation_scores, dim=-1)
            elif self.att_type == 'cr':
                user_relation_scores_normalized = tf.nn.softmax(self_relation_scores, dim=-1)
            elif self.att_type == 'ur+cr':
                user_relation_scores_normalized = tf.nn.softmax(user_relation_scores + self_relation_scores, dim=-1)
            elif self.att_type == 'ur+cn':
                user_relation_scores_normalized = tf.nn.softmax(user_relation_scores + self_neighbor_scores, dim=-1)
            elif self.att_type == 'ur+cr+cn':
                user_relation_scores_normalized = tf.nn.softmax(user_relation_scores + self_relation_scores +
                                                                self_neighbor_scores, dim=-1)
            else:
                raise Exception("Unknown attention type: " + self.att_type)

            # [batch_size, -1, n_neighbor, 1]
            user_relation_scores_normalized = tf.expand_dims(user_relation_scores_normalized, axis=-1)

            # [batch_size, -1, n_neighbor, dim]
            neighbors_aggregated = user_relation_scores_normalized * neighbor_vectors
            if self.agg == 'sum':
                neighbors_aggregated = tf.reduce_sum(neighbors_aggregated, axis=2)
            else:
                neighbors_aggregated = tf.reduce_mean(neighbors_aggregated, axis=2)
        else:
            # [batch_size, -1, dim]
            neighbors_aggregated = tf.reduce_mean(neighbor_vectors, axis=2)

        return neighbors_aggregated


class SumAggregator(Aggregator):
    def __init__(self, n_neighbor, batch_size, dim, dropout=0., act=tf.nn.relu, feature_transform=True, activation=True, att_type='ur', name=None, agg=None):
        super(SumAggregator, self).__init__(n_neighbor, batch_size, dim, dropout, act, feature_transform, activation, att_type, name, agg)

        with tf.variable_scope(self.name):
            self.weights = tf.get_variable(
                shape=[self.dim, self.dim], initializer=tf.contrib.layers.xavier_initializer(), name='weights')
            self.bias = tf.get_variable(shape=[self.dim], initializer=tf.zeros_initializer(), name='bias')

    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, masks=None):
        # [batch_size, -1, dim]
        neighbors_agg = self._mix_neighbor_vectors(self_vectors, neighbor_vectors, neighbor_relations, user_embeddings)

        output = tf.reshape(self_vectors + neighbors_agg, [-1, self.dim])
        output = tf.nn.dropout(output, keep_prob=1-self.dropout)
        if self.feature_transform:
            output = tf.matmul(output, self.weights) + self.bias
        else:
            output = output

        # [batch_size, -1, dim]
        output = tf.reshape(output, [self.batch_size, -1, self.dim])

        if self.activation:
            return self.act(output)
        else:
            return output


class ConcatAggregator(Aggregator):
    def __init__(self, n_neighbor, batch_size, dim, dropout=0., act=tf.nn.relu, feature_transform=True, activation=True, att_type='ur', name=None, agg=None):
        super(ConcatAggregator, self).__init__(n_neighbor, batch_size, dim, dropout, act, feature_transform, activation, att_type, name, agg)

        with tf.variable_scope(self.name):
            self.weights = tf.get_variable(
                shape=[self.dim * 2, self.dim], initializer=tf.contrib.layers.xavier_initializer(), name='weights')
            self.bias = tf.get_variable(shape=[self.dim], initializer=tf.zeros_initializer(), name='bias')

    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, masks):
        # [batch_size, -1, dim]
        neighbors_agg = self._mix_neighbor_vectors(self_vectors, neighbor_vectors, neighbor_relations, user_embeddings)
        # [batch_size, -1, dim * 2]
        output = tf.concat([self_vectors, neighbors_agg], axis=-1)

        # [-1, dim * 2]
        output = tf.reshape(output, [-1, self.dim * 2])
        output = tf.nn.dropout(output, keep_prob=1-self.dropout)

        # [-1, dim]
        if self.feature_transform:
            output = tf.matmul(output, self.weights) + self.bias
        else:
            output = output

        # [batch_size, -1, dim]
        output = tf.reshape(output, [self.batch_size, -1, self.dim])

        if self.activation:
            return self.act(output)
        else:
            return output


class NeighborAggregator(Aggregator):
    def __init__(self, n_neighbor, batch_size, dim, dropout=0., act=tf.nn.relu, feature_transform=True, activation=True, att_type='ur', name=None, agg=None):
        super(NeighborAggregator, self).__init__(n_neighbor, batch_size, dim, dropout, act, feature_transform, activation, att_type, name, agg)

        with tf.variable_scope(self.name):
            self.weights = tf.get_variable(
                shape=[self.dim, self.dim], initializer=tf.contrib.layers.xavier_initializer(), name='weights')
            self.bias = tf.get_variable(shape=[self.dim], initializer=tf.zeros_initializer(), name='bias')

    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, masks):
        # [batch_size, -1, dim]
        neighbors_agg = self._mix_neighbor_vectors(self_vectors, neighbor_vectors, neighbor_relations, user_embeddings)

        # [-1, dim]
        output = tf.reshape(neighbors_agg, [-1, self.dim])
        output = tf.nn.dropout(output, keep_prob=1-self.dropout)
        if self.feature_transform:
            output = tf.matmul(output, self.weights) + self.bias
        else:
            output = output

        # [batch_size, -1, dim]
        output = tf.reshape(output, [self.batch_size, -1, self.dim])

        if self.activation:
            return self.act(output)
        else:
            return output


class TopAggregator(Aggregator):
    def __init__(self, n_neighbor, batch_size, dim, dropout=0., act=tf.nn.relu, feature_transform=True, activation=True, att_type='ur', name=None, agg=None):
        super(TopAggregator, self).__init__(n_neighbor, batch_size, dim, dropout, act, feature_transform, activation, att_type, name, agg)

        with tf.variable_scope(self.name):
            self.weights = tf.get_variable(
                shape=[self.dim, self.dim], initializer=tf.contrib.layers.xavier_initializer(), name='weights')
            self.bias = tf.get_variable(shape=[self.dim], initializer=tf.zeros_initializer(), name='bias')

    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, masks):
        neighbors_agg = self._mix_neighbor_vectors(self_vectors, neighbor_vectors, neighbor_relations, user_embeddings)
        output = tf.reshape(self_vectors + neighbors_agg, [-1, self.dim])
        output = tf.nn.dropout(output, keep_prob=1-self.dropout)
        if self.feature_transform:
            output = tf.matmul(output, self.weights) + self.bias
        else:
            output = output

        # [batch_size, -1, dim]
        output = tf.reshape(output, [self.batch_size, -1, self.dim])

        if self.activation:
            return self.act(output)
        else:
            return output

    def _mix_neighbor_vectors(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings):
        avg = False
        if not avg:
            # [batch_size, 1, 1, dim]
            user_embeddings = tf.reshape(user_embeddings, [self.batch_size, 1, 1, self.dim])
            self_vectors = tf.reshape(self_vectors, [self.batch_size, -1, 1, self.dim])

            # [batch_size, -1, n_neighbor]
            user_relation_scores = tf.reduce_sum(user_embeddings * neighbor_relations, axis=-1)
            self_relation_scores = tf.reduce_sum(self_vectors * neighbor_relations, axis=-1)
            self_neighbor_scores = tf.reduce_sum(self_vectors * neighbor_vectors, axis=-1)
            if self.att_type == 'ur':
                user_relation_scores_normalized = tf.nn.softmax(user_relation_scores, dim=-1)
            elif self.att_type == 'ur+cr':
                user_relation_scores_normalized = tf.nn.softmax(user_relation_scores + self_relation_scores, dim=-1)
            elif self.att_type == 'ur+cn':
                user_relation_scores_normalized = tf.nn.softmax(user_relation_scores + self_neighbor_scores, dim=-1)
            elif self.att_type == 'ur+cr+cn':
                user_relation_scores_normalized = tf.nn.softmax(user_relation_scores + self_relation_scores +
                                                                self_neighbor_scores, dim=-1)
            else:
                raise Exception("Unknown attention type: " + self.att_type)

            # [batch_size, -1, n_neighbor, 1]
            user_relation_scores_normalized = tf.expand_dims(user_relation_scores_normalized, axis=-1)

            # [batch_size, -1, n_neighbor, dim]
            neighbors_aggregated = user_relation_scores_normalized * neighbor_vectors
            neighbors_aggregated = tf.transpose(neighbors_aggregated, [0, 1, 3, 2])
            neighbors_aggregated = tf.nn.top_k(neighbors_aggregated, k=self.top)[0]
            neighbors_aggregated = tf.reduce_mean(neighbors_aggregated, axis=-1)
        else:
            # [batch_size, -1, dim]
            neighbors_aggregated = tf.reduce_mean(neighbor_vectors, axis=2)

        return neighbors_aggregated


class DiAggregator(Aggregator):
    def __init__(self, n_neighbor, batch_size, dim, dropout=0., act=tf.nn.relu, feature_transform=True, activation=True, att_type='ur', name=None, agg=None):
        super(DiAggregator, self).__init__(n_neighbor, batch_size, dim, dropout, act, feature_transform, activation, att_type, name, agg)

        with tf.variable_scope(self.name):
            self.weights = tf.get_variable(
                shape=[self.dim, self.dim], initializer=tf.contrib.layers.xavier_initializer(), name='weights')
            self.bias = tf.get_variable(shape=[self.dim], initializer=tf.zeros_initializer(), name='bias')

    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, masks):
        # [batch_size, -1, dim]
        neighbors_agg = tf.reduce_mean(neighbor_vectors, axis=2)

        # [-1, dim]
        output = tf.reshape(self_vectors + neighbors_agg, [-1, self.dim])
        output = tf.nn.dropout(output, keep_prob=1-self.dropout)
        if self.feature_transform:
            output = tf.matmul(output, self.weights) + self.bias
        else:
            output = output

        # [batch_size, -1, dim]
        output = tf.reshape(output, [self.batch_size, -1, self.dim])

        if self.activation:
            return self.act(output)
        else:
            return output


class LabelAggregator(Aggregator):
    def __init__(self, n_neighbor, batch_size, dim, dropout=0., act=tf.nn.relu, feature_transform=True, activation=True, att_type='ur', name=None, agg=None):
        super(LabelAggregator, self).__init__(n_neighbor, batch_size, dim, dropout, act, feature_transform, activation, att_type, name, agg)

    def _call(self, self_labels, neighbor_labels, neighbor_relations, user_embeddings, masks):
        # [batch_size, 1, 1, dim]
        user_embeddings = tf.reshape(user_embeddings, [self.batch_size, 1, 1, self.dim])

        # [batch_size, -1, n_neighbor]
        user_relation_scores = tf.reduce_mean(user_embeddings * neighbor_relations, axis=-1)
        user_relation_scores_normalized = tf.nn.softmax(user_relation_scores, dim=-1)

        # [batch_size, -1]
        neighbors_aggregated = tf.reduce_mean(user_relation_scores_normalized * neighbor_labels, axis=-1)
        output = tf.cast(masks, tf.float32) * self_labels + tf.cast(
            tf.logical_not(masks), tf.float32) * neighbors_aggregated

        return output