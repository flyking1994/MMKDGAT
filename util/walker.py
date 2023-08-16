import numpy as np
import networkx as nx
from joblib import Parallel, delayed
import random
import itertools
import collections
import os
import random


class Graph(object):
    def __init__(self, nx_G, is_directed=False, p=0, q=0, alpha=0, dataset=None):
        self.G = nx_G
        self.is_directed = is_directed
        self.p = p
        self.q = q
        self.alpha = alpha
        self.dataset = dataset

    def node2vec_walk(self, walk_length, start_node):
        '''
        Simulate a random walk starting from start node
        :param walk_length:
        :param start_node:
        :return:
        '''
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0], alias_edges[(prev, cur)][1])]
                    walk.append(next)
            else:
                break
        return walk

    def simulate_walks(self, num_walks, walk_length, workers=1, verbose=0):
        '''
        Repeatedly simulate random walks from each node
        :param num_walks:
        :param walk_length:
        :return:
        '''
        G = self.G

        nodes = list(G.nodes())
        # results = Parallel(n_jobs=workers, verbose=verbose, )(
        #     delayed(self._simulate_walks)(nodes, num, walk_length) for num in
        #     partition_num(num_walks, workers))
        #
        # walks = list(itertools.chain(*results))

        walks = []
        for _ in range(num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))

        return walks

    def _simulate_walks(self, nodes, num_walks, walk_length,):
        walks = []
        for _ in range(num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))

        return walks

    def get_alias_edge(self, src, dst):
        '''
        Get the alias edge setup lists for a given edge
        :param src: previous visited node
        :param dst: current node
        :return:
        '''
        G = self.G
        p = self.p
        q = self.q
        alpha = self.alpha

        unnormalized_probs = []
        for dst_nbr in G.neighbors(dst):
            weight = G[dst][dst_nbr]['weight']
            if weight == alpha:
                unnormalized_probs.append(alpha)
            else:
                if dst_nbr == src:  # d(src, dst_nbr) == 0, return to the previous node
                    unnormalized_probs.append(weight/p)
                elif G.has_edge(dst_nbr, src):  # d(src, dst_nbr) == 1, walk to a neighbor of the previous node
                    unnormalized_probs.append(weight)
                else:  # d(src, dst_nbr) == 1, walk to a neighbor of the current node
                    unnormalized_probs.append(weight/q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob)/norm_const for u_prob in unnormalized_probs]

        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self):
        '''
        Preprocessing of transition probalities for guiding the random walks
        :return:
        '''
        root = './random_walk'
        alias_nodes_path = os.path.join(root, self.dataset+'_p'+str(self.p)+'_q'+str(self.q)+'_alias_nodes.npy')
        alias_edges_path = os.path.join(root, self.dataset+'_p'+str(self.p)+'_q'+str(self.q)+'_alias_edges.npy')
        if os.path.exists(alias_nodes_path) and os.path.join(alias_edges_path):
            alias_nodes = np.load(alias_nodes_path, allow_pickle=True).item()
            alias_edges = np.load(alias_edges_path, allow_pickle=True).item()

        else:
            G = self.G
            is_directed = self.is_directed

            alias_nodes = {}
            for node in G.nodes():
                # if the current node is a user, the next candidate node is all items
                # if the current node is an item, the next candidate node is all users
                unnormalized_probs = [G[node][nbr]['weight'] for nbr in G.neighbors(node)]
                norm_const = sum(unnormalized_probs)
                normalized_probs = [float(u_prob)/norm_const for u_prob in unnormalized_probs]
                alias_nodes[node] = alias_setup(normalized_probs)

            alias_edges = {}
            if is_directed:
                for edge in G.edges():
                    alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
            else:
                for edge in G.edges():
                    alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                    alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

            np.save(alias_nodes_path, alias_nodes)
            np.save(alias_edges_path, alias_edges)

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges
        return


def alias_setup(probs):
    '''
    Compute utility lists for non-uniform sampling from discrete distributions
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
	for details
    :param probs:
    :return:
    '''
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    '''
    Draw sample from a non-uniform discrete distribution using alias sampling
    :param J:
    :param q:
    :return:
    '''
    K = len(J)

    kk = int(np.float(np.random.rand() * K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]


def partition_num(num, workers):
    if num % workers == 0:
        return [num//workers]*workers
    else:
        return [num//workers]*workers + [num % workers]