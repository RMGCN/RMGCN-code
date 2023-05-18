import gc
import time
import os
import random
import math
import csv

import numpy as np

import ray
import scipy
import tensorflow as tf
import scipy.sparse as sp
from sklearn import preprocessing

from openea.modules.utils.util import generate_out_folder
from openea.modules.utils.util import task_divide, merge_dic
from openea.modules.finding.evaluation import early_stop
import openea.modules.load.read as rd
from openea.models.basic_model import BasicModel
from openea.modules.base.optimizers import generate_optimizer

from search_neighbor import search_kg1_to_kg2_1nn_neighbor, search_kg1_to_kg2_1nn_neighbor_new

from eval import valid, test, greedy_alignment, eval_margin, get_results, eval_margin_new
from analyse import show_raw_plot, double_gaussian, double_cauchy, double_laplace, gaussian, cauchy, laplace, h
from scipy.optimize import curve_fit, fsolve



def sparse_to_tuple(sparse_mx):
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)
    return sparse_mx


def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def func(triples):
    head = {}
    cnt = {}
    for tri in triples:
        if tri[1] not in cnt:
            cnt[tri[1]] = 1
            head[tri[1]] = {tri[0]}
        else:
            cnt[tri[1]] += 1
            head[tri[1]].add(tri[0])
    r2f = {}
    for r in cnt:
        r2f[r] = len(head[r]) / cnt[r]
    return r2f


def ifunc(triples):
    tail = {}
    cnt = {}
    for tri in triples:
        if tri[1] not in cnt:
            cnt[tri[1]] = 1
            tail[tri[1]] = {tri[2]}
        else:
            cnt[tri[1]] += 1
            tail[tri[1]].add(tri[2])
    r2if = {}
    for r in cnt:
        r2if[r] = len(tail[r]) / cnt[r]
    return r2if


def get_weighted_adj(e, triples):
    r2f = func(triples)
    r2if = ifunc(triples)
    M = {}
    for tri in triples:
        if tri[0] == tri[2]:
            continue
        if (tri[0], tri[2]) not in M:
            M[(tri[0], tri[2])] = max(r2if[tri[1]], 0.3)
        else:
            M[(tri[0], tri[2])] += max(r2if[tri[1]], 0.3)
        if (tri[2], tri[0]) not in M:
            M[(tri[2], tri[0])] = max(r2f[tri[1]], 0.3)
        else:
            M[(tri[2], tri[0])] += max(r2f[tri[1]], 0.3)
    row = []
    col = []
    data = []
    for key in M:
        row.append(key[1])
        col.append(key[0])
        data.append(M[key])
    data = np.array(data, dtype='float32')
    adj = sp.coo_matrix((data, (row, col)), shape=(e, e))
    adj = preprocess_adj(adj)
    return adj


def generate_rel_ht(triples):
    rel_ht_dict = dict()
    for h, r, t in triples:
        hts = rel_ht_dict.get(r, list())
        hts.append((h, t))
        rel_ht_dict[r] = hts
    return rel_ht_dict


def no_weighted_adj(total_ent_num, triple_list, is_two_adj=False):
    start = time.time()
    edge = dict()
    for item in triple_list:
        if item[0] not in edge.keys():
            edge[item[0]] = set()
        if item[2] not in edge.keys():
            edge[item[2]] = set()
        edge[item[0]].add(item[2])
        edge[item[2]].add(item[0])
    row = list()
    col = list()
    for i in range(total_ent_num):
        if i not in edge.keys():
            continue
        key = i
        value = edge[key]
        add_key_len = len(value)
        add_key = (key * np.ones(add_key_len)).tolist()
        row.extend(add_key)
        col.extend(list(value))
    data_len = len(row)
    data = np.ones(data_len)
    one_adj = sp.coo_matrix((data, (row, col)), shape=(total_ent_num, total_ent_num))
    one_adj = preprocess_adj(one_adj)
    print('generating one-adj costs time: {:.4f}s'.format(time.time() - start))
    if not is_two_adj:
        return one_adj, None
    expend_edge = dict()
    row = list()
    col = list()
    temp_len = 0
    for key, values in edge.items():
        if key not in expend_edge.keys():
            expend_edge[key] = set()
        for value in values:
            add_value = edge[value]
            for item in add_value:
                if item not in values and item != key:
                    expend_edge[key].add(item)
                    no_len = len(expend_edge[key])
                    if temp_len != no_len:
                        row.append(key)
                        col.append(item)
                    temp_len = no_len
    data = np.ones(len(row))
    two_adj = sp.coo_matrix((data, (row, col)), shape=(total_ent_num, total_ent_num))
    two_adj = preprocess_adj(two_adj)
    print('generating one- and two-adj costs time: {:.4f}s'.format(time.time() - start))
    return one_adj, two_adj


def enhance_triples(kg1, kg2, ents1, ents2):
    assert len(ents1) == len(ents2)
    print("before enhanced:", len(kg1.triples), len(kg2.triples))
    enhanced_triples1, enhanced_triples2 = set(), set()
    links1 = dict(zip(ents1, ents2))
    links2 = dict(zip(ents2, ents1))
    for h1, r1, t1 in kg1.triples:
        h2 = links1.get(h1, None)
        t2 = links1.get(t1, None)
        if h2 is not None and t2 is not None and t2 not in kg2.out_related_ents_dict.get(h2, set()):
            enhanced_triples2.add((h2, r1, t2))
    for h2, r2, t2 in kg2.triples:
        h1 = links2.get(h2, None)
        t1 = links2.get(t2, None)
        if h1 is not None and t1 is not None and t1 not in kg1.out_related_ents_dict.get(h1, set()):
            enhanced_triples1.add((h1, r2, t1))
    print("after enhanced:", len(enhanced_triples1), len(enhanced_triples2))
    return enhanced_triples1, enhanced_triples2


def dropout(inputs, drop_rate, noise_shape, is_sparse):
    if not is_sparse:
        return tf.nn.dropout(inputs, drop_rate)
    return sparse_dropout(inputs, drop_rate, noise_shape)


def sparse_dropout(x, drop_rate, noise_shape):
    keep_prob = 1 - drop_rate
    random_tensor = keep_prob
    random_tensor += tf.random.uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse.retain(x, dropout_mask)
    return pre_out * (1. / keep_prob)


def generate_neighbours(entity_embeds1, entity_list1, entity_embeds2, entity_list2, neighbors_num, threads_num=4):
    ent_frags = task_divide(np.array(entity_list1, dtype=np.int32), threads_num)
    ent_frag_indexes = task_divide(np.array(range(len(entity_list1)), dtype=np.int32), threads_num)
    dic = dict()
    rest = []
    for i in range(len(ent_frags)):
        res = find_neighbours.remote(ent_frags[i], entity_embeds1[ent_frag_indexes[i], :],
                                     np.array(entity_list2, dtype=np.int32),
                                     entity_embeds2, neighbors_num)
        rest.append(res)
    for res in ray.get(rest):
        dic = merge_dic(dic, res)
    gc.collect()
    return dic


@ray.remote(num_cpus=1)
def find_neighbours(frags, sub_embed1, entity_list2, embed2, k):
    dic = dict()
    sim_mat = np.matmul(sub_embed1, embed2.T)
    for i in range(sim_mat.shape[0]):
        sort_index = np.argpartition(-sim_mat[i, :], k)
        neighbors_index = sort_index[0:k]
        neighbors = entity_list2[neighbors_index].tolist()
        dic[frags[i]] = neighbors
    del sim_mat
    return dic


class AKG:
    def __init__(self, triples, ori_triples=None):
        self.triples = set(triples)
        self.triple_list = list(self.triples)
        self.triples_num = len(self.triples)

        self.heads = set([triple[0] for triple in self.triple_list])
        self.props = set([triple[1] for triple in self.triple_list])
        self.tails = set([triple[2] for triple in self.triple_list])
        self.ents = self.heads | self.tails

        print("triples num", self.triples_num)
        print("head ent num", len(self.heads))
        print("total ent num", len(self.ents))

        self.prop_list = list(self.props)
        self.ent_list = list(self.ents)
        self.prop_list.sort()
        self.ent_list.sort()

        if ori_triples is None:
            self.ori_triples = None
        else:
            self.ori_triples = set(ori_triples)

        self._generate_related_ents()
        self._generate_triple_dict()
        self._generate_ht()
        self.__generate_weight()

    def _generate_related_ents(self):
        self.out_related_ents_dict = dict()
        self.in_related_ents_dict = dict()
        for h, r, t in self.triple_list:
            out_related_ents = self.out_related_ents_dict.get(h, set())
            out_related_ents.add(t)
            self.out_related_ents_dict[h] = out_related_ents
            in_related_ents = self.in_related_ents_dict.get(t, set())
            in_related_ents.add(h)
            self.in_related_ents_dict[t] = in_related_ents

    def _generate_triple_dict(self):
        self.rt_dict, self.hr_dict = dict(), dict()
        for h, r, t in self.triple_list:
            rt_set = self.rt_dict.get(h, set())
            rt_set.add((r, t))
            self.rt_dict[h] = rt_set
            hr_set = self.hr_dict.get(t, set())
            hr_set.add((h, r))
            self.hr_dict[t] = hr_set

    def _generate_ht(self):
        self.ht = set()
        for h, r, t in self.triples:
            self.ht.add((h, t))

    def __generate_weight(self):
        triple_num = dict()
        n = 0
        for h, r, t in self.triples:
            if t in self.heads:
                n = n + 1
                triple_num[h] = triple_num.get(h, 0) + 1
                triple_num[t] = triple_num.get(t, 0) + 1
        self.weighted_triples = list()
        self.additional_triples = list()
        ave = math.ceil(n / len(self.heads))
        print("ave outs:", ave)
        for h, r, t in self.triples:
            w = 1
            if t in self.heads and triple_num[h] <= ave:
                w = 2.0
                self.additional_triples.append((h, r, t))
            self.weighted_triples.append((h, r, t, w))
        print("additional triples:", len(self.additional_triples))


class GraphConvolution:
    def __init__(self, input_dim, output_dim, adj,
                 num_features_nonzero,
                 dropout_rate=0.0,
                 name='GCN',
                 is_sparse_inputs=False,
                 activation=tf.tanh,
                 use_bias=True):
        self.activation = activation
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.adjs = [tf.SparseTensor(indices=am[0], values=am[1], dense_shape=am[2]) for am in adj]
        self.num_features_nonzero = num_features_nonzero
        self.dropout_rate = dropout_rate
        self.is_sparse_inputs = is_sparse_inputs
        self.use_bias = use_bias
        self.kernels = list()
        self.bias = list()
        self.name = name
        self.data_type = tf.float32
        self._get_variable()

    def _get_variable(self):
        for i in range(len(self.adjs)):
            self.kernels.append(tf.get_variable(self.name + '_kernel_' + str(i),
                                                shape=(self.input_dim, self.output_dim),
                                                initializer=tf.glorot_uniform_initializer(),
                                                regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                                                dtype=self.data_type))
        if self.use_bias:
            self.bias = tf.get_variable(self.name + '_bias', shape=[self.output_dim, ],
                                        initializer=tf.zeros_initializer(),
                                        dtype=self.data_type)

    def call(self, inputs):
        if self.dropout_rate > 0.0:
            inputs = dropout(inputs, self.dropout_rate, self.num_features_nonzero, self.is_sparse_inputs)
        hidden_vectors = list()
        for i in range(len(self.adjs)):
            pre_sup = tf.matmul(inputs, self.kernels[i], a_is_sparse=self.is_sparse_inputs)
            hidden_vector = tf.sparse_tensor_dense_matmul(tf.cast(self.adjs[i], tf.float32), pre_sup)
            hidden_vectors.append(hidden_vector)
        outputs = tf.add_n(hidden_vectors)
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def update_adj(self, adj):
        print("gcn update adj...")
        self.adjs = [tf.SparseTensor(indices=am[0], values=am[1], dense_shape=am[2]) for am in adj]


class GCN_vev:
    def __init__(self, 
                 entity_num,
                 triples_num,
                 edge,
                 name='GCN_v2e2v',
                 embed_dim=500,
                 ):
        self.size_v = entity_num
        self.size_e = triples_num
        self.name = name
        self.data_type = tf.float32
        
    def v2e(self, x_v, edge):
        edge_j, edge_i, _ = tf.unstack(edge, axis=1)
        deg = tf.zeros([self.size_e], dtype=tf.float32)
        ones = tf.ones_like(edge_i, dtype=tf.float32)
        idx = tf.expand_dims(edge_i, axis=1)
        updates = tf.expand_dims(ones, axis=1)
        
        deg = tf.scatter_nd_add(tf.Variable(tf.zeros([self.size_e, 1], dtype=tf.float32)), idx, updates)[:, 0]
        
        w = tf.gather(tf.math.pow(deg, -1),edge_i)
        indices = tf.stack([tf.concat([edge[:, 1], edge[:, 1]], axis=0), tf.concat([edge[:, 0], edge[:, 2]], axis=0)], axis=1)
        m = tf.SparseTensor(tf.cast(indices, tf.int64), tf.concat([w, w], axis=0), [self.size_e, self.size_v])
        
        slice = x_v.shape[1] // 5

        x_e1 = tf.sparse.sparse_dense_matmul(m, x_v[:, 0:slice])
        x_e2 = tf.sparse.sparse_dense_matmul(m, x_v[:, slice:2*slice])
        x_e3 = tf.sparse.sparse_dense_matmul(m, x_v[:, 2*slice:3*slice])
        x_e4 = tf.sparse.sparse_dense_matmul(m, x_v[:, 3*slice:4*slice])
        x_e5 = tf.sparse.sparse_dense_matmul(m, x_v[:, 4*slice:])

        x_e = tf.concat([x_e1, x_e2, x_e3, x_e4, x_e5], axis=1)
        del x_e1, x_e2, x_e3, x_e4, x_e5
        
        x_e = tf.nn.relu(x_e)
        return x_e


        
    
    def e2v(self, x_e, edge):
        edge_i, edge_j, edge_k = tf.unstack(edge, axis=1)
        ones = tf.ones_like(edge_i, dtype=tf.float32)
        idx = tf.expand_dims(edge_i,  axis=1)
        updates = tf.expand_dims(ones, axis=1)
        deg1 = tf.scatter_nd_add(tf.Variable(tf.zeros([self.size_v, 1], dtype=tf.float32)), idx, updates)[:, 0]
        ones = tf.ones_like(edge_k, dtype=tf.float32)
        idx = tf.expand_dims(edge_k,  axis=1)
        updates = tf.expand_dims(ones, axis=1)
        deg2 = tf.scatter_nd_add(tf.Variable(tf.zeros([self.size_v, 1], dtype=tf.float32)), idx, updates)[:, 0]
        deg1 = tf.math.pow(deg1, -1)
        deg1 = tf.where(tf.math.is_nan(deg1), tf.zeros_like(deg1), deg1)
        deg2 = tf.math.pow(deg2, -1)
        deg2 = tf.where(tf.math.is_nan(deg2), tf.zeros_like(deg2), deg2)
        
        w1 = tf.gather(deg1, edge_i)
        w2 = tf.gather(deg2, edge_k)

        indices = tf.stack([tf.concat([edge[:, 0], edge[:, 2]], axis=0), tf.concat([edge[:, 1], edge[:, 1]],axis=0)], axis=1)
        m = tf.SparseTensor(tf.cast(indices, tf.int64), tf.concat([w1, w2], axis=0), [self.size_v, self.size_e])
                
        slice = x_e.shape[1] // 5
      
        x_v1=tf.sparse.sparse_dense_matmul(m, x_e[:, :slice])
        x_v2=tf.sparse.sparse_dense_matmul(m, x_e[:, slice:2*slice])
        x_v3=tf.sparse.sparse_dense_matmul(m, x_e[:, 2*slice:3*slice])
        x_v4=tf.sparse.sparse_dense_matmul(m, x_e[:, 3*slice:4*slice])
        x_v5=tf.sparse.sparse_dense_matmul(m, x_e[:, 4*slice:])

        x_v = tf.concat([x_v1, x_v2, x_v3, x_v4, x_v5], axis=1)
        del x_v1, x_v2, x_v3, x_v4, x_v5
        
        x_v = tf.nn.relu(x_v)
        return x_v

    def call(self, inputs, edge):
        edge = tf.constant(edge)
        x_e = self.v2e(inputs, edge)
        x_v = tf.math.l2_normalize(self.e2v(x_e, edge), axis=1)
        return x_v


class RMGCN(BasicModel):
    def set_kgs(self, kgs):
        self.kgs = kgs
        self.kg1 = AKG(self.kgs.kg1.relation_triples_set)
        self.kg2 = AKG(self.kgs.kg2.relation_triples_set)

    def set_args(self, args):
        self.args = args
        self.out_folder = generate_out_folder(self.args.output, self.args.training_data, self.args.dataset_division,
                                              self.__class__.__name__)

    def init(self):
        self.ref_ent1 = self.kgs.test_entities1 + self.kgs.valid_entities1
        self.ref_ent2 = self.kgs.test_entities2 + self.kgs.valid_entities2
        self.sup_ent1 = self.kgs.train_entities1
        self.sup_ent2 = self.kgs.train_entities2
        self.linked_ents = set(self.kgs.train_entities1 +
                               self.kgs.train_entities2 +
                               self.kgs.valid_entities1 +
                               self.kgs.test_entities1 +
                               self.kgs.test_entities2 +
                               self.kgs.valid_entities2)
        enhanced_triples1, enhanced_triples2 = enhance_triples(self.kg1,
                                                               self.kg2,
                                                               self.sup_ent1,
                                                               self.sup_ent2)
        ori_triples = self.kg1.triple_list + self.kg2.triple_list   
        triples = ori_triples + list(enhanced_triples1) + list(enhanced_triples2)

        rel_ht_dict = generate_rel_ht(triples)
        
        saved_data_path = self.args.training_data + 'alinet_noweight_' + self.args.align_direction + 'saved_data.pkl'
        
        triples_set = set(triples)
        self.triples_set = triples_set
        
        one_adj, _ = no_weighted_adj(self.kgs.entities_num, triples, is_two_adj=False)
        
        adj = [one_adj]
        if self.is_two:
            dangling_ents = set(item[0] for item in self.kgs.train_unlinked_entities1+self.kgs.train_unlinked_entities2)
            triple_to_del = set(item for item in triples_set if item[0] in dangling_ents or item[2] in dangling_ents)
            triples_set = triples_set - triple_to_del
            triples = list(triples_set)
            self.masked_triples = triples
            two_adj, _ = no_weighted_adj(self.kgs.entities_num, triples, is_two_adj=False)
            adj.append(two_adj)
        self.adj = adj
        self.ori_adj = [adj[0]]

        self.rel_ht_dict = rel_ht_dict
        self.rel_win_size = self.args.min_rel_win

        sup_ent1 = np.array(self.sup_ent1).reshape((len(self.sup_ent1), 1))
        sup_ent2 = np.array(self.sup_ent2).reshape((len(self.sup_ent1), 1))
        weight = np.ones((len(self.kgs.train_entities1), 1), dtype=np.float)
        self.sup_links = np.hstack((sup_ent1, sup_ent2, weight))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)
        from tensorflow.python.keras.backend import set_session
        set_session(self.session)
        self._get_variable()

        if self.args.rel_param > 0.0:
            self._generate_rel_graph()
        else:
            self._generate_graph()

        if self.args.detection_mode == "margin":
            self._define_distance_margin_graph()

        if self.args.use_NCA_loss:
            self._define_NCA_loss_graph()

        self.saver = tf.train.Saver()
        
        tf.global_variables_initializer().run(session=self.session)

        if self.args.resume:
            self.saver.restore(self.session, self.args.resume_path+'model.ckpt')


    def __init__(self):
        super().__init__()
        self.adj = None
        self.one_hop_layers = None
        self.two_hop_layers = None
        self.layers_outputs = None
        self.new_edges1, self.new_edges2 = set(), set()
        self.new_links = set()
        self.pos_link_batch = None
        self.neg_link_batch = None
        self.sup_links_set = set()
        self.rel_ht_dict = None
        self.rel_win_size = None
        self.start_augment = None
        self.is_two = True
        self.new_sup_links_set = set()
        self.input_embeds, self.output_embeds_list = None, None
        self.sup_links = None
        self.model = None
        self.optimizer = None
        self.ref_ent1 = None
        self.ref_ent2 = None
        self.sup_ent1 = None
        self.sup_ent2 = None
        self.linked_ents = None
        self.session = None

    def _get_variable(self):
        self.init_embedding = tf.get_variable('init_embedding',
                                              shape=(self.kgs.entities_num, self.args.layer_dims[0]),
                                              initializer=tf.glorot_uniform_initializer(),
                                              dtype=tf.float32)

    def _define_model(self):
        print('Getting AliNet model...')
        layer_num = len(self.args.layer_dims) - 1
        output_embeds = self.init_embedding
        one_layers = list()
        two_layers = list()
        layers_outputs = list()

        gcnvev = GCN_vev(entity_num=self.kgs.entities_num,triples_num=self.kgs.relations_num, edge=list(self.triples_set))
        vev_output = gcnvev.call(output_embeds, list(self.triples_set))
        layers_outputs.append(vev_output)

        for i in range(layer_num):
            gcn_layer = GraphConvolution(input_dim=self.args.layer_dims[i],
                                          output_dim=self.args.layer_dims[i + 1],
                                          adj=[self.adj[1]],
                                          num_features_nonzero=self.args.num_features_nonzero,
                                          dropout_rate=0.0,
                                          name='gcn_' + str(i))
            one_layers.append(gcn_layer)
            one_output_embeds = gcn_layer.call(output_embeds)
            output_embeds = one_output_embeds
            layers_outputs.append(output_embeds)

        self.one_hop_layers = one_layers
        self.two_hop_layers = two_layers
        self.output_embeds_list = layers_outputs

    def compute_loss(self, pos_links, neg_links, only_pos=False):
        index1 = pos_links[:, 0]
        index2 = pos_links[:, 1]
        neg_index1 = neg_links[:, 0]
        neg_index2 = neg_links[:, 1]

        embeds_list = list()
        for output_embeds in self.output_embeds_list + [self.init_embedding]:
            output_embeds = tf.nn.l2_normalize(output_embeds, 1)
            embeds_list.append(output_embeds)
        output_embeds = tf.concat(embeds_list, axis=1)
        output_embeds = tf.nn.l2_normalize(output_embeds, 1)
        embeds1 = tf.nn.embedding_lookup(output_embeds, tf.cast(index1, tf.int32))
        embeds2 = tf.nn.embedding_lookup(output_embeds, tf.cast(index2, tf.int32))
        pos_loss = tf.reduce_sum(tf.reduce_sum(tf.square(embeds1 - embeds2), 1))
        embeds1 = tf.nn.embedding_lookup(output_embeds, tf.cast(neg_index1, tf.int32))
        embeds2 = tf.nn.embedding_lookup(output_embeds, tf.cast(neg_index2, tf.int32))
        neg_distance = tf.reduce_sum(tf.square(embeds1 - embeds2), 1)
        neg_loss = tf.reduce_sum(tf.keras.activations.relu(self.args.neg_margin - neg_distance))

        return pos_loss + self.args.neg_margin_balance * neg_loss

    def compute_rel_loss(self, hs, ts):
        embeds_list = list()
        for output_embeds in self.output_embeds_list + [self.init_embedding]:
            output_embeds = tf.nn.l2_normalize(output_embeds, 1)
            embeds_list.append(output_embeds)
        output_embeds = tf.concat(embeds_list, axis=1)
        output_embeds = tf.nn.l2_normalize(output_embeds, 1)
        h_embeds = tf.nn.embedding_lookup(output_embeds, tf.cast(hs, tf.int32))
        t_embeds = tf.nn.embedding_lookup(output_embeds, tf.cast(ts, tf.int32))
        r_temp_embeds = tf.reshape(h_embeds - t_embeds, [-1, self.rel_win_size, output_embeds.shape[-1]])
        r_temp_embeds = tf.reduce_mean(r_temp_embeds, axis=1, keepdims=True)
        r_embeds = tf.tile(r_temp_embeds, [1, self.rel_win_size, 1])
        r_embeds = tf.reshape(r_embeds, [-1, output_embeds.shape[-1]])
        r_embeds = tf.nn.l2_normalize(r_embeds, 1)
        return tf.reduce_sum(tf.reduce_sum(tf.square(h_embeds - t_embeds - r_embeds), 1)) * self.args.rel_param

    def _generate_graph(self):
        self.pos_links = tf.placeholder(tf.int32, shape=[None, 3], name="pos")
        self.neg_links = tf.placeholder(tf.int32, shape=[None, 2], name='neg')
        self._define_model()
        self.loss = self.compute_loss(self.pos_links, self.neg_links)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.args.learning_rate).minimize(self.loss)

    def _generate_rel_graph(self):
        print('Building relational embedding graph...')
        print("rel_win_size:", self.rel_win_size)
        self.pos_links = tf.placeholder(tf.int32, shape=[None, 3], name="pos")
        self.neg_links = tf.placeholder(tf.int32, shape=[None, 2], name='neg')
        self.hs = tf.placeholder(tf.int32, shape=[None], name="hs")
        self.ts = tf.placeholder(tf.int32, shape=[None], name='ts')
        self._define_model()
        self.loss = self.compute_loss(self.pos_links, self.neg_links) + \
                    self.compute_rel_loss(self.hs, self.ts)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.args.learning_rate).minimize(self.loss)

    def _define_distance_margin_graph(self):
        print("build distance margin graph...")
        with tf.name_scope('entity_placeholder'):
            self.input_ents1 = tf.placeholder(tf.int32, shape=[None])
            self.input_ents2 = tf.placeholder(tf.int32, shape=[None])
        with tf.name_scope('entity_lookup'):
            x1 = self.lookup_embeds(self.input_ents1)
            x2 = self.lookup_embeds(self.input_ents2)
            x1 = tf.nn.l2_normalize(x1, 1)
            x2 = tf.nn.l2_normalize(x2, 1)
        with tf.name_scope('dis_margin_loss'):
            dis = tf.reduce_sum(tf.square(x1 - x2), axis=1)
        self.dis_loss = tf.reduce_sum(tf.nn.relu(self.args.distance_margin - dis))
        self.dis_optimizer = tf.train.AdamOptimizer(learning_rate=self.args.learning_rate).minimize(self.dis_loss)
        print('finished')


    def _NCA_loss(self, embed1, embed2):
        self.NCA_embed1 = embed1
        with tf.name_scope('NCA_loss'):
            sim_scores = tf.matmul(embed1, tf.transpose(embed2))
            sim_diags = tf.diag(tf.linalg.tensor_diag_part(sim_scores))
            S_ = tf.exp(self.alpha * (sim_scores - self.ep))
            S_ = S_ - tf.diag(tf.linalg.tensor_diag_part(S_)) 
            self.S_ = S_
            loss_diag = -tf.log(1 + self.beta * tf.nn.relu(tf.reduce_sum(sim_diags, 0)))
            pos_scores = tf.log(1 + tf.reduce_sum(S_, 0)) / self.alpha
            neg_scores = tf.log(1 + tf.reduce_sum(S_, 1)) / self.alpha
            self.pos_scores = pos_scores
            self.neg_scores = neg_scores
            self.in_pos_scores = 1 + tf.reduce_sum(S_, 0)
            loss = tf.reduce_mean(pos_scores
                                    + neg_scores
                                    + loss_diag)
        return loss

    def _define_NCA_loss_graph(self):
        print("build NCA loss graph...")
        with tf.name_scope('NCA_input_placeholder'):
            self.NCA_input_ents1 = tf.placeholder(tf.int32, shape=[None])
            self.NCA_input_ents2 = tf.placeholder(tf.int32, shape=[None])
            self.alpha = tf.placeholder(tf.float32, shape=[])
            self.beta = tf.placeholder(tf.float32, shape=[])
            self.ep = tf.placeholder(tf.float32, shape=[])
        with tf.name_scope('NCA_lookup'):
            embed1 = self.lookup_embeds(self.NCA_input_ents1)
            embed2 = self.lookup_embeds(self.NCA_input_ents2)
            embed1 = tf.nn.l2_normalize(embed1, 1)
            embed2 = tf.nn.l2_normalize(embed2, 1)
            self.NCA_embed2 = embed2
        with tf.name_scope('NCA_loss'):
            self.NCA_loss = self._NCA_loss(embed1, embed2)
            self.NCA_optimizer = generate_optimizer(self.NCA_loss, self.args.learning_rate,
                                                    opt=self.args.optimizer)

    def get_source_and_candidates(self, source_ents_and_labels, is_test):
        total_ent_embeds = self.lookup_embeds(None).eval(session=self.session)
        source_ents = [x[0] for x in source_ents_and_labels]
        source_embeds = total_ent_embeds[np.array(source_ents),]
        if is_test:
            target_candidates = list(set(self.kgs.kg2.entities_list) -
                                     set(self.kgs.train_entities2) -
                                     set(self.kgs.valid_entities2))
        else:
            target_candidates = list(set(self.kgs.kg2.entities_list) - set(self.kgs.train_entities2))
        target_embeds = total_ent_embeds[np.array(target_candidates),]
        source_ent_y = [x[1] for x in source_ents_and_labels]
        return source_embeds, source_ents, source_ent_y, target_embeds, target_candidates, None

    def get_source_and_candidates_for_train(self, source_ents_and_labels, is_test):
        total_ent_embeds = self.lookup_embeds(None).eval(session=self.session)
        source_ents = [x[0] for x in source_ents_and_labels]
        source_embeds = total_ent_embeds[np.array(source_ents),] 
        target_candidates = self.kgs.kg2.entities_list
        target_embeds = total_ent_embeds[np.array(target_candidates),]
        source_ent_y = [x[1] for x in source_ents_and_labels]
        return source_embeds, source_ents, source_ent_y, target_embeds, target_candidates, None

    def evaluate_margin(self, source_ents_and_labels, margin, is_test=False):
        print("dangling entity detection...")
        source_embeds, source_ents, source_ent_y, target_embeds, target_candidates, mapping_mat = \
            self.get_source_and_candidates(source_ents_and_labels, is_test)
        nns, sims = search_kg1_to_kg2_1nn_neighbor_new(source_embeds, target_embeds, target_candidates, mapping_mat,
                                                   return_sim=True, soft_nn=1, metric=self.args.eval_metric)
        dis_vec = 1 - np.array(sims)
        mean_dis = np.mean(dis_vec)
        print(mean_dis, dis_vec)
        dis_list = dis_vec.tolist()
        source_label = [x[1] for x in source_ents_and_labels]
        with open(self.out_folder+f'test_label_and_sims.csv', mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            for row in [source_label]+[sims]:
                writer.writerow(row)
        return eval_margin(source_ents, dis_list, source_ent_y, margin=mean_dis)

    
    def two_step_evaluation_margin(self, matchable_source_ents1, dangling_source_ents1, threshold=-1, is_test=False):
        print("evaluating two-step alignment (margin)...")
        if is_test and self.args.test_batch_num > 1:
            final_label11_ents = list()
            final_label1_num = 0
            final_num_one_one, final_num_one_zero, final_num_zero_zero, final_num_zero_one, final_num_total_examples, \
            final_num_one_labels = 0, 0, 0, 0, 0, 0
            batch_num = self.args.test_batch_num
            print("test via batches...\n")
            matchable_source_ents1_tasks = task_divide(matchable_source_ents1, batch_num)
            dangling_source_ents1_tasks = task_divide(dangling_source_ents1, batch_num)
            for i in range(batch_num):
                label11_ents, label1_num, \
                num_one_one, num_one_zero, num_zero_zero, num_zero_one, num_total_examples, num_one_labels = \
                    self.evaluate_margin_new(matchable_source_ents1_tasks[i] +
                                            dangling_source_ents1_tasks[i],
                                            self.args.distance_margin, is_test=is_test)
                final_label11_ents += label11_ents
                final_label1_num += label1_num
                final_num_one_one += num_one_one
                final_num_one_zero += num_one_zero
                final_num_zero_zero += num_zero_zero
                final_num_zero_one += num_zero_one
                final_num_total_examples += num_total_examples
                final_num_one_labels += num_one_labels
                print()
            print("final test results:")
            get_results(final_num_one_one, final_num_one_zero, final_num_zero_zero, final_num_zero_one,
                        final_num_total_examples, final_num_one_labels)
        else:
            final_label11_ents, final_label1_num, \
            num_one_one, num_one_zero, num_zero_zero, num_zero_one, num_total_examples, num_one_labels = \
                self.evaluate_margin_new(matchable_source_ents1 + dangling_source_ents1,
                                        self.args.distance_margin, is_test=is_test)
        return self.real_entity_alignment_evaluation(final_label11_ents, final_label1_num, matchable_source_ents1)

    def real_entity_alignment_evaluation(self, label11_ents, label1_num, matchable_source_ents1):
        if label11_ents is None or len(label11_ents) == 0:
            print("no predicated matchable entities")
            return 0.
        total_ent_embeds = self.lookup_embeds(None).eval(session=self.session)
        label11_source_embeds = total_ent_embeds[np.array(label11_ents),]
        true_targets = []
        matchable_ents1 = self.kgs.valid_entities1 + self.kgs.test_entities1
        matchable_ents2 = self.kgs.valid_entities2 + self.kgs.test_entities2
        for e in label11_ents:
            idx = matchable_ents1.index(e)
            true_targets.append(matchable_ents2[idx])
        assert len(true_targets) == len(label11_ents)
        candidate_list = true_targets + list(self.kgs.kg2.entities_set
                                             - set(self.kgs.train_entities2)
                                             - set(self.kgs.valid_entities2)
                                             - set(true_targets))
        candidate_embeds = total_ent_embeds[np.array(candidate_list),]
        _, hits, _, _, _ = greedy_alignment(label11_source_embeds, candidate_embeds,
                                            self.args.top_k, self.args.test_threads_num,
                                            self.args.eval_metric, True, 10, False, False)
        hits1 = hits[0]
        hits10 = hits[2]
        precision = hits1 * len(label11_ents) / label1_num
        recall = hits1 * len(label11_ents) / len(matchable_source_ents1)
        f1 = 2 * precision * recall / (precision + recall)
        recall10 = hits10 * len(label11_ents) / len(matchable_source_ents1)
        print("two-step results, precision = {:.3f}, recall = {:.3f}, f1 = {:.3f}, recall@10 = {:.3f}\n".format(
            precision, recall, f1, recall10))
        return f1

    def _eval_valid_embeddings(self):
        ent1 = self.kgs.valid_entities1
        ent2 = self.kgs.valid_entities2 + list(self.kgs.kg2.entities_set
                                               - set(self.kgs.train_entities2)
                                               - set(self.kgs.valid_entities2))
        embeds1 = self.eval_embeddings(ent1)
        embeds2 = self.eval_embeddings(ent2)
        return embeds1, embeds2, None

    def _eval_test_embeddings(self):
        ent1 = self.kgs.test_entities1
        ent2 = self.kgs.test_entities2 + list(self.kgs.kg2.entities_set
                                              - set(self.kgs.train_entities2)
                                              - set(self.kgs.valid_entities2)
                                              - set(self.kgs.test_entities2))
        embeds1 = self.eval_embeddings(ent1)
        embeds2 = self.eval_embeddings(ent2)
        return embeds1, embeds2, None

    def save(self):
        with tf.device("/cpu:0"):
            embeds_list = list()
            input_embeds = self.init_embedding
            output_embeds_list = self.output_embeds_list
            for output_embeds in [input_embeds] + output_embeds_list:
                output_embeds = tf.nn.l2_normalize(output_embeds, 1)
                output_embeds = np.array(output_embeds.eval(session=self.session))
                embeds_list.append(output_embeds)
            ent_embeds = np.concatenate(embeds_list, axis=1)
            rd.save_embeddings(self.out_folder, self.kgs, ent_embeds, None, None, mapping_mat=None)

    def generate_input_batch(self, batch_size, neighbors1=None, neighbors2=None):
        if batch_size > len(self.sup_ent1):
            batch_size = len(self.sup_ent1)
        index = np.random.choice(len(self.sup_ent1), batch_size)
        pos_links = self.sup_links[index,]
        neg_links = list()
        if neighbors1 is None:
            neg_ent1 = list()
            neg_ent2 = list()
            for i in range(self.args.neg_triple_num):
                neg_ent1.extend(random.sample(self.kgs.kg1.entities_list, batch_size))
                neg_ent2.extend(random.sample(self.kgs.kg2.entities_list, batch_size))
            neg_links.extend([(neg_ent1[i], neg_ent2[i]) for i in range(len(neg_ent1))])
        else:
            for i in range(batch_size):
                e1 = pos_links[i, 0]
                candidates = random.sample(neighbors1.get(e1), self.args.neg_triple_num // 2)
                neg_links.extend([(e1, candidate) for candidate in candidates])
                e2 = pos_links[i, 1]
                candidates = random.sample(neighbors2.get(e2), self.args.neg_triple_num // 2)
                neg_links.extend([(candidate, e2) for candidate in candidates])

            neg_ent1 = list()
            neg_ent2 = list()
            for i in range(self.args.neg_triple_num):
                neg_ent1.extend(random.sample(self.kgs.kg1.entities_list, batch_size // 2))
                neg_ent2.extend(random.sample(self.kgs.kg2.entities_list, batch_size // 2))
            neg_links.extend([(neg_ent1[i], neg_ent2[i]) for i in range(len(neg_ent1))])
        neg_links = set(neg_links) - self.sup_links_set
        neg_links = neg_links - self.new_sup_links_set
        neg_links = np.array(list(neg_links))
        return pos_links, neg_links

    def generate_rel_batch(self):
        hs, rs, ts = list(), list(), list()
        for r, hts in self.rel_ht_dict.items():
            hts_batch = [random.choice(hts) for _ in range(self.rel_win_size)]
            for h, t in hts_batch:
                hs.append(h)
                ts.append(t)
                rs.append(r)
        return hs, rs, ts


    def lookup_3d_embeds(self, entities):
        input_embeds = self.init_embedding
        output_embeds_list = self.output_embeds_list
        res = []
        for output_embeds in [input_embeds] + output_embeds_list:
            res.append(tf.nn.l2_normalize(output_embeds, 1))
        embeds = tf.concat(res, axis=1)
        embeds = tf.nn.l2_normalize(embeds, 1)
        return tf.nn.embedding_lookup(embeds, entities)

    def lookup_embeds(self, entities):
        input_embeds = self.init_embedding
        output_embeds_list = self.output_embeds_list
        res = []
        for output_embeds in [input_embeds] + output_embeds_list:
            if entities is None:
                embeds1 = output_embeds
            else:
                embeds1 = tf.nn.embedding_lookup(output_embeds, entities)
            embeds1 = tf.nn.l2_normalize(embeds1, 1)
            res.append(embeds1)
        return tf.concat(res, axis=1)
    
    def lookup_dang_embeds(self, entities):
        input_embeds = self.init_embedding
        output_embeds = self.dang_embeds
        res = []
        for output_embeds in [input_embeds] + [output_embeds]:
            if entities is None:
                embeds1 = output_embeds
            else:
                embeds1 = tf.nn.embedding_lookup(output_embeds, entities)
            embeds1 = tf.nn.l2_normalize(embeds1, 1)
            res.append(embeds1)
        return tf.concat(res, axis=1)

    def lookup_last_embeds(self, entities):
        output_embeds = self.output_embeds_list[-1]
        if entities is None:
            embeds1 = output_embeds
        else:
            embeds1 = tf.nn.embedding_lookup(output_embeds, entities)
        embeds1 = tf.nn.l2_normalize(embeds1, 1)
        return embeds1

    def eval_embeddings(self, entity_list):
        embeds1 = self.lookup_embeds(entity_list)
        embeds1 = tf.nn.l2_normalize(embeds1, 1)
        return embeds1.eval(session=self.session)

    def generate_neighbors(self):
        t1 = time.time()
        assert 0.0 < self.args.truncated_epsilon < 1.0
        output_embeds_list = self.output_embeds_list
        total_ent_embeds = output_embeds_list[-1].eval(session=self.session)
        total_ent_embeds = preprocessing.normalize(total_ent_embeds)
        ents1 = self.sup_ent1
        ents2 = self.sup_ent2
        embeds1 = total_ent_embeds[np.array(ents1),]
        embeds2 = total_ent_embeds[np.array(ents2),]
        num1 = len(self.kgs.kg1.entities_list) // 2
        if len(self.kgs.kg1.entities_list) > 200000:
            num1 = len(self.kgs.kg1.entities_list) // 3
        kg1_random_ents = random.sample(self.kgs.kg1.entities_list, num1)
        random_embeds1 = total_ent_embeds[np.array(kg1_random_ents),]
        num2 = len(self.kgs.kg2.entities_list) // 2
        if len(self.kgs.kg2.entities_list) > 200000:
            num2 = len(self.kgs.kg2.entities_list) // 3
        kg2_random_ents = random.sample(self.kgs.kg2.entities_list, num2)
        random_embeds2 = total_ent_embeds[np.array(kg2_random_ents),]
        neighbors_num1 = int((1 - self.args.truncated_epsilon) * num1)
        neighbors_num2 = int((1 - self.args.truncated_epsilon) * num2)
        print("generating neighbors...", neighbors_num1, neighbors_num2)
        neighbors1 = generate_neighbours(embeds1, ents1, random_embeds2, kg2_random_ents, neighbors_num2,
                                         threads_num=self.args.test_threads_num)
        neighbors2 = generate_neighbours(embeds2, ents2, random_embeds1, kg1_random_ents, neighbors_num1,
                                         threads_num=self.args.test_threads_num)
        print("generating neighbors ({}, {}) costs {:.3f} s.".format(num1, num2, time.time() - t1))
        del embeds1, embeds2, total_ent_embeds, random_embeds1, random_embeds2
        return neighbors1, neighbors2

    def valid_alignment(self, stop_metric):
        print("\nevaluating synthetic alignment...")
        embeds1, embeds2, mapping = self._eval_valid_embeddings()
        hits, mrr_12, sim_list = valid(embeds1, embeds2, mapping, self.args.top_k,
                                       self.args.test_threads_num, metric=self.args.eval_metric,
                                       normalize=self.args.eval_norm, csls_k=0, accurate=False)
        print()
        return hits[0] if stop_metric == 'hits1' else mrr_12

    def test(self, save=False):
        with tf.device("/cpu:0"):
            print("\ntesting synthetic alignment...")
            embeds1, embeds2, mapping = self._eval_test_embeddings()
            _, _, _, sim_list = test(embeds1, embeds2, mapping, self.args.top_k, self.args.test_threads_num,
                                    metric=self.args.eval_metric, normalize=self.args.eval_norm, csls_k=0, accurate=True)
            test(embeds1, embeds2, mapping, self.args.top_k, self.args.test_threads_num,
                metric=self.args.eval_metric, normalize=self.args.eval_norm, csls_k=self.args.csls, accurate=True)
            print()
            if self.args.detection_mode == "margin" or self.args.detection_mode == "open":
                self.two_step_evaluation_margin(self.kgs.test_linked_entities1,
                                                self.kgs.test_unlinked_entities1, is_test=True)

    def launch_distance_margin_training_1epo(self, epoch, triple_steps):
        start = time.time()
        epoch_loss = 0
        trained_samples_num = 0
        training_data = self.kgs.train_unlinked_entities1
        batch_size = len(training_data) // triple_steps
        batch_size = int(min(batch_size, len(self.kgs.train_unlinked_entities1)) / 1.1)
        embeds = self.lookup_embeds(None).eval(session=self.session)
        mapping_mat = None
        embeds2 = embeds[np.array(self.kgs.kg2.entities_list),]
        steps_num = max(1, len(self.kgs.train_unlinked_entities1) // batch_size)
        for i in range(steps_num):
            batch_data1 = random.sample(training_data, batch_size)
            ent1 = [x[0] for x in batch_data1]
            embeds1 = embeds[np.array(ent1),]
            ent12 = search_kg1_to_kg2_1nn_neighbor(embeds1, embeds2, self.kgs.kg2.entities_list, mapping_mat, soft_nn=1)
            batch_loss, _ = self.session.run(fetches=[self.dis_loss, self.dis_optimizer],
                                             feed_dict={self.input_ents1: ent1,
                                                        self.input_ents2: ent12})
            epoch_loss += batch_loss
            trained_samples_num += len(batch_data1)
        print('epoch {}, margin loss: {:.8f}, cost time: {:.1f}s'.format(epoch, epoch_loss, time.time() - start))


    def launch_NCA_training_1epo(self, epoch, triple_steps):
        start = time.time()
        epoch_loss = 0
        trained_samples_num = 0
        batch_size = len(self.kgs.train_links) // triple_steps
        for i in range(triple_steps):
            if batch_size > len(self.kgs.train_links):
                batch_size = len(self.kgs.train_links) 
            links_batch = random.sample(self.kgs.train_links, batch_size)
            ent1 = [x[0] for x in links_batch]
            ent2 = [x[1] for x in links_batch]
            batch_loss, _, S_, pos_scores, neg_scores, in_pos_scores, embed1, embed2 = self.session.run(fetches=[self.NCA_loss, self.NCA_optimizer, self.S_, self.pos_scores, self.neg_scores, self.in_pos_scores, self.NCA_embed1, self.NCA_embed2],
                                             feed_dict={self.NCA_input_ents1: ent1,
                                                        self.NCA_input_ents2: ent2,
                                                        self.alpha: self.args.NCA_alpha,
                                                        self.beta: self.args.NCA_beta,
                                                        self.ep: 0.0})
            epoch_loss += batch_loss
            trained_samples_num += len(links_batch)
        epoch_loss /= trained_samples_num
        print('epoch {}, avg. NCA loss: {:.8f}, cost time: {:.1}s'.format(epoch, epoch_loss, time.time() - start))

    def run(self):
        print('start training...')
        print("output folder:", self.out_folder)
        if not os.path.exists(self.out_folder):
            os.makedirs(self.out_folder)
        steps = max(1, len(self.sup_ent2) // self.args.batch_size)
        neighbors1, neighbors2 = None, None
        if steps == 0:
            steps = 1
        for epoch in range(1, self.args.max_epoch + 1):
            start = time.time()
            epoch_loss = 0.0
            with tf.device('/gpu:0'):
                for step in range(steps):
                    self.pos_link_batch, self.neg_link_batch = self.generate_input_batch(self.args.batch_size,
                                                                                         neighbors1=neighbors1,
                                                                                         neighbors2=neighbors2)
                    feed_dict = {self.pos_links: self.pos_link_batch,
                                 self.neg_links: self.neg_link_batch}
                    if self.args.rel_param > 0.0:
                        hs, _, ts = self.generate_rel_batch()
                        feed_dict = {self.pos_links: self.pos_link_batch,
                                     self.neg_links: self.neg_link_batch,
                                     self.hs: hs, self.ts:ts}

                    fetches = {"loss": self.loss, "optimizer": self.optimizer}
                    results = self.session.run(fetches=fetches, feed_dict=feed_dict)
                    batch_loss = results["loss"]
                    epoch_loss += batch_loss
                print('epoch {}, loss: {:.8f}, cost time: {:.4f}s'.format(epoch, epoch_loss, time.time() - start))

            if self.args.use_NCA_loss:
                self.launch_NCA_training_1epo(epoch, steps)
            
            self.launch_distance_margin_training_1epo(epoch, steps)

            with tf.device("/gpu:0"):
                if epoch % self.args.eval_freq == 0 and epoch >= self.args.start_valid:
                    flag = self.valid_alignment(self.args.stop_metric)
                    if self.args.detection_mode == "margin" or self.args.detection_mode == "open":
                        flag = self.two_step_evaluation_margin(self.kgs.valid_linked_entities1,
                                                                   self.kgs.valid_unlinked_entities1)
                        self.flag1, self.flag2, self.early_stop = early_stop(self.flag1, self.flag2, flag)

                    if self.early_stop:
                        print("\n == training stop == \n")
                        break
                    self.save_model()

                    if epoch < self.args.max_epoch:
                        neighbors1, neighbors2 = self.generate_neighbors() 
                    
            tf.reset_default_graph()
        

    def evaluate_margin_new(self, source_ents_and_labels, margin, is_test=False):
        print("dangling entity detection...")
        source_embeds, source_ents, source_ent_y, target_embeds, target_candidates, mapping_mat = \
            self.get_source_and_candidates(source_ents_and_labels, is_test)

        nns, sims = search_kg1_to_kg2_1nn_neighbor_new(source_embeds, target_embeds, target_candidates, mapping_mat,
                                                   return_sim=True, soft_nn=1)
       
        if not is_test:
            mean_dis = np.mean(sims)
            return eval_margin_new(source_ents, sims, source_ent_y, margin=mean_dis)
        else:
            print("开始拟合")
            source_label = [x[1] for x in source_ents_and_labels]      
            xdata, ydata = show_raw_plot(sims)
            double_func_list = [double_gaussian, double_cauchy, double_laplace]
            single_func_list = [gaussian, cauchy, laplace]
            popt_list = []
            rss_list = []
            for double_func in double_func_list:
                popt, pcov = curve_fit(double_func, xdata, ydata, maxfev=5000, bounds=([0, -0.1, 0, 0, -0.1, 0], [1, 0.9, 2, 1, 0.9, 2]))
                residuals = ydata - double_func(xdata, *popt)
                rss = np.sum(residuals ** 2)    
                popt_list.append(popt)
                rss_list.append(rss)
            min_rss_idx = rss_list.index(min(rss_list))
            print("rss_list:", rss_list)
            print("popt:", popt_list)
            if min_rss_idx == 0:
                print("拟合为gaussian")
            elif min_rss_idx == 1:
                print("拟合为cauchy")
            elif min_rss_idx == 2:
                print("拟合为laplace")
            double_func = double_func_list[min_rss_idx] 
            single_func = single_func_list[min_rss_idx] 
            split = fsolve(h, 0.1, args=tuple([single_func]) + tuple(popt_list[min_rss_idx]))
            print("交点:", split)
            print("平均值点：", np.mean(sims))
            if split<0 or split >0.5:
                print("用交点分割的值")
                eval_margin_new(source_ents, sims, source_ent_y, margin=split[0])
                print("split <0 or split >0.5, using mean split")
                print("用均值点分割的值，返回以下值给第二步")
                return eval_margin_new(source_ents, sims, source_ent_y, margin=np.mean(sims))
            
            print("用均值点分割的值")
            eval_margin_new(source_ents, sims, source_ent_y, margin=np.mean(sims))
            print("用交点分割的值，并返回以下值给第二步")
            return eval_margin_new(source_ents, sims, source_ent_y, margin=split[0])


    def save_model(self):
        with tf.device("/cpu:0"):
            print('saving the model...')
            save_path = self.saver.save(self.session, self.out_folder+f'model.ckpt')
            print(f'Model saved in {save_path}')

