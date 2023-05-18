import math
import multiprocessing as mp
import random
import time
import gc

import tensorflow as tf
import numpy as np
from sklearn import preprocessing
from openea.modules.finding.similarity import sim

from utils import MyKGs, find_alignment

def search_kg1_to_kg2_1nn_neighbor(embeds1, embeds2, ents2, mapping_mat, return_sim=False, soft_nn=10):
    if mapping_mat is not None:
        embeds1 = np.matmul(embeds1, mapping_mat) 
        embeds1 = preprocessing.normalize(embeds1)
    sim_mat = np.matmul(embeds1, embeds2.T)
    nearest_pairs = find_alignment(sim_mat, soft_nn)
    nns = [ents2[x[0][1]] for x in nearest_pairs]
    if return_sim:
        sim_list = []
        for pair in nearest_pairs:
            sim_list.append(sim_mat[pair[0][0], pair[0][1]]) 
        return nns, sim_list 
    return nns

def search_kg1_to_kg2_1nn_neighbor_new(embeds1, embeds2, ents2, mapping_mat, return_sim=False, soft_nn=10, metric="inner"):
    if mapping_mat is not None:
        embeds1 = np.matmul(embeds1, mapping_mat)
        embeds1 = preprocessing.normalize(embeds1)
    sim_mat = sim(embeds1, embeds2, metric=metric, normalize=True, csls_k=5)
    nearest_pairs = find_alignment(sim_mat, soft_nn)
    nns = [ents2[x[0][1]] for x in nearest_pairs]
    if return_sim:
        sim_list = []
        for pair in nearest_pairs:
            sim_list.append(sim_mat[pair[0][0], pair[0][1]])
        return nns, sim_list
    return nns

def search_kg1_to_kg2_ordered_all_nns(embeds1, embeds2, ents2, mapping_mat, return_all_sim=False, soft_nn=10):
    if mapping_mat is not None:
        embeds1 = np.matmul(embeds1, mapping_mat) 
        embeds1 = preprocessing.normalize(embeds1)
    sim_mat = np.matmul(embeds1, embeds2.T)
    nearest_pairs = find_alignment(sim_mat, soft_nn)
    nns = [ents2[x[0][1]] for x in nearest_pairs]
    if return_all_sim:
        sim_list = []
        for idx, pairs in enumerate(nearest_pairs):
            cur_sim_list = []
            for elements in pairs:
                s, t = elements[0], elements[1]
                cur_sim_list.append(sim_mat[s, t])
            idx_sorted = np.argsort(cur_sim_list)[::-1]
            nearest_pairs[idx] = np.array(pairs, dtype=np.int32)[idx_sorted]
            sim_list.append(np.array(cur_sim_list)[idx_sorted])
        nns = [[ents2[x[1]] for x in pairs] for pairs in nearest_pairs] 
        del sim_mat
        return nns, sim_list 
    del sim_mat
    return nns


def search_kg2_to_kg1_ordered_all_nns(embeds2, embeds1, ents1, mapping_mat, return_all_sim=False, soft_nn=10):
    if mapping_mat is not None:
        embeds1 = np.matmul(embeds1, mapping_mat) 
        embeds1 = preprocessing.normalize(embeds1)
    s2t_topk = embeds2.shape[1] 
    batch_size = embeds2.shape[0]
    embeds2 = embeds2.reshape((-1, embeds1.shape[1]))
    reversed_sim_mat = np.matmul(embeds1, embeds2.T).T
    nearest_pairs = find_alignment(reversed_sim_mat, soft_nn)
    nns = [ents1[x[0][1]] for x in nearest_pairs]
    if return_all_sim:
        sim_list = []
        for idx, pairs in enumerate(nearest_pairs):
            cur_sim_list = []
            for elements in pairs:
                t, s = elements[0], elements[1]
                cur_sim_list.append(reversed_sim_mat[t, s])
            idx_sorted = np.argsort(cur_sim_list)[::-1]
            nearest_pairs[idx] = np.array(pairs, dtype=np.int32)[idx_sorted]
            sim_list.append(np.array(cur_sim_list)[idx_sorted])
        nns = [[ents1[x[1]] for x in pairs] for pairs in nearest_pairs] 
        sim_list = np.array(sim_list).reshape(batch_size, -1) 
        del reversed_sim_mat
        return nns, sim_list 
    del reversed_sim_mat
    return nns
