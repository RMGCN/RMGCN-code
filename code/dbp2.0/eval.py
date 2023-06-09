import gc
import time
import ray
import numpy as np
from openea.modules.finding.similarity import sim
from openea.modules.utils.util import task_divide, merge_dic


def calculate_rank(idx, sim_mat, top_k, accurate, total_num):
    assert 1 in top_k
    mr = 0
    mrr = 0
    hits = [0] * len(top_k)
    hits1_rest = set()
    for i in range(len(idx)):
        gold = idx[i] 
        if accurate:
            rank = (-sim_mat[i, :]).argsort()
        else:
            rank = np.argpartition(-sim_mat[i, :], np.array(top_k) - 1) 
        hits1_rest.add((gold, rank[0]))
        assert gold in rank
        rank_index = np.where(rank == gold)[0][0]
        mr += (rank_index + 1)
        mrr += 1 / (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                hits[j] += 1
    mr /= total_num
    mrr /= total_num
    return mr, mrr, hits, hits1_rest


def greedy_alignment(embed1, embed2, top_k, nums_threads, metric, normalize, csls_k, accurate, is_print):
    t = time.time()
    sim_mat = sim(embed1, embed2, metric=metric, normalize=normalize, csls_k=csls_k)
    num = sim_mat.shape[0]
    if nums_threads > 1:
        hits = [0] * len(top_k)
        mr, mrr = 0, 0
        alignment_rest = set()
        rests = list()
        search_tasks = task_divide(np.array(range(num)), nums_threads)
    

        for task in search_tasks:
            mat = sim_mat[task, :]
            res = calculate_rank.remote(task, mat, top_k, accurate, num)
            rests.append(res)
        for rest in ray.get(rests):
            sub_mr, sub_mrr, sub_hits, sub_hits1_rest = rest
            mr += sub_mr
            mrr += sub_mrr
            hits += np.array(sub_hits)
            alignment_rest |= sub_hits1_rest
    else:
        mr, mrr, hits, alignment_rest = calculate_rank(list(range(num)), sim_mat, top_k, accurate, num)
    assert len(alignment_rest) == num
    hits = np.array(hits) / num
    for i in range(len(hits)):
        hits[i] = round(hits[i], 3)
    cost = time.time() - t
    if is_print:
        if accurate:
            if csls_k > 0:
                print("accurate results with csls: csls={}, hits@{} = {}, mr = {:.3f}, mrr = {:.6f}, time = {:.3f} s ".
                      format(csls_k, top_k, hits, mr, mrr, cost))
            else:
                print("accurate results: hits@{} = {}, mr = {:.3f}, mrr = {:.6f}, time = {:.3f} s ".
                      format(top_k, hits, mr, mrr, cost))
        else:
            if csls_k > 0:
                print("quick results with csls: csls={}, hits@{} = {}, time = {:.3f} s ".format(csls_k, top_k, hits,
                                                                                                cost))
            else:
                print("quick results: hits@{} = {}, time = {:.3f} s ".format(top_k, hits, cost))

    sim_list = []
    for i, j in alignment_rest:
        sim_list.append(sim_mat[i, j])
    del sim_mat
    gc.collect()
    return alignment_rest, hits, mr, mrr, sim_list


def calculate_rank_new(hit_sim, idx, sim_mat, top_k, accurate, total_num):
    assert 1 in top_k
    mr = 0
    mrr = 0
    hits = [0] * len(top_k)
    hits1_rest = set()
    for i in range(len(idx)):
        gold = idx[i]
        if accurate:
            rank = (-sim_mat[i, :]).argsort()
        else:
            rank = np.argpartition(-sim_mat[i, :], np.array(top_k) - 1)
        hits1_rest.add((gold, rank[0]))
        assert gold in rank
        rank_index = np.where(rank == gold)[0][0]
        mr += (rank_index + 1)
        mrr += 1 / (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                hits[j] += 1
                hit_sim[j].append(sim_mat[i][rank_index])

    mr /= total_num
    mrr /= total_num
    return mr, mrr, hits, hits1_rest


def valid(embeds1, embeds2, mapping, top_k, threads_num, metric='inner', normalize=False, csls_k=0, accurate=False,
          is_print=True):
    if mapping is None:
        _, hits1_12, mr_12, mrr_12, sim_list = greedy_alignment(embeds1, embeds2, top_k, threads_num,
                                                                metric, normalize, csls_k, accurate, is_print)
    else:
        test_embeds1_mapped = np.matmul(embeds1, mapping)
        _, hits1_12, mr_12, mrr_12, sim_list = greedy_alignment(test_embeds1_mapped, embeds2, top_k, threads_num,
                                                                metric, normalize, csls_k, accurate, is_print)
    return hits1_12, mrr_12, sim_list


def test(embeds1, embeds2, mapping, top_k, threads_num, metric='inner', normalize=False, csls_k=0, accurate=True,
         is_print=True):
    if mapping is None:
        alignment_rest_12, hits1_12, mr_12, mrr_12, sim_list = greedy_alignment(embeds1, embeds2, top_k, threads_num,
                                                                                metric, normalize, csls_k, accurate,
                                                                                is_print)
    else:
        test_embeds1_mapped = np.matmul(embeds1, mapping)
        alignment_rest_12, hits1_12, mr_12, mrr_12, sim_list = greedy_alignment(test_embeds1_mapped, embeds2, top_k,
                                                                                threads_num,
                                                                                metric, normalize, csls_k, accurate,
                                                                                is_print)
    return alignment_rest_12, hits1_12, mrr_12, sim_list


def get_results(num_one_one, num_one_zero, num_zero_zero, num_zero_one, num_total_examples, num_one_labels):
    precision, recall, f1, accu = 0.0, 0.0, 0.0, 0.0
    assert num_one_one + num_one_zero + num_zero_zero + num_zero_one == num_total_examples
    if num_one_one > 0:
        precision = num_one_one / (num_one_one + num_zero_one)
        print('precision = {:.3f}, '.format(precision))
    if num_one_one > 0:
        recall = num_one_one / num_one_labels
        print('recall = {:.3f}, '.format(recall))
        print('f1 = {:.3f}, '.format(2 * precision * recall / (precision + recall)))
    if num_one_one + num_zero_zero > 0:
        print('accuracy = {:.3f}, '.format((num_one_one + num_zero_zero) / num_total_examples))


def eval_margin(input_ents, dis_list, true_labels, margin):
    num_one_labels = true_labels.count(1.0) 
    num_zero_labels = true_labels.count(0.0)
    num_total_examples = len(true_labels)
    assert num_one_labels + num_zero_labels == num_total_examples == len(true_labels)
    num_one_one, num_one_zero, num_zero_zero, num_zero_one = 0, 0, 0, 0

    predicated_match_ents = []
    for i in range(num_total_examples):
        y = true_labels[i]
        dis = dis_list[i]
        if y == 1.0 and dis > margin: 
            num_one_one += 1
        elif y == 1.0 and dis <= margin:
            num_one_zero += 1
        elif y == 0.0 and dis > margin:
            num_zero_one += 1
        else:
            num_zero_zero += 1
            predicated_match_ents.append(input_ents[i])

    get_results(num_one_one, num_one_zero, num_zero_zero, num_zero_one, num_total_examples, num_one_labels)

    print("matchable and predicted matchable: {}; predicated matchable: {}".format(len(predicated_match_ents),
                                                                                   num_zero_zero + num_one_zero))

    return predicated_match_ents, num_zero_zero + num_one_zero, \
           num_one_one, num_one_zero, num_zero_zero, num_zero_one, num_total_examples, num_one_labels


def eval_margin_new(input_ents, dis_list, true_labels, margin):
    num_one_labels = true_labels.count(1.0)
    num_zero_labels = true_labels.count(0.0)
    num_total_examples = len(true_labels)
    assert num_one_labels + num_zero_labels == num_total_examples == len(true_labels)
    num_one_one, num_one_zero, num_zero_zero, num_zero_one = 0, 0, 0, 0

    predicated_match_ents = []
    sim_11_10_01_00 = [[],[],[],[]]
    for i in range(num_total_examples):
        y = true_labels[i]
        dis = dis_list[i]
        if y == 1.0 and dis < margin:
            num_one_one += 1
            sim_11_10_01_00[0].append(dis)
        elif y == 1.0 and dis >= margin:
            num_one_zero += 1
            sim_11_10_01_00[1].append(dis)
        elif y == 0.0 and dis < margin:
            num_zero_one += 1
            sim_11_10_01_00[2].append(dis)
        else:
            num_zero_zero += 1
            sim_11_10_01_00[3].append(dis)
            predicated_match_ents.append(input_ents[i])

    get_results(num_one_one, num_one_zero, num_zero_zero, num_zero_one, num_total_examples, num_one_labels)

    print("matchable and predicted matchable: {}; predicated matchable: {}".format(len(predicated_match_ents),
                                                                                   num_zero_zero + num_one_zero))

    return predicated_match_ents, num_zero_zero + num_one_zero, num_one_one, num_one_zero, num_zero_zero, num_zero_one, num_total_examples, num_one_labels


def eval_dangling_binary_cls(input_ents, y_predict, true_labels, threshold):
    num_one_labels = true_labels.count(1.0)  
    num_zero_labels = true_labels.count(0.0)
    num_total_examples = len(true_labels)
    assert num_one_labels + num_zero_labels == num_total_examples == len(true_labels)

    if threshold == -1:
        threshold = np.mean(y_predict)
        print(f'use the mean prob as the threshold, the mean probs: {np.mean(y_predict)}')

    original_y_predict = y_predict
    y_predict = (y_predict > threshold).astype(np.int32)
    one_one = np.array([i for i in range(len(true_labels)) if true_labels[i] == 1 if y_predict[i] == 1]).astype(np.int32)
    one_zero = np.array([i for i in range(len(true_labels)) if true_labels[i] == 1 if y_predict[i] == 0]).astype(np.int32)
    zero_zero = np.array([i for i in range(len(true_labels)) if true_labels[i] == 0 if y_predict[i] == 0]).astype(np.int32)
    zero_one = np.array([i for i in range(len(true_labels)) if true_labels[i] == 0 if y_predict[i] == 1]).astype(np.int32)
    num_one_one = len(one_one)
    num_one_zero = len(one_zero)
    num_zero_one = len(zero_one)
    num_zero_zero = len(zero_zero)

    predicted_match_ents = np.array(input_ents, dtype=np.int32)[zero_zero]
    predicted_match_ents = list(predicted_match_ents)

    print(f'num_one_one: {num_one_one}, num_one_zero: {num_one_zero}, num_zero_one: {num_zero_one}, num_zero_zero: {num_zero_zero}')
    print(f'one_zero, mean(y_predict): {np.mean(original_y_predict[one_zero])}, median(y_predict): {np.median(original_y_predict[one_zero])}')
    print(f'zero_one, mean(y_predict): {np.mean(original_y_predict[zero_one])}, median(y_predict): {np.median(original_y_predict[zero_one])}')
    print(f'zero_zero, mean(y_predict): {np.mean(original_y_predict[zero_zero])}, median(y_predict): {np.median(original_y_predict[zero_zero])}')
    get_results(num_one_one, num_one_zero, num_zero_zero, num_zero_one, num_total_examples, num_one_labels)

    print("matchable and predicted matchable: {}; predicted matchable: {}".format(num_zero_zero,
                                                                            num_zero_zero + num_one_zero))

    return predicted_match_ents, num_zero_zero + num_one_zero, \
           num_one_one, num_one_zero, num_zero_zero, num_zero_one, num_total_examples, num_one_labels
