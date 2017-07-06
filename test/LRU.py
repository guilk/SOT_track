import numpy as np
import random
import mxnet as mx
random.seed(100)

class LRUCache(object):
    """
    A simple class that implements LRU algorithm
    if 
    """
    def __init__(self, qfeat, gpu_id, cache_size = 5, threshold = 0.9):
        self.cache_size = cache_size
        self.template = mx.nd.zeros((self.cache_size, qfeat.shape[0]),mx.gpu(gpu_id)) # num_templates, num_dim
        self.timestamp = np.zeros(self.cache_size)
        self.template[0] = qfeat.T[0].copy()
        self.tmpl_counter = 1
        self.update_thres = threshold

        self.counter = np.zeros(self.cache_size)
        self.scores = np.zeros(self.cache_size)


    def update(self, cand_feat, cache_score):
        old_idx = -1
        update = -1
        if self.tmpl_counter < self.cache_size:
            self.template[self.tmpl_counter] = cand_feat.copy()
            self.tmpl_counter += 1
            update = 2
        else:
            # prob = random.random()
            # if prob > self.update_thres:
            #     min_ind = np.argmin(self.scores / self.counter)
            #     self.template[min_ind] = cand_feat.copy()
            #     self.counter[min_ind] = 1
            #     # max_ind = np.argmax(self.timestamp)
            #     # self.timestamp[max_ind] = 0
            #     # self.template[max_ind] = cand_feat.copy()
            # else:
            #     self.scores += cache_score
            #     self.counter += 1
            prob = random.random()
            if prob > self.update_thres:
                max_ind = np.argmax(self.timestamp)
                self.timestamp[max_ind] = 0
                self.template[max_ind] = cand_feat.copy()
                old_idx = max_ind
                update = 1
            else:
                min_ind = np.argmin(cache_score)
                self.timestamp[min_ind] += 1
        return old_idx,update

    def cal_sim(self, cand_feats, nr_samples, query_scores, use_template = False):

        '''
        :param cand_feat: numpy array: [num_boxes, dim_features] 
        :return: similarity scores
        '''

        templ_sims = mx.nd.dot(cand_feats, self.template.T).asnumpy().squeeze()[:nr_samples,:self.tmpl_counter] # [num_boxes, cache_size]
        templ_scores = np.median(templ_sims, axis=1)

        if use_template == True:
            decision_scores = query_scores + templ_scores
            max_idx = np.argmax(decision_scores)
        else:
            decision_scores = query_scores
            max_idx = np.argmax(decision_scores)
        old_idx,update = self.update(cand_feats[max_idx], templ_sims[max_idx,:])
        return max_idx, decision_scores.squeeze(), old_idx, update