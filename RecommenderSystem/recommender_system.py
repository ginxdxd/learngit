# -*- coding: utf-8 -*-
# !usr/bin/env python

__author__ = 'quanxiandeng'
'''
   item_matrix * (user_matrix.T)
   n_feature 表示对商品的 n_features 个方面进行评分
'''
import numpy as np

N_FEATURES = 100
LEARNING_RATE = 0.1

def random_matrix(mat_size, low, high):
    return (high-low) * np.random.random(mat_size) + low

class recommend_sys:
    def __init__(self, n_items, n_users):
        self.item_weight_mat = random_matrix((n_items, N_FEATURES), -0.1, 0.1)  # init
        self.user_weight_mat = random_matrix((n_users, N_FEATURES), -0.1, 0.1)  # init
        self.r_mat = np.zeros((n_items, n_users))
        self.diff_mat = np.zeros_like(self.r_mat, dtype=float)

    def train(self, training_set, steps):
        total_error = 0.0
        for s in range(steps):
            user_rate_mat = np.dot(self.user_weight_mat, self.item_weight_mat.transpose())
            total_error = self.calculate_error(user_rate_mat, training_set)
            self.update_weights()
        return total_error

    def calculate_error(self, rate_mat, training_set):
        for i in self.r_mat.shape[0]:
            for j in self.r_mat.shape[1]:
                if self.r_mat[i][j] == 1:
                    self.diff_mat[i][j] = (rate_mat[i][j] - training_set[i][j])
                else:
                    self.diff_mat[i][j] = 0
        total_error = (self.diff_mat ** 2).sum(axis=1).sum(axis=0)
        return total_error

    def update_weights(self):
        for i in self.r_mat.shape[0]:
            for u in self.r_mat.shape[1]:
                if self.r_mat[i][u] == 1:
                    for n in N_FEATURES:
                        self.item_weight_mat[i][n] -= LEARNING_RATE * self.diff_mat[i][u] * self.user_weight_mat[u][n]
                        self.user_weight_mat[u][n] -= LEARNING_RATE * self.diff_mat[i][u] * self.item_weight_mat[i][n]

def recommender_sys(training_set):
    n_items, n_users = training_set.shape
    reco_sys = recommend_sys(n_items, n_users)

    for i in n_items:
        for u in n_users:
            if training_set[i][u] == 0:
                reco_sys.r_mat[i][u] == 0
            else:
                reco_sys.r_mat[i][u] == 1

    reco_sys.train(training_set, 100)






