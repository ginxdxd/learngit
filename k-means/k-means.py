# -*- coding: utf-8 -*-
# !usr/bin/env python

"""
    @author : ginxd
    @contact : ginxdxd@gmail.com
    @file : k-means.py
    @time : 21/12/15
"""

import numpy as np
import random

K_TYPES = 7

def main():
    dataset = []
    k_center_mat, info_mat = k_means(dataset, K_TYPES)
    print k_center_mat
    print '==========='
    print info_mat

def k_means(dataset, k):
    n_samples = dataset.shape[0]
    k_center_mat = create_k_center(dataset, k)  # 构造中心点
    info_mat = np.zeros((n_samples, 2))  # 存储最近的中心点索引，距离
    change = True
    while change:
        change = False
        for i in range(n_samples):  # 对每个样本，找出最相近的中心点
            min_index, min_dis = min_index_and_distance(dataset[i], k_center_mat, k)
            if info_mat[i, :][1] != min_dis:
                change = True
            info_mat[i, :] = min_index, min_dis
        for k in range(k_center_mat.shape[0]):  # 更新相同蔟数据集的中心点
            the_k_type = dataset[info_mat[:, 0] == k]
            k_center_mat[k, :] = np.mean(the_k_type, axis=1)
    return k_center_mat, info_mat

def create_k_center(dataset, k):  # 打乱后取前k个作为初始中心点
    n_samples = dataset.shape[0]
    index = range(n_samples)
    random.shuffle(index)
    return dataset[index][:k]

def min_index_and_distance(sample, k_center_mat, k):
    diff_mat = np.tile(sample, (k, 1)) - k_center_mat
    sq_diff_mat = diff_mat ** 2
    sum_sq_diff_mat = sq_diff_mat.sum(axis=1)
    distance_mat = sum_sq_diff_mat ** 0.5
    return distance_mat.argmin(), distance_mat.min()

if __name__ == '__main__':
    main()
