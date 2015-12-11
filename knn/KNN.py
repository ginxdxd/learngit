# -*- coding: utf-8 -*-
# !usr/bin/env python

"""
    @author : ginxd
    @contact : ginxdxd@gmail.com
    @file : KNN.py
    @time : 18/11/15
"""
import numpy as np
import collections

"""
    knn算法：
        一种分类的算法，找出与输入数据距离最相近的k个样本，距离公式可选欧氏，皮特森距离等，
    k个样本中占比最多的类别就被预测为输入数据的类别。
"""

'''
 k的取值一般不超过20， 工程上k的取值为训练集的平方根
 array.sum(axis=1)按行累加，axis=0为按列累加
 argsort()是numpy中的方法，得到矩阵中每个元素的排序序号
'''

def knn(test_set, training_input, training_output, k):  # 欧式距离公式
    size = training_input.shape[0]
    diff_mat = np.tile(test_set, (size, 1)) - training_input
    sq_diff_mat = diff_mat ** 2
    sum_sq_diff_mat = sq_diff_mat.sum(axis=1)
    distance_mat = sum_sq_diff_mat ** 0.5   # distance_mat is a (size * 1) matrix
    sorted_distance_mat = np.argsort(distance_mat)  # sorted_distance_mat is a index_sorted matrix
    the_first_k_similar_target = collections.defaultdict(int)

    for i in range(k):
        training_out_label = training_output[sorted_distance_mat[i]]
        the_first_k_similar_target[training_out_label] += 1
    test_label_sorted = sorted(the_first_k_similar_target.items(), key=lambda d: d[1], reverse=True)
    return test_label_sorted[0][0]





