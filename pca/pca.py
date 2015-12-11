# -*- coding: utf-8 -*-
#!usr/bin/env python

"""
    @author : ginxd
    @contact : ginxdxd@gmail.com
    @file : pca.py
    @time : 20/11/15
"""
#  np.dot 矩阵乘法   mat_a * mat_b 对位相乘



import numpy as np
"""
    suppose that training_set'type is np.array
"""

def pca(training_set, percent):
    mean_mat = training_set.mean(axis=0)
    diff_mat = training_set - mean_mat
    # cov_mat = np.dot(diff_mat.T, diff_mat)
    cov_mat = np.cov(diff_mat, rowvar=0)  # rowvar 为1， 每行代表一个变量； 为0， 每列代表一个变量
    U, Sigma, VT = np.linalg.svd(cov_mat)
    k = get_k_from_percent(Sigma, percent)
    return np.dot(training_set, U[:k+1])


def get_k_from_percent(mat, percent):
    for i in range(len(mat)):
        if sum(mat[:i+1]) >= sum(mat) * percent:
            return i
        else:
            i += 1

# def feature_scaling(training_set, mat):
