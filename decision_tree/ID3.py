# -*- coding: utf-8 -*-
# !usr/bin/env python

"""
    @author : ginxd
    @contact : ginxdxd@gmail.com
    @file : ID3.py
    @time : 18/12/15
"""

import math
import numpy as np
import pandas as pd



def main():
    test = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    data_set = np.array(test)
    feature_name_set = ['no surfacing', 'flippers']
    # data_set, feature_name_set = []
    print ID3(data_set, feature_name_set)

def ID3(data_set, feature_name_set):
    label = set(data_set[:, -1])  # 查看最后一列的取值还有几种
    if len(label) == 1:  # 叶子节点只有一种类别，这是情况1
        return list(label)[0]  # 因为set只能迭代，不能索引，所以要转化为list
    if data_set.shape[1] == 1:  # 每个特征都处理过了，目前数据集剩下label唯一一列，只能投票决定类别，这是情况2
        return majority_vote(data_set)
    first_feature = the_best_feature_to_split(data_set)
    best_feature = feature_name_set[first_feature]
    tree = {best_feature: {}}
    del(feature_name_set[first_feature])  # 当feature_name_set为list时能用此方法， np.array另外
    first_feature_value = set(data_set[:, first_feature])

    for value in first_feature_value:
        label_name_set_cop = feature_name_set[:]
        tree[best_feature][value] = ID3(the_remained_data_set(data_set, first_feature, value), label_name_set_cop)

    return tree


def the_remained_data_set(data_set, feature_index, value):  # 剔除最优特征后的数据集
    index = data_set[:, feature_index] == value  # 构造boolean索引
    new_data_set = data_set[index]
    remained_data_set = np.concatenate([new_data_set[:, :feature_index], new_data_set[:, feature_index + 1:]], axis=1)
    return remained_data_set


def the_best_feature_to_split(data_set):
    n_samples = data_set.shape[0]  # 样本数
    n_features = data_set.shape[1] - 1  # 特征数
    shannon_value = calculate_shannon(data_set)
    best_info_gain = 0.0
    best_feature_index = -1

    for f in range(n_features):
        value_set = set(data_set[:, f])
        new_shannon_value = 0.0
        for value in value_set:
            remained_data_set = the_remained_data_set(data_set, f, value)
            prob = remained_data_set.shape[0] / float(n_samples)
            new_shannon_value += prob * calculate_shannon(remained_data_set)  # 更新后的数据集的香农熵用加权平均算
        info_gain = shannon_value - new_shannon_value
        if info_gain > best_info_gain:  # 最后保留 info_gain 最大值，意义为最大信息增益
            best_info_gain = info_gain
            best_feature_index = f
    return best_feature_index

def calculate_shannon(data_set):   # data_set 最后一列是label列
    n_samples = data_set.shape[0]
    label_count = {}
    label = set(data_set[:, -1])
    for value in label:
        label_count[value] = data_set[data_set[:, -1] == value].shape[0]

    shannon_value = 0.0
    for key in label_count.keys():
        prob = float(label_count[key]) / n_samples
        shannon_value -= prob * math.log(prob, 2)
    return shannon_value

def majority_vote(label_list):
    label_count = {}
    label = set(label_list)
    for key in label:
        label_count[key] = label_list[label_list == key].shape[0]
    sort_label = sorted(label_count.iteritems(), key=lambda key_value: key_value[1], reverse=True)  # 按value排列，倒序
    return sort_label[0][0]

if __name__ == '__main__':
    main()



