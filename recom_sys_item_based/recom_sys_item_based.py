# -*-coding:utf-8 -*-
# !usr/bin/env python

"""
    @author : ginxd
    @contact : ginxdxd@gmail.com
    @file : recom_sys_item_based.py
    @time : 9/12/15
"""


import numpy as np
import pandas as pd

N = 2   # 推荐商品的个数
PATH = ''

def main():
    data_set = load_data(PATH)
    training_set = clean_data(data_set)
    training_set = np.array(training_set)
    result_list = remcom(training_set, 0)
    print result_list    # 返回的是被推荐商品的索引，以及预计的评分


def predict_rate(training_set, user, item):
    n_item = training_set.shape[0]
    sim_total = 0.0
    rat_sim_total = 0.0

    for i in range(n_item):
        user_rate = training_set[i, user]
        if user_rate == 0:
            continue
        user_rate_both = np.nonzero(np.logical_and(training_set[item, :] > 0, training_set[i, :] > 0))[0]   # 因为是矢量跟矢量的逻辑运算，nonzero返回一维
        if len(user_rate_both) == 0:
            sim_val = 0
        else:
            sim_val = get_sim(training_set[item, user_rate_both], training_set[i, user_rate_both])

        sim_total += sim_val
        rat_sim_total += sim_val * user_rate

    if sim_total == 0:
        return 0
    else:
        return rat_sim_total / sim_total


def get_sim(item_a, item_b):   # 欧氏距离
    return 1.0 / (1.0 + np.linalg.norm(item_a - item_b))


def remcom(training_set, user):
    unrate_item = np.nonzero(training_set[:, user] == 0)[0]   # 找出该用户还没评分的商品索引
    if len(unrate_item) == 0:
        print 'we have nothing to recommend for you, sorry!'  # 找不到该用户还没评分的项，无法进行推荐
    item_score = []
    for item in unrate_item:
        score = predict_rate(training_set, user, item)
        item_score.append((item, score))
    return sorted(item_score, key=lambda j: j[1], reverse=True)[:N]  # 倒序排列取前n个商品


def clean_data(mat):
    mat = mat.fillna(0)
    clean_mat = mat.drop(mat.columns[0], axis=1)
    return clean_mat


def load_data(PATH):
    file_in = pd.read_csv(PATH, encoding="gb2312")
    return file_in


if __name__ == '__main__':
    main()

