# -*- coding: utf-8 -*-
# !usr/bin/env python

"""
    @author : ginxd
    @contact : ginxdxd@gmail.com
    @file : get_titanic_data.py
    @time : 9/12/15 下午10:50
"""


import pandas as pd
import numpy as np

PATH = ''
COLUMN_KEEPED = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

def load_data():
    training_data_set = pd.read_csv(PATH + '/train.csv')
    test_data_set = pd.read_csv(PATH + '/test.csv')

    age_mean = training_data_set['Age'].median()
    training_data_set_cleaned = clean_data(training_data_set, age_mean)
    labels = np.array(training_data_set_cleaned['Survived'])
    training_set = np.array(training_data_set_cleaned[COLUMN_KEEPED])

    test_data_set_cleaned = clean_data(test_data_set, age_mean)
    test_set = np.array(test_data_set_cleaned[COLUMN_KEEPED])


    return training_set, labels, test_set

def clean_data(frame, age_mean):
    frame['Embarked'] = frame['Embarked'].fillna('S')

    frame.loc[frame['Sex'] == 'male', 'Sex'] = 0   # 定位到某一列，然后修改此列
    frame.loc[frame['Sex'] == 'female', 'Sex'] = 1

    frame.loc[frame['Embarked'] == 'S', 'Embarked'] = 0
    frame.loc[frame['Embarked'] == 'C', 'Embarked'] = 1
    frame.loc[frame['Embarked'] == 'Q', 'Embarked'] = 2

    frame['Fare'] = frame['Fare'].fillna(frame['Fare'].median())
    frame['Age'] = frame['Age'].fillna(age_mean)
    return frame


