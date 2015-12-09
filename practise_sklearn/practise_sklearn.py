# -*- coding: utf-8 -*-
# !usr/bin/env python

"""
    @author : ginxd
    @contact : ginxdxd@gmail.com
    @file : practise.py
    @time : 3/12/15 上午10:13
"""


from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import SGDClassifier
# from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import csv
import get_titanic_data
import numpy as np
import pandas as pd


'''
PassengerId -- A numerical id assigned to each passenger.
Survived -- Whether the passenger survived (1), or didn't (0). We'll be making predictions for this column.
Pclass -- The class the passenger was in -- first class (1), second class (2), or third class (3).
Name -- the name of the passenger.
Sex -- The gender of the passenger -- male or female.
Age -- The age of the passenger. Fractional.
SibSp -- The number of siblings and spouses the passenger had on board.
Parch -- The number of parents and children the passenger had on board.
Ticket -- The ticket number of the passenger.
Fare -- How much the passenger paid for the ticker.
Cabin -- Which cabin the passenger was in.
Embarked -- Where the passenger boarded the Titanic.
'''

def main():
    training_set, label, test_set = get_titanic_data.load_data()

    # output_l_r = linear_regression(training_set, label, test_set)
    # output_l_r = list(output_l_r)
    # write_file(output_l_r)

    output_svm = svm(training_set, label, test_set)
    output_svm = list(output_svm)
    write_file(output_svm)


def write_file(output_label):
    f_out = open("/Users/quanxiandeng/Downloads/Data/titanic/output_svm.csv", "wb")
    writer = csv.writer(f_out)
    # writer.writerow(output_label.item)
    id = 892
    writer.writerow(('PassengerId', 'Survived'))
    for item in output_label:
        writer.writerow((id, item))
        id += 1
    f_out.close()


def linear_regression(training_set, label, test_set):
    clf = LogisticRegression()
    clf.fit(training_set, label)
    linear_output = clf.predict(test_set)

    return linear_output


def svm(training_set, label, test_set):
    clf = SVC(kernel='linear')  # 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
    clf.fit(training_set, label)
    svm_output = clf.predict(test_set)

    return svm_output


if __name__ == '__main__':
    main()