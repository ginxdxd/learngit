# !usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'quanxiandeng'


from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import SGDClassifier
# from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import csv
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


def main():
    path = '/Users/quanxiandeng/Downloads/Data/titanic'

    output_l_r = linear_regression(path)
    # output_l_r = list(output_l_r)
    # write_file(output_l_r)

    output_svm = svm(path)
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


def linear_regression(path):
    clf = LogisticRegression()

    column_keeped = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    label, training_set_cleaned, age_mean = prepare_data(path, column_keeped)

    clf.fit(training_set_cleaned, label)
    # print 'the accuracy rate is  %d' %(clf.score(training_set_cleaned, label))
    test_data = pd.read_csv(path + '/test.csv')
    test_set = clean_data(test_data, age_mean)
    test_set = np.array(test_set[column_keeped])

    linear_output = clf.predict(test_set)

    return linear_output


def prepare_data(path, column_keeped):
    training_set = pd.read_csv(path + '/train.csv')

    age_mean = training_set['Age'].median()
    training_set_cleaned = clean_data(training_set, age_mean)
    label = np.array(training_set_cleaned['Survived'])

    # column_keeped = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    training_set_cleaned = np.array(training_set_cleaned[column_keeped])


    return label, training_set_cleaned, age_mean


def svm(path):
    clf = SVC(kernel='linear')  # 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'

    column_keeped = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    label, training_set_cleaned, age_mean = prepare_data(path, column_keeped)

    clf.fit(training_set_cleaned, label)

    test_data = pd.read_csv(path + '/test.csv')
    test_set = clean_data(test_data, age_mean)
    test_set = np.array(test_set[column_keeped])

    svm_output = clf.predict(test_set)

    return svm_output


if __name__ == '__main__':
    main()