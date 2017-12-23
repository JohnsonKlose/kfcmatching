# -*- coding: UTF-8 -*-

import matplotlib.pyplot as plt
import psycopg2
import pandas as pd
import numpy as np

from sklearn import linear_model

conn = psycopg2.connect(database='scarp', user='postgres', password='86732629jj', host='123.206.102.193',
                            port='5432')
cur = conn.cursor()

cur.execute("SELECT * FROM public.\"MATCHRESULT\"")
matchResult = cur.fetchall()

df = pd.DataFrame(list(matchResult))

right_x = []
right_y = []
wrong_x = []
wrong_y = []
X = []
y = []

for index, row in df.iterrows():
    X.append([float(row[4]), float(row[5])])
    if row[6] == '1':
        right_x.append(float(row[4]))
        right_y.append(float(row[5]))
        print str(index) + "is right"
        y.append(1)
    else:
        wrong_x.append(float(row[4]))
        wrong_y.append(float(row[5]))
        print str(index) + "is wrong"
        y.append(0)

regr_origin = linear_model.LogisticRegression()
regr_origin.fit(X, y)
print regr_origin.coef_[0], regr_origin.intercept_[0]
k_origin = -regr_origin.coef_[0][0] / regr_origin.coef_[0][1]
b_origin = -regr_origin.intercept_[0] / regr_origin.coef_[0][1]
x_origin = np.linspace(0.5, 1)
y_origin = k_origin * x_origin + b_origin

regr_optimize = linear_model.LogisticRegression(class_weight={1: 5})
regr_optimize.fit(X, y)
print regr_optimize.coef_[0], regr_optimize.intercept_[0]
k_optimize = -regr_optimize.coef_[0][0] / regr_optimize.coef_[0][1]
b_optimize = -regr_optimize.intercept_[0] / regr_optimize.coef_[0][1]
x_optimize = np.linspace(0.4, 0.82)
y_optimize = k_optimize * x_optimize + b_optimize

plt.plot(x_origin, y_origin, 'k--', label='no weights')
plt.plot(x_optimize, y_optimize, 'k-', label='with weights')
plt.scatter(right_x, right_y, c='r')
plt.scatter(wrong_x, wrong_y)

plt.show()