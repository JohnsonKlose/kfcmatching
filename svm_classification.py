# -*- coding: UTF-8 -*-

import matplotlib.pyplot as plt
import psycopg2
import pandas as pd
import numpy as np

from sklearn import svm

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

clf = svm.SVC(kernel='linear', class_weight={1: 5})
clf.fit(X, y)

clf_origin = svm.SVC(kernel='linear')
clf_origin.fit(X, y)
print "------fit end------"

xx = np.linspace(0.35, 0.85)
w = clf.coef_[0]
a = -w[0] / w[1]
yy = a * xx - clf.intercept_[0] / w[1]
print a, clf.intercept_[0] / w[1]

xx_origin = np.linspace(0.25, 0.75)
w_orgin = clf_origin.coef_[0]
a_origin = -w_orgin[0] / w_orgin[1]
yy_origin = a * xx_origin - clf_origin.intercept_[0] / w_orgin[1]
print a_origin, clf_origin.intercept_[0] / w_orgin[1]

plt.plot(xx, yy, 'k-', label='with weights')
plt.plot(xx_origin, yy_origin, 'k--', label='no weights')
plt.scatter(right_x, right_y, c='r')
plt.scatter(wrong_x, wrong_y)

plt.show()