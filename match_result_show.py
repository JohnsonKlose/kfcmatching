# -*- coding: UTF-8 -*-

import matplotlib.pyplot as plt
import psycopg2
import pandas as pd

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
for index, row in df.iterrows():
    if row[6] == '1':
        right_x.append(float(row[4]))
        right_y.append(float(row[5]))
        print str(index) + "is right"
    else:
        wrong_x.append(float(row[4]))
        wrong_y.append(float(row[5]))
        print str(index) + "is wrong"

plt.scatter(right_x, right_y, c='r')
plt.scatter(wrong_x, wrong_y)
plt.xlabel('coord_similarity')
plt.ylabel('word_similarity')

plt.show()