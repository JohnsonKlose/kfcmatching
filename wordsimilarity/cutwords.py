# -*- coding: UTF-8 -*-

import jieba.posseg as pseg
import codecs

stopword_file = "/Users/yifengjiao/PycharmProjects/scrapDemo/dbcomments/stopwords.txt"
stopwords = codecs.open(stopword_file, 'r', encoding='utf8').readlines()
stopwords = [w.strip() for w in stopwords]
stop_flag = ['x', 'c', 'u', 'd', 'p', 't', 'uj', 'm', 'f', 'r']

def cutwords(text):
    result = []
    words = pseg.cut(text)
    for word, flag in words:
        if flag not in stop_flag and word not in stopwords:
            result.append(word)
    return result

if __name__ == "__main__":
    print cutwords('快餐 美食 西式快餐 餐馆')