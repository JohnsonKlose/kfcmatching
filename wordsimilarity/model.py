# -*- coding: UTF-8 -*-

from cutwords import cutwords
from gensim import corpora, models, similarities


def wordmodeling(querywords, *matchwords):
    # print querywords
    # print matchwords
    corpus = []
    for each in matchwords:
        corpus.append(cutwords(each))
    # print corpus

    # 建立词袋模型
    dictionary = corpora.Dictionary(corpus)
    # print dictionary

    doc_vectors = [dictionary.doc2bow(text) for text in corpus]
    # print len(doc_vectors)
    # print doc_vectors

    # 建立TF-IDF模型
    tfidf = models.TfidfModel(doc_vectors)
    tfidf_vectors = tfidf[doc_vectors]

    # 构建查询内容
    query = cutwords(querywords)
    query_bow = dictionary.doc2bow(query)
    # print len(query_bow)
    # print query_bow

    # 计算相似度
    query_tfidf = tfidf[query_bow]
    # print query_lsi

    index = similarities.MatrixSimilarity(tfidf_vectors)
    sims = index[query_tfidf]
    return list(enumerate(sims))

if __name__ == "__main__":
    result = wordmodeling('麦当劳（中央大街餐厅）哈尔滨市道里区中央大街123号（哈尔滨市道里区中央大街83号）', '麦当劳（哈西万达店）南岗区中兴大道168号哈西万达广场步行街1069', '麦当劳（中兴大道店）')
    print result