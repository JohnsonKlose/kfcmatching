# -*- coding: UTF-8 -*-

import psycopg2

from coordsimilarity.coordtransform import *
from coordsimilarity.model import coordmodeling
from wordsimilarity.model import wordmodeling

conn = psycopg2.connect(database='scarp', user='postgres', password='86732629jj', host='123.206.102.193',
                            port='5432')
cur = conn.cursor()

cur.execute("SELECT * FROM public.\"BAIDUKFC\"")
bmaprows = cur.fetchall()

cur.execute("SELECT * FROM public.\"AMAPKFC\"")
amaprows = cur.fetchall()

if __name__ == "__main__":
    for bmaprow in bmaprows:

        bmap_id = bmaprow[0]
        bmap_lng = bmaprow[16]
        bmap_lat = bmaprow[17]
        bmap_name = bmaprow[9]
        bmap_address = bmaprow[2]
        # bmap_object = data_model(bmap_id, bmap_lng, bmap_lat, bmap_name, bmap_address)
        coord = bd09togcj02(float(bmap_lng), float(bmap_lat))

        coordsimilarity_list = []
        matchingwords_list = []
        amapid_list = []
        amapname_list = []

        # 遍历高德地图结果
        for amaprow in amaprows:

            amap_id = amaprow[0]
            amap_lng = amaprow[8]
            amap_lat = amaprow[7]
            amap_name = amaprow[6]
            amap_address = amaprow[5]
            # amap_object = data_model(amap_id, amap_lng, amap_lat, amap_name, amap_address)

            # 计算空间相似性, 存储在coordsimilarity_list中
            coordsimilarity_rating = coordmodeling(coord, [amap_lng, amap_lat])
            coordsimilarity_list.append(coordsimilarity_rating)

            # 收集待匹配的语义文字
            matchingwords_list.append(amap_name + " " + amap_address)
            amapid_list.append(amap_id)
            amapname_list.append(amap_name)

        # 计算语义相似性
        wordsimilarity_rating = wordmodeling(bmap_name + " " +bmap_address, *matchingwords_list)

        for k in range(0, len(amaprows)):
            value = 1
            if wordsimilarity_rating[k][1] < 0.5:
                value = 0
            elif coordsimilarity_list[k] < 0.5:
                value = 0

            cur.execute(
                """insert into "MATCHRESULT" (firstid, firstname, secondid, secondname, coordsimilarity, wordsimilarity, value)
                VALUES (%(firstid)s, %(firstname)s, %(secondid)s, %(secondname)s, %(coordsimilarity)s, %(wordsimilarity)s, %(value)s)""",
                {
                    'firstid': str(bmap_id),
                    'firstname': str(bmap_name),
                    'secondid': str(amapid_list[k]),
                    'secondname': str(amapname_list[k]),
                    'coordsimilarity': str(coordsimilarity_list[k]),
                    'wordsimilarity': str(wordsimilarity_rating[k][1]),
                    'value': str(value)
                })
            print "add data:" + str(bmap_name) + " match " + str(amapname_list[k]) + " value is " + str(value)

    conn.commit()
    print "------END------"