# -*- coding: UTF-8 -*-

import psycopg2

'''
    数据清洗, 处理包含洗手间宅急送内容的等数据
'''


def iskfc(text):
    if "洗手间" in text:
        return False
    if "宅急送" in text:
        return False
    if "甜品站" in text:
        return False
    if "KFC" in text:
        return False
    if "肯德基" == text:
        return False
    if "中国人的肯德基" == text:
        return False
    else:
        return True

if __name__ == "__main__":
    print iskfc("肯德基(高淳淳溪餐厅)-洗手间")
    conn = psycopg2.connect(database='scarp', user='postgres', password='86732629jj', host='123.206.102.193',
                            port='5432')
    cur = conn.cursor()

    cur.execute("SELECT * FROM public.\"BAIDUKFC\"")
    bmaprows = cur.fetchall()

    cur.execute("SELECT * FROM public.\"AMAPKFC\"")
    amaprows = cur.fetchall()

    for bmaprow in bmaprows:
        bmap_name = bmaprow[9]
        if not iskfc(bmap_name):
            guid = bmaprow[0]
            cur.execute("DELETE FROM public.\"BAIDUKFC\" WHERE id = '" + str(guid) + "'")
            print "delete" + bmap_name

    for amaprow in amaprows:
        amap_name = amaprow[6]
        if not iskfc(amap_name):
            guid = amaprow[0]
            cur.execute("DELETE FROM public.\"BAIDUKFC\" WHERE id = '" + str(guid) + "'")
            print "delete" + amap_name

    conn.commit()