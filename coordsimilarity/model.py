# -*- coding: utf-8 -*-

import math

threshold = 0.01381770


def coordmodeling(coord1, coord2):

    distance = math.sqrt((float(coord1[0])-float(coord2[0]))**2 + (float(coord1[1])-float(coord2[1]))**2)
    if distance >= threshold:
        return 0
    else:
        return 1 - distance/threshold