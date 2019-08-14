#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/2/25 上午10:38
# @Author  : jlinka
# @File    : shuffle.py

import random


def shuffle_both(x: list, y: list):
    both = list(zip(x, y))
    random.shuffle(both)
    shuffle_x, shuffle_y = zip(*both)

    return list(shuffle_x), list(shuffle_y)