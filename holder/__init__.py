#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/5 上午10:09
# @Author  : jlinka
# @File    : __init__.py

__all__ = ["jiebah", "wordvech"]

import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    M = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    M = np.array(M)
    v = M.sum(axis=0)
    print(v)
    v = v**2
    print(v)