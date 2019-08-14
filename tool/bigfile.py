#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/2/25 上午10:36
# @Author  : jlinka
# @File    : bigfile.py

import logging


def get_lines(filepath: str):
    try:
        file = open(filepath, "r", errors='ignore')
        for line in file:
            yield line
        file.close()

    except Exception as e:
        logging.error(e)
        exit(1)
