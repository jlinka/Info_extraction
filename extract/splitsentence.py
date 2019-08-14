#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/2/25 上午10:18
# @Author  : jlinka
# @File    : splitsentence.py


import re


# 将单条新闻分句
def news2sentences(srcnews: str):
    # 停用掉某些符号 《》<>（）()「」{}|
    pattern = r'[(](.*?)[)]|[（](.*?)[）]|[《》<>「」{}【】\[\]"“”]'  # 将'《》<>「」{}'"“” 去掉 （）及其里面内容去掉去掉
    pat = re.compile(pattern)
    srcnews = re.sub(pat, '', srcnews)  # 将'《》<>「」{}'去掉

    # 分句 ，。？！；
    pattern1 = r'[||:：。?？!！;；\n\t\r]'
    pat1 = re.compile(pattern1)
    sentences = re.split(pat1, srcnews.strip())  # 以'，。？！；'为句子分隔符分割句子
    return [sentence for sentence in sentences if sentence]
