#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/2/25 下午3:30
# @Author  : jlinka
# @File    : __init_.py

__all__ = ["wordpre"]
import os
import config
import tool.bigfile as bigfile
import jieba
if __name__ == '__main__':
    tagdata_filepaths = os.listdir(config.PRE_DIC)
    datas = []
    para = []
    if '.DS_Store' in tagdata_filepaths:
        tagdata_filepaths.remove('.DS_Store')
    for tagdata_filepath in tagdata_filepaths:
        if os.path.exists(config.PRE_DIC + '/' + tagdata_filepath):
            para = []
            num = 0
            for line in bigfile.get_lines(config.PRE_DIC + '/' + tagdata_filepath):
                if num == 0:
                    num = 1
                    continue
                if line != ',,,;;;;;;;;;;,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,\n' and line != ',,,;;;;;;;;;;,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,\n':
                    spline = line.replace('\n', '').split(',,,,')[0]
                    # para.append(line.replace('\n', ''))
                    para.append(spline)
                else:
                    pass
    print(para)
    for i in para:
        datas.extend(jieba.lcut(i))
    datas = list(set(datas))
    print(datas)
    with open('news_word.txt', 'w') as wf:
        for j in datas:
            wf.write(j+' ')