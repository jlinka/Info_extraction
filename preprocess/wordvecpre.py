#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/2/26 下午7:12
# @Author  : jlinka
# @File    : wordvecpre.py

import config
import os
import tool.bigfile as bigfile
import jieba
import time


def split_sentence(sentence_list: list):
    word_list = []
    for sentence in sentence_list:
        # try:
        #     result = config.client.lexer(sentence)
        #     time.sleep(2)
        # except:
        #     print("error")
        #
        # if 'items' in result:
        #     for word in result['items']:
        #         print(word['item'])
        #         word_list.append(word['item'])
        word_list.extend(jieba.lcut(sentence, HMM=True))
    word_list = list(set(word_list))
    return word_list



if __name__ == '__main__':
    datas = []
    file_name_list = os.listdir(config.TAG_DIC + '/sr/seq/')
    if '.DS_store' in file_name_list:
        file_name_list.remove('.DS_store')
    for file_name in file_name_list:
        if os.path.exists(config.TAG_DIC + '/sr/seq/' + file_name):
            num = 0
            for line in bigfile.get_lines(config.TAG_DIC + '/sr/seq/' + file_name):
                if num == 0:
                    num = 1
                    continue
                if line != ',,,;;;;;;;;;;\n':
                    datas.append(line.replace('\n', ''))

        else:
            raise FileNotFoundError('{} 标注数据文件不存在'.format(file_name))
    words = split_sentence(datas)
    with open('news_word.txt', 'a', encoding='UTF-8-sig') as wf1:
        for word in words:
            wf1.write(word + ' ')
    print("write over")
