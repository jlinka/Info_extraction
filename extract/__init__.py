#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/2/25 上午10:15
# @Author  : jlinka
# @File    : __init__.py

__all__ = ["splitsentence", "extract", "predict"]

# import os
# import config
# from tool import bigfile

# import logging
# import codecs
#
#
# def get_lines(filepath: str):
#     try:
#         file = codecs.open(filepath, "rb", 'gb2312')
#         for line in file:
#             yield line
#         file.close()
#
#     except Exception as e:
#         logging.error(e)
#         exit(1)


# if __name__ == '__main__':
# datas = list()
# if os.path.exists('0__1000.csv'):
#     para = []
#     num = 0
#     for line in get_lines('0__1000.csv'):
#         if num == 0:
#             num = 1
#             continue
#         if line != ',,,;;;;;;;;;;\n':
#             para.append(line)
#         else:
#             datas.append(para)
#             para = []
#     for a in para:
#         print(a)
#
# with codecs.open('0__1000.csv', 'rb', 'gb2312', errors='ignore') as csvfile:
#     for line in csvfile:
#         print(line)
#         line = line.replace('\t\t\t\t\t\t\r', '')
#         line = line.replace('\t', ',')
#         line = line.replace(',,,,,,', '')
#         print(line)
#         wf.write(line)

# for line in bigfile.get_lines(config.TAG_DIC + '/sr/seq/' + 'file4.csv'):
#     print(line)
# with codecs.open('0_1000.csv', 'rb', 'utf-8-sig', errors='ignore') as csvfile:
#     for line in csvfile:
#         print(line)

from tool import bigfile
import os
import jieba

def split_line(datas: list):
    words_list = list()  # 保存分词后的句子 每一项是字符串： 词 词
    investor_labels_list = list()  # 保存标签 每一项是字符串： label label
    investee_labels_list = list()
    mount_labels_list = list()
    pre_investor_labels_list = list()
    pre_investee_labels_list = list()
    pre_mount_labels_list = list()

    for news in datas:
        words = []
        investor = []
        investee = []
        mount = []
        pre_investor = []
        pre_investee = []
        pre_mount = []
        for sentence in news:
            if sentence[6:] != '':
                investor.append(sentence[:1])
                investee.append(sentence[2:3])
                mount.append(sentence[4:5])
                pre_investor.append(sentence[6:7])
                pre_investee.append(sentence[8:9])
                pre_mount.append(sentence[10:11])
                words.append(sentence[12:])
        investor_labels_list.append(investor)
        investee_labels_list.append(investee)
        mount_labels_list.append(mount)
        pre_investor_labels_list.append(pre_investor)
        pre_investee_labels_list.append(pre_investee)
        pre_mount_labels_list.append(pre_mount)
        words_list.append(words)

    return investor_labels_list, investee_labels_list, mount_labels_list, pre_investor_labels_list, \
           pre_investee_labels_list, pre_mount_labels_list, words_list


if __name__ == '__main__':
    datas = list()

    wf1 = open('train1.csv', 'a', encoding='UTF-8-sig')
    wf2 = open('train2.csv', 'a', encoding='UTF-8-sig')
    if os.path.exists('preditor.csv'):
        para = []
        num = 0
        flag = 0
        for line in bigfile.get_lines('preditor.csv'):
            if num == 0:
                num = 1
                continue
            if line != ',,,,,,;;;;;;;;;;\n':
                para.append(line.replace('\n', ''))
            else:
                # flag += 1
                datas.append(para)
                para = []
            if flag == 2:
                break

    investor_labels_list, investee_labels_list, mount_labels_list, pre_investor_labels_list, pre_investee_labels_list, pre_mount_labels_list, words_list = split_line(
        datas)
    print(datas)
    wf1.write('investor,investee,mount\n')
    wf2.write('investor,investee,mount\n')
    eee = 0
    for i in range(len(investee_labels_list)):
        eee += 1
        if eee < 500:
            for j in range(len(investee_labels_list[i])):
                if investor_labels_list[i][j] == '1' or investee_labels_list[i][j] == '1' or mount_labels_list[i][j] == '1':
                    wf1.write(
                        investor_labels_list[i][j] + ',' + investee_labels_list[i][j] + ',' + mount_labels_list[i][
                            j] + ',')
                    for k in jieba.lcut(words_list[i][j]):
                        wf1.write(k+',')
                    wf1.write('\n')
                    wf1.write(',,,')
                    for l in jieba.lcut(words_list[i][j]):
                        wf1.write('0,')
                    wf1.write('\n')
                    # wf2.write(
                    #     pre_investor_labels_list[i][j] + ',' + pre_investee_labels_list[i][j] + ',' +
                    #     pre_mount_labels_list[i][
                    #         j] + ',' +
                    #     words_list[i][j] + '\n')
            wf1.write(',,,;;;;;;;;;;\n')
        else:
            for j in range(len(investee_labels_list[i])):
                if investor_labels_list[i][j] == '1' or investee_labels_list[i][j] == '1' or mount_labels_list[i][j] == '1':
                    wf2.write(
                        investor_labels_list[i][j] + ',' + investee_labels_list[i][j] + ',' + mount_labels_list[i][
                            j] + ',')
                    for k in jieba.lcut(words_list[i][j]):
                        wf2.write(k+',')
                    wf2.write('\n')
                    wf2.write(',,,')
                    for l in jieba.lcut(words_list[i][j]):
                        wf2.write('0,')
                    wf2.write('\n')
                    # wf2.write(
                    #     pre_investor_labels_list[i][j] + ',' + pre_investee_labels_list[i][j] + ',' +
                    #     pre_mount_labels_list[i][
                    #         j] + ',' +
                    #     words_list[i][j] + '\n')
            wf2.write(',,,;;;;;;;;;;\n')