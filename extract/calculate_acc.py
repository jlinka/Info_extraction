#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/19 上午9:45
# @Author  : jlinka
# @File    : calculate_acc.py

import os
import config
from tool import bigfile


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
                investor.append(int(sentence[:1]))
                investee.append(int(sentence[2:3]))
                mount.append(int(sentence[4:5]))
                pre_investor.append(int(sentence[6:7]))
                pre_investee.append(int(sentence[8:9]))
                pre_mount.append(int(sentence[10:11]))
                words.append(sentence[12:])
        investor_labels_list.append(investor)
        investee_labels_list.append(investee)
        mount_labels_list.append(mount)
        pre_investor_labels_list.append(pre_investor)
        pre_investee_labels_list.append(pre_investee)
        pre_mount_labels_list.append(pre_mount)
        words_list.append(words)

    return investor_labels_list, investee_labels_list, mount_labels_list, pre_investor_labels_list, \
           pre_investee_labels_list, pre_mount_labels_list


def calculate(investor_labels_list, investee_labels_list, mount_labels_list, pre_investor_labels_list, \
              pre_investee_labels_list, pre_mount_labels_list):
    head = 0
    test = 0
    test1 = 0
    for i in range(len(investee_labels_list)):
        for j in range(len(investee_labels_list[i])):
            print(investor_labels_list[i][j], investee_labels_list[i][j], mount_labels_list[i][j])
            if investor_labels_list[i][j] == 1 and investee_labels_list[i][j] == 1 and mount_labels_list[i][j] == 1:
                test1 += 1
                if pre_investor_labels_list[i][j] == 1 and pre_investee_labels_list[i][j] == 1 and \
                        pre_mount_labels_list[i][j] == 1:
                    head += 1
                    test += 1
                    continue
                elif pre_investor_labels_list[i][j] == 1 and pre_investee_labels_list[i][j] == 1 and \
                        pre_mount_labels_list[i][j] == 0:
                    head += 1
                    test += 1
                    continue
                elif pre_investor_labels_list[i][j] == 1 and pre_investee_labels_list[i][j] == 0 and \
                        pre_mount_labels_list[i][j] == 1:
                    head += 1
                    test += 1
                    continue
                elif pre_investor_labels_list[i][j] == 0 and pre_investee_labels_list[i][j] == 1 and \
                        pre_mount_labels_list[i][j] == 1:
                    head += 1
                    test += 1
                    continue
                #
                # elif pre_investor_labels_list[i][j] == 1 and pre_investee_labels_list[i][j] == 0 and \
                #         pre_mount_labels_list[i][j] == 0:
                #     head += 1
                #     continue
                # elif pre_investor_labels_list[i][j] == 0 and pre_investee_labels_list[i][j] == 1 and \
                #          pre_mount_labels_list[i][j] == 0:
                #     head += 1
                #     continue
                # elif pre_investor_labels_list[i][j] == 0 and pre_investee_labels_list[i][j] == 0 and \
                #         pre_mount_labels_list[i][j] == 1:
                #     head += 1
                #     continue
                else:
                    pass
            elif investor_labels_list[i][j] == 1 and investee_labels_list[i][j] == 1 and mount_labels_list[i][j] == 0:
                if pre_investor_labels_list[i][j] == 1 and pre_investee_labels_list[i][j] == 1 and \
                        pre_mount_labels_list[i][j] == 0:
                    head += 1
                    continue
                elif pre_investor_labels_list[i][j] == 1 and pre_investee_labels_list[i][j] == 0 and \
                        pre_mount_labels_list[i][j] == 0:
                    head += 1
                    continue
                elif pre_investor_labels_list[i][j] == 0 and pre_investee_labels_list[i][j] == 1 and \
                        pre_mount_labels_list[i][j] == 0:
                    head += 1
                    continue
                else:
                    pass
            elif investor_labels_list[i][j] == 1 and investee_labels_list[i][j] == 0 and mount_labels_list[i][j] == 1:
                if pre_investor_labels_list[i][j] == 1 and pre_investee_labels_list[i][j] == 0 and \
                        pre_mount_labels_list[i][j] == 1:
                    head += 1
                    continue
                elif pre_investor_labels_list[i][j] == 1 and pre_investee_labels_list[i][j] == 0 and \
                        pre_mount_labels_list[i][j] == 0:
                    head += 1
                    continue
                elif pre_investor_labels_list[i][j] == 0 and pre_investee_labels_list[i][j] == 0 and \
                        pre_mount_labels_list[i][j] == 1:
                    head += 1
                    continue
                else:
                    pass
            elif investor_labels_list[i][j] == 0 and investee_labels_list[i][j] == 1 and mount_labels_list[i][j] == 1:
                if pre_investor_labels_list[i][j] == 0 and pre_investee_labels_list[i][j] == 1 and \
                        pre_mount_labels_list[i][j] == 1:
                    head += 1
                    continue
                elif pre_investor_labels_list[i][j] == 0 and pre_investee_labels_list[i][j] == 1 and \
                        pre_mount_labels_list[i][j] == 0:
                    head += 1
                    continue
                elif pre_investor_labels_list[i][j] == 0 and pre_investee_labels_list[i][j] == 0 and \
                        pre_mount_labels_list[i][j] == 1:
                    head += 1
                    continue
                else:
                    pass
            elif investor_labels_list[i][j] == 1 and investee_labels_list[i][j] == 0 and mount_labels_list[i][j] == 0:
                if pre_investor_labels_list[i][j] == 1 and pre_investee_labels_list[i][j] == 0 and \
                        pre_mount_labels_list[i][j] == 0:
                    head += 1
                    continue
                else:
                    pass
            elif investor_labels_list[i][j] == 0 and investee_labels_list[i][j] == 1 and mount_labels_list[i][j] == 0:
                if pre_investor_labels_list[i][j] == 0 and pre_investee_labels_list[i][j] == 1 and \
                        pre_mount_labels_list[i][j] == 0:
                    head += 1
                    continue
                else:
                    pass
            elif investor_labels_list[i][j] == 0 and investee_labels_list[i][j] == 0 and mount_labels_list[i][j] == 1:
                if pre_investor_labels_list[i][j] == 0 and pre_investee_labels_list[i][j] == 0 and \
                        pre_mount_labels_list[i][j] == 1:
                    head += 1
                    continue
                else:
                    pass
            elif investor_labels_list[i][j] == 0 and investee_labels_list[i][j] == 0 and mount_labels_list[i][j] == 0:
                if pre_investor_labels_list[i][j] == 0 and pre_investee_labels_list[i][j] == 0 and \
                        pre_mount_labels_list[i][j] == 0:
                    head += 1
                    continue
                else:
                    pass
            else:
                print("error")
    count_len = 0
    for a in investee_labels_list:
        for b in a:
            count_len += 1

    print(head / count_len)
    print(test / test1)


def board_calulate(investor_labels_list, investee_labels_list, mount_labels_list, pre_investor_labels_list, \
                   pre_investee_labels_list, pre_mount_labels_list):
    head = 0
    test = 0
    test1 = 0
    cal = 0
    for i in range(len(investee_labels_list)):
        investor_flag = 0
        investee_flag = 0
        mount_flag = 0

        for j in range(len(investee_labels_list[i])):
            if investor_flag == 1 and investee_flag == 1 and mount_flag == 1:
                cal += 1
                break
            # if (investor_flag == 1 and investee_flag == 1) or (investor_flag == 1 and mount_flag == 1) or (
            #         mount_flag == 1 and investee_flag == 1):
            #     cal += 1
            #     break

            print(investor_labels_list[i][j], investee_labels_list[i][j], mount_labels_list[i][j])
            if investor_labels_list[i][j] == 1 and investee_labels_list[i][j] == 1 and mount_labels_list[i][j] == 1:
                test1 += 1
                if pre_investor_labels_list[i][j] == 1 and pre_investee_labels_list[i][j] == 1 and \
                        pre_mount_labels_list[i][j] == 1:
                    head += 1
                    test += 1
                    if investor_flag == 0:
                        investor_flag = 1
                    if investee_flag == 0:
                        investee_flag = 1
                    if mount_flag == 0:
                        mount_flag = 1
                    continue
                elif pre_investor_labels_list[i][j] == 1 and pre_investee_labels_list[i][j] == 1 and \
                        pre_mount_labels_list[i][j] == 0:
                    head += 1
                    test += 1
                    if investor_flag == 0:
                        investor_flag = 1
                    if investee_flag == 0:
                        investee_flag = 1
                    continue
                elif pre_investor_labels_list[i][j] == 1 and pre_investee_labels_list[i][j] == 0 and \
                        pre_mount_labels_list[i][j] == 1:
                    head += 1
                    test += 1
                    if investor_flag == 0:
                        investor_flag = 1
                    if mount_flag == 0:
                        mount_flag = 1
                    continue
                elif pre_investor_labels_list[i][j] == 0 and pre_investee_labels_list[i][j] == 1 and \
                        pre_mount_labels_list[i][j] == 1:
                    head += 1
                    test += 1
                    if investee_flag == 0:
                        investee_flag = 1
                    if mount_flag == 0:
                        mount_flag = 1
                    continue

                elif pre_investor_labels_list[i][j] == 1 and pre_investee_labels_list[i][j] == 0 and \
                        pre_mount_labels_list[i][j] == 0:
                    head += 1
                    test += 1
                    if investor_flag == 0:
                        investor_flag = 1
                    continue
                elif pre_investor_labels_list[i][j] == 0 and pre_investee_labels_list[i][j] == 1 and \
                        pre_mount_labels_list[i][j] == 0:
                    head += 1
                    test += 1
                    if investee_flag == 0:
                        investee_flag = 1
                    continue
                elif pre_investor_labels_list[i][j] == 0 and pre_investee_labels_list[i][j] == 0 and \
                        pre_mount_labels_list[i][j] == 1:
                    head += 1
                    test += 1
                    if mount_flag == 0:
                        mount_flag = 1
                    continue
                else:
                    pass
            elif investor_labels_list[i][j] == 1 and investee_labels_list[i][j] == 1 and mount_labels_list[i][j] == 0:
                if pre_investor_labels_list[i][j] == 1 and pre_investee_labels_list[i][j] == 1 and \
                        pre_mount_labels_list[i][j] == 0:
                    head += 1
                    if investor_flag == 0:
                        investor_flag = 1
                    if investee_flag == 0:
                        investee_flag = 1
                    continue
                elif pre_investor_labels_list[i][j] == 1 and pre_investee_labels_list[i][j] == 0 and \
                        pre_mount_labels_list[i][j] == 0:
                    head += 1
                    if investor_flag == 0:
                        investor_flag = 1
                    continue
                elif pre_investor_labels_list[i][j] == 0 and pre_investee_labels_list[i][j] == 1 and \
                        pre_mount_labels_list[i][j] == 0:
                    head += 1
                    if investee_flag == 0:
                        investee_flag = 1
                    continue
                elif pre_investor_labels_list[i][j] == 1 and pre_investee_labels_list[i][j] == 0 and \
                        pre_mount_labels_list[i][j] == 1:
                    head += 1
                    if investor_flag == 0:
                        investor_flag = 1
                    continue
                elif pre_investor_labels_list[i][j] == 0 and pre_investee_labels_list[i][j] == 1 and \
                        pre_mount_labels_list[i][j] == 1:
                    head += 1
                    if investee_flag == 0:
                        investee_flag = 1
                    continue
                else:
                    pass
            elif investor_labels_list[i][j] == 1 and investee_labels_list[i][j] == 0 and mount_labels_list[i][j] == 1:
                if pre_investor_labels_list[i][j] == 1 and pre_investee_labels_list[i][j] == 0 and \
                        pre_mount_labels_list[i][j] == 1:
                    head += 1
                    if investor_flag == 0:
                        investor_flag = 1
                    if mount_flag == 0:
                        mount_flag = 1
                    continue
                elif pre_investor_labels_list[i][j] == 1 and pre_investee_labels_list[i][j] == 0 and \
                        pre_mount_labels_list[i][j] == 0:
                    head += 1
                    if investor_flag == 0:
                        investor_flag = 1
                    continue
                elif pre_investor_labels_list[i][j] == 0 and pre_investee_labels_list[i][j] == 0 and \
                        pre_mount_labels_list[i][j] == 1:
                    head += 1
                    if mount_flag == 0:
                        mount_flag = 1
                    continue
                elif pre_investor_labels_list[i][j] == 1 and pre_investee_labels_list[i][j] == 1 and \
                        pre_mount_labels_list[i][j] == 0:
                    head += 1
                    if investor_flag == 0:
                        investor_flag = 1
                    continue
                elif pre_investor_labels_list[i][j] == 0 and pre_investee_labels_list[i][j] == 1 and \
                        pre_mount_labels_list[i][j] == 1:
                    head += 1
                    if mount_flag == 0:
                        mount_flag = 1
                    continue
                else:
                    pass
            elif investor_labels_list[i][j] == 0 and investee_labels_list[i][j] == 1 and mount_labels_list[i][j] == 1:
                if pre_investor_labels_list[i][j] == 0 and pre_investee_labels_list[i][j] == 1 and \
                        pre_mount_labels_list[i][j] == 1:
                    head += 1
                    if investee_flag == 0:
                        investee_flag = 1
                    if mount_flag == 0:
                        mount_flag = 1
                    continue
                elif pre_investor_labels_list[i][j] == 0 and pre_investee_labels_list[i][j] == 1 and \
                        pre_mount_labels_list[i][j] == 0:
                    head += 1
                    if investee_flag == 0:
                        investee_flag = 1
                    continue
                elif pre_investor_labels_list[i][j] == 0 and pre_investee_labels_list[i][j] == 0 and \
                        pre_mount_labels_list[i][j] == 1:
                    head += 1
                    if mount_flag == 0:
                        mount_flag = 1
                    continue
                elif pre_investor_labels_list[i][j] == 1 and pre_investee_labels_list[i][j] == 1 and \
                        pre_mount_labels_list[i][j] == 0:
                    head += 1
                    if investee_flag == 0:
                        investee_flag = 1
                    continue
                elif pre_investor_labels_list[i][j] == 1 and pre_investee_labels_list[i][j] == 0 and \
                        pre_mount_labels_list[i][j] == 1:
                    head += 1
                    if mount_flag == 0:
                        mount_flag = 1
                    continue
                else:
                    pass
            elif investor_labels_list[i][j] == 1 and investee_labels_list[i][j] == 0 and mount_labels_list[i][j] == 0:
                if pre_investor_labels_list[i][j] == 1:
                    head += 1
                    if investor_flag == 0:
                        investor_flag = 1
                    continue
                else:
                    pass
            elif investor_labels_list[i][j] == 0 and investee_labels_list[i][j] == 1 and mount_labels_list[i][j] == 0:
                if pre_investee_labels_list[i][j] == 1:
                    head += 1
                    if investee_flag == 0:
                        investee_flag = 1
                    continue
                else:
                    pass
            elif investor_labels_list[i][j] == 0 and investee_labels_list[i][j] == 0 and mount_labels_list[i][j] == 1:
                if pre_mount_labels_list[i][j] == 1:
                    head += 1
                    if mount_flag == 0:
                        mount_flag = 1
                    continue
                else:
                    pass
            elif investor_labels_list[i][j] == 0 and investee_labels_list[i][j] == 0 and mount_labels_list[i][j] == 0:
                if pre_investor_labels_list[i][j] == 0 and pre_investee_labels_list[i][j] == 0 and \
                        pre_mount_labels_list[i][j] == 0:
                    head += 1
                    continue
                else:
                    pass
            else:
                print("error")

    count_len = 0
    for a in investee_labels_list:
        for b in a:
            count_len += 1

    print(cal / len(investor_labels_list))
    print(test / test1)
    print(test)
    print(test1)


if __name__ == '__main__':
    datas = list()

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
    investor_labels_list, \
    investee_labels_list, \
    mount_labels_list, \
    pre_investor_labels_list, \
    pre_investee_labels_list, \
    pre_mount_labels_list = split_line(datas)
    # calculate(investor_labels_list, investee_labels_list, mount_labels_list, pre_investor_labels_list, \
    #           pre_investee_labels_list, pre_mount_labels_list)
    board_calulate(investor_labels_list, investee_labels_list, mount_labels_list, pre_investor_labels_list, \
                   pre_investee_labels_list, pre_mount_labels_list)

    # print(datas)
