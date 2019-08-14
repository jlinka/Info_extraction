#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/2/25 上午10:23
# @Author  : jlinka
# @File    : extract.py

# import extract.predict as predict
# from .predict import BertPredictor
# from extract.predict import BertPredictor
from predict import BertPredictor
from preprocess.bert_wordpre import bert_wpreprocess
# from extract.predict import BertPredictor
# from preprocess.bert_wordpre_test import default_preprocess as wpreprocess
import logging
import numpy as np
import os
import jieba
import config
from tool import shuffle, bigfile
from preprocess.bert_wordpre import BertVector, InputExample, InputFeatures

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class Extractor(object):
    def __init__(self):
        self._predictor = BertPredictor()

    def single_extract(self, resume: list):
        bert = BertVector()
        third_word_labels_list, test_word_labels_list, train_word_labels_list = self._extract(resume, bert)
        all_list = []
        for i in range(len(third_word_labels_list[0])):
            all_list.append(str(test_word_labels_list[0][i]) + "," + str(train_word_labels_list[0][i]) + "," + str(
                third_word_labels_list[0][i]))
        return all_list

    def batch_extract(self, resumes: list):
        bert = BertVector()
        regresumes = []

        for resume in resumes:
            third_word_labels_list, test_word_labels_list, train_word_labels_list = self._extract(resume, bert)
            all_list = []
            for i in range(len(third_word_labels_list[0])):
                all_list.append(str(test_word_labels_list[0][i]) + "," + str(train_word_labels_list[0][i]) + "," + str(
                    third_word_labels_list[0][i]))
            print(all_list)
            regresumes.append(all_list)
        return regresumes

    # 批量简历抽取生成器
    def batch_extract_generator(self, resumes: list):
        for resume in resumes:
            yield self._extract(resume)

    def _extract(self, resume: list, bert):

        regwords_list = bert_wpreprocess.sentence2regwords(resume)
        wordvecs_list = bert_wpreprocess.test_word2vec(regwords_list, bert)
        third_word_labels_list = self._predictor.third_word_predict(wordvecs_list)
        test_word_labels_list = self._predictor.test_word_predict(wordvecs_list)
        train_word_labels_list = self._predictor.train_word_predict(wordvecs_list)

        # all_list = []
        # for i in range(len(third_word_labels_list)):
        #     all_list.append(str(test_word_labels_list[i]) + "," + str(train_word_labels_list[i]) + "," + str(
        #         third_word_labels_list))
        return third_word_labels_list, test_word_labels_list, train_word_labels_list


def _split_tagdata(datas: list):
    words_list = list()  # 保存分词后的句子 每一项是字符串： 词 词
    labels_list = list()  # 保存标签 每一项是字符串： label label

    for news in datas:
        words = []
        labels = []
        for sentence in news:
            if sentence[6:] != '':
                labels.append(sentence[:5])
                words.append(sentence[6:])
        labels_list.append(labels)
        words_list.append([words])

    return words_list, labels_list


if __name__ == '__main__':
    resume = [[
        "2017年3月22日消息，印度全渠道支付平台Airpay宣布获得376万美元A轮融资，本次交易的领投方为KalaariCapital，早期投资人RakeshJhunjhunwala以及RajeshJhunjhunwala均参与了本轮融资",
        "据创投时报了解，Airpay计划利用本轮资金提升技术研发能力，建设销售以及配送团队，同时提升基础设施从而快速拓展企业客户", "谈及本轮融资，KalaariCapital总裁RajeshRaju表示",
        "对于印度支付领域而言，渠道以及工具的结合是非常复杂的，需要一套创新性的一站式解决方案，而Airpay所提供的解决方案就具备这样的特点",
        "由于Airpay有能力将印度推进到一个无现金经济的时代，我们非常高兴能够与他们达成合作",
        "据创投时报项目库数据显示，Airpay由KunalJhunjhunwala、AmitKapoor以及RohanDeshpande于2012年联合创办，总部位于印度孟买，是一家全渠道支付公司，为企业客户提供SaaS解决方案，既能够接受消费者C2B的付款，也能够处理对外B2B的结算问题"]]
    extractor = Extractor()
    # a = extractor.single_extract(resume)
    # print(a)
    b = extractor.batch_extract([resume, resume])

    print(b)
    # datas = list()
    #
    # if os.path.exists(config.TEST_TAG_DIC + '/4000__5000.csv'):
    #     para = []
    #     num = 0
    #     flag = 0
    #     for line in bigfile.get_lines(config.TEST_TAG_DIC + '/4000__5000.csv'):
    #         if num == 0:
    #             num = 1
    #             continue
    #         if line != ',,,;;;;;;;;;;\n':
    #             para.append(line.replace('\n', ''))
    #         else:
    #             # flag += 1
    #             datas.append(para)
    #             para = []
    #         if flag == 2:
    #             break
    #
    # words_list, labels_list = _split_tagdata(datas)
    # extractor = Extractor()
    #
    # all_list = extractor.batch_extract(words_list)
    # # np.save('/test_data.npy', np.array(all_list))
    # with open('preditor.csv', 'a', encoding='UTF-8-sig') as wf:
    #     wf.write("investor,investee,mount,pre_investor,pre_investee,pre_mount,sentence\n")
    #     for i in range(len(labels_list)):
    #         for j in range(len(labels_list[i])):
    #             num += 1
    #             wf.write(labels_list[i][j] + ',' + all_list[i][j] + ',' + words_list[i][0][j] + '\n')
    #             if j == 40:
    #                 break
    #         wf.write(",,,,,,;;;;;;;;;;\n")
