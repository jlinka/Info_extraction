#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/18 下午3:53
# @Author  : jlinka
# @File    : extract_invest.py


from predict import BertPredictor
# from preprocess.bert_wordpre import bert_wpreprocess
from preprocess.bert_extract import bert_wpreprocess
# from preprocess.bert_wordpre import BertVector, InputExample, InputFeatures
import jieba

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class Extractor(object):
    def __init__(self):
        self._predictor = BertPredictor()

    def single_extract(self, resume: list):
        # bert = BertVector()
        resume = jieba.lcut(resume[0])
        resumes = []
        resumes.append(resume)
        final_label = self._extract(resumes)
        investor = []
        investee = []
        mount = []
        for i in range(len(resume)):
            if final_label[0][i] == '1':
                investor.append(resume[i])
            if final_label[0][i] == '2':
                investee.append(resume[i])
            if final_label[0][i] == '3':
                mount.append(resume[i])
        return investor, investee, mount

    def batch_extract(self, resumes: list):
        # bert = BertVector()
        regresumes = []

        for resume in resumes:
            third_word_labels_list, test_word_labels_list, train_word_labels_list = self._extract(resume)
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

    def _extract(self, resume: list):

        regwords_list = bert_wpreprocess.sentence2regwords(resume)
        wordvecs_list = bert_wpreprocess.word2vec(regwords_list)
        final_label_list = self._predictor.final_predict(wordvecs_list)

        # all_list = []
        # for i in range(len(third_word_labels_list)):
        #     all_list.append(str(test_word_labels_list[i]) + "," + str(train_word_labels_list[i]) + "," + str(
        #         third_word_labels_list))
        return final_label_list


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
    resume = ["4月18日，高思教育宣布完成D轮1.4亿美元融资，本轮投资方为华平投资。"]
    extractor = Extractor()
    a, b, c = extractor.single_extract(resume)
    print(a)
    print(b)
    print(c)
    # b = extractor.batch_extract([resume, resume])
    # print(b)
