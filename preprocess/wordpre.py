#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/2/25 上午10:52
# @Author  : jlinka
# @File    : wordpre.py

import logging
import numpy as np
import os
import jieba
import config
from tool import shuffle, bigfile
from holder.wordvech import wordvec_holder


class WordPreprocess(object):
    _total_labels = ['0', '1', 'no-info']

    # 获取标签列表
    def get_total_labels(self):
        return self._total_labels

    # 获取标签向量
    def get_labelvec(self, label: str):
        if label in self._total_labels:
            return self._total_labels.index(label)
        else:
            logging.warning("label {} is not exist".format(label))
            return len(self._total_labels) - 1

    # 获取标签
    def get_label(self, labelvec):
        labels = self._total_labels

        try:
            return labels[labelvec]
        except Exception as e:
            logging.error(e)
            exit(1)

    # 加载训练数据
    def load_traindata(self):
        try:
            wtrain_x = np.load(config.PREDATA_DIC + '/wtrain_x.npy')
            wtrain_y = np.load(config.PREDATA_DIC + '/wtrain_y.npy')
            return wtrain_x, wtrain_y
        except Exception as e:
            logging.error(e)
            exit(1)

    # 加载测试数据
    def load_testdata(self):
        try:
            wtest_x = np.load(config.PREDATA_DIC + '/wtest_x.npy')
            wtest_y = np.load(config.PREDATA_DIC + '/wtest_y.npy')
            return wtest_x, wtest_y
        except Exception as e:
            logging.error(e)
            exit(1)

    # 获取打乱后的训练数据
    def get_traindata(self):
        wtrain_x, wtrain_y = self.load_traindata()

        wtrain_x, wtrain_y = shuffle.shuffle_both(wtrain_x, wtrain_y)  # 打乱数据

        if len(wtrain_x) > 0:
            return wtrain_x, wtrain_y
        else:
            logging.error("train data length is less than 0")
            exit(1)

    # 获取打乱后的测试数据
    def get_testdata(self):
        wtest_x, wtest_y = self.load_testdata()

        wtest_x, wtest_y = shuffle.shuffle_both(wtest_x, wtest_y)  # 打乱数据

        if len(wtest_x) > 0:
            return wtest_x, wtest_y
        else:
            logging.error("test data length is less than 0")
            exit(1)

    # 批量获取打乱后的训练数据
    def get_batch_traindata(self, batch_size: int):
        wtrain_x, wtrain_y = self.get_traindata()

        total_size = len(wtrain_x)
        start = 0
        while start + batch_size < total_size:
            yield wtrain_x[start:start + batch_size], wtrain_y[start:start + batch_size]
            start += batch_size
        if len(wtrain_x[start:]) > 0:
            yield wtrain_x[start:], wtrain_y[start:]

    # 批量获取打乱后的测试数据
    def get_batch_testdata(self, batch_size: int):
        wtest_x, wtest_y = self.get_testdata()

        total_size = len(wtest_x)
        start = 0
        while start + batch_size < total_size:
            yield wtest_x[start:start + batch_size], wtest_y[start:start + batch_size]
            start += batch_size
        if len(wtest_x[start:]) > 0:
            yield wtest_x[start:], wtest_y[start:]

    def save_data(self, filename: str, data: list):
        try:
            if len(data) == 0:
                logging.warning("data length is 0")
                return
            np.save(config.PREDATA_DIC + "/" + filename, np.array(data))
            logging.info("save data file {} sucess".format(filename))
        except Exception as e:
            logging.error(e)
            exit(1)

    # 删除训练数据
    def remove_traindata(self):
        try:
            os.remove(config.PREDATA_DIC + "/wtrain_x.npy")
            os.remove(config.PREDATA_DIC + "/wtrain_y.npy")
            logging.info("remove train data success")
        except Exception as e:
            logging.warning(e)

    # 删除测试数据
    def remove_testdata(self):
        try:
            os.remove(config.PREDATA_DIC + "/wtest_x.npy")
            os.remove(config.PREDATA_DIC + "/wtest_y.npy")
            logging.info("remove test data success")
        except Exception as e:
            logging.warning(e)

    # 句子分词 返回处理好的词列表
    def sentence2regwords(self, words_list: list):
        new_words_list = []

        for words in words_list:
            new_words = []

            sent_len = 0
            for word in words:
                if sent_len < config.SENTENCE_LEN:
                    new_words.append(word)
                else:
                    break
                sent_len += 1

            while sent_len < config.SENTENCE_LEN:
                new_words.append('。')
                sent_len += 1

            new_words_list.append(new_words)

        return new_words_list

    # 句子分词
    def sentence2words(self, sentences: list):
        words_list = []

        for sentence in sentences:
            words_list.append(jieba.lcut(sentence, HMM=True))
            # words_list.append(pynlpir.segment(sentence, pos_tagging=False))

        return words_list

    # 标签序列处理 返回相同长度的标签序列
    def labels2reglabels(self, labels_list: list):
        logging.info("deal labels to reglabels")
        reglabels_list = list()
        for labels in labels_list:
            new_labels = list()

            sent_len = 0
            for label in labels:
                if sent_len < config.SENTENCE_LEN:
                    new_labels.append(label)
                else:
                    break
                sent_len += 1

            while sent_len < config.SENTENCE_LEN:
                new_labels.append(self._total_labels[0])
                sent_len += 1

            reglabels_list.append(new_labels)
        return reglabels_list

    # 词转句向量
    def word2vec(self, words_list: list):
        logging.info("word to vector")
        docvecs_list = list()
        for words in words_list:
            docvecs = []
            for word in words:
                wordvecs = []
                for w in word:
                    try:
                        wordvecs.append(wordvec_holder.get(w))
                    except:
                        wordvecs.append(wordvec_holder.get("。"))
                M = np.array(wordvecs)
                v = M.sum(axis=0)
                result = v / np.sqrt((v ** 2).sum())
                try:
                    tuple(result)
                except:
                    result = wordvec_holder.get("。")
                docvecs.append(result)
            docvecs_list.append(docvecs)
        return docvecs_list

    # 标签转标签向量
    def label2vec(self, labels_list: list):
        logging.info("label to vector")
        labelvecs_list = list()  # 标签向量列表
        for labels in labels_list:
            labelvecs = list()
            for label in labels:
                labelvecs.append(self.get_labelvec(label))
            labelvecs_list.append(labelvecs)
        return labelvecs_list

    # 处理标注数据
    def deal_tagdata(self, tagdata_filepaths: list, rate: float = config.WR_RATE):
        logging.info("begin deal word tag data")
        if rate < 0 or rate > 1:
            logging.error("rate is not between 0 and 1")
            exit(1)

        datas = list()
        if '.DS_Store' in tagdata_filepaths:
            tagdata_filepaths.remove('.DS_Store')
        for tagdata_filepath in tagdata_filepaths:
            if os.path.exists(config.TAG_DIC + '/sr/seq/' + tagdata_filepath):
                para = []
                num = 0
                for line in bigfile.get_lines(config.TAG_DIC + '/sr/seq/' + tagdata_filepath):
                    if num == 0:
                        num = 1
                        continue
                    if line != ',,,;;;;;;;;;;\n':
                        para.append(line.replace('\n', ''))
                    else:
                        datas.append(para)
                        para = []
            else:
                logging.warning("tag data file {} is not exist".format(tagdata_filepath))
                raise FileNotFoundError('{} 标注数据文件不存在'.format(tagdata_filepath))

        words_list, labels_list = self.split_tagdata(datas)
        datas = None

        regwords_list = self.sentence2regwords(words_list)
        reglabels_list = self.labels2reglabels(labels_list)
        words_list = None
        labels_list = None

        regwords_list, reglabels_list = shuffle.shuffle_both(regwords_list, reglabels_list)

        wordvecs_list = self.word2vec(regwords_list)
        labelvecs_list = self.label2vec(reglabels_list)
        regwords_list = None
        reglabels_list = None

        # 将数据保存下来
        total_size = len(labelvecs_list)

        train_x = wordvecs_list[:int(total_size * rate)]
        train_y = labelvecs_list[:int(total_size * rate)]
        test_x = wordvecs_list[int(total_size * rate):]
        test_y = labelvecs_list[int(total_size * rate):]
        wordvecs_list = None
        labelvecs_list = None

        logging.info("deal word tag data end")
        return train_x, train_y, test_x, test_y

    def split_tagdata(self, datas: list):
        return self._split_tagdata(datas)

    def _split_tagdata(self, datas: list):
        words_list = list()  # 保存分词后的句子 每一项是字符串： 词 词
        labels_list = list()  # 保存标签 每一项是字符串： label label
        num = 0
        for news in datas:
            words = []
            labels = []
            for sentence in news:
                if sentence[4:5] == '1':
                    num += 1
                labels.append(sentence[2:3])
                words.append(jieba.lcut(sentence[6:]))
            labels_list.append(labels)
            words_list.append(words)
        print(num)
        return words_list, labels_list


preprocess = WordPreprocess()

if __name__ == '__main__':
    a, b, c, d = preprocess.deal_tagdata(os.listdir(config.TAG_DIC + '/sr/seq/'))
    np.save(config.PREDATA_DIC + '/wtrain_x.npy', np.array(a))
    np.save(config.PREDATA_DIC + '/wtrain_y.npy', np.array(b))
    np.save(config.PREDATA_DIC + '/wtest_x.npy', np.array(c))
    np.save(config.PREDATA_DIC + '/wtest_y.npy', np.array(d))
