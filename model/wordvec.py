#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/2/25 上午10:35
# @Author  : jlinka
# @File    : wordvec.py

import gensim
import logging
from multiprocessing import cpu_count
import sys
import config


class WordVecModel(object):
    def __init__(self, corpus_filepath: str = config.PREDATA_DIC + "/totalpart.txt",
                 wordvec_filepath: str = config.MODEL_DIC + "/wordvec.model",
                 wordvec_size: int = config.WORDVEC_SIZE):
        self._corpus_filepath = corpus_filepath
        self._wordvec_filepath = wordvec_filepath
        self._wordvec_size = wordvec_size

    # 训练词向量模型
    def train(self):
        logging.info("train word vector model")
        sentences = gensim.models.word2vec.Text8Corpus(self._corpus_filepath)  # 加载分词后的文件
        model = gensim.models.Word2Vec(sentences, size=self._wordvec_size, window=5, min_count=1,
                                       workers=cpu_count())  # 训练词向量模型
        model.save(self._wordvec_filepath)  # 保存词向量模型

    # 加载训练好的模型
    def load_trained_model(self):
        try:
            model = gensim.models.Word2Vec.load(self._wordvec_filepath)
            logging.info("load trained wordvec model")
            return model
        except Exception as e:
            logging.error(e)
            exit(1)

    # 训练更多的词向量
    def train_more(self, more_filepaths: list):
        if len(more_filepaths) == 0:
            logging.warning("more_filepaths length is 0")
        logging.info("continue train word vector model")
        model = self.load_trained_model()
        for more_filepath in more_filepaths:
            sentences = gensim.models.word2vec.Text8Corpus(more_filepath)  # 加载分词后的文件
            model.train(sentences, epochs=model.iter, total_examples=model.corpus_count)
        model.save(self._wordvec_filepath)  # 保存词向量模型

if __name__ == '__main__':
    wordvec = WordVecModel()
    wordvec.train_more(['news_word.txt'])
