#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/2/25 下午2:53
# @Author  : jlinka
# @File    : predict.py

# from model.wordrec import WordRecModel
from model.bert_wordrec import BertWordRecModel
from model.bert_extract_model import WordRecModel


class BertPredictor(object):
    def __init__(self, third_model_name: str = "third/bert_wordrec.ckpt",
                 test_model_name: str = "test/bert_wordrec.ckpt",
                 train_model_name: str = "train/bert_wordrec.ckpt",
                 final_model_name: str = "wordrec.ckpt"):
        self._third_model = BertWordRecModel(model_name=third_model_name, predictor=True)
        self._test_model = BertWordRecModel(model_name=test_model_name, predictor=True)
        self._train_model = BertWordRecModel(model_name=train_model_name, predictor=True)
        self._final_model = WordRecModel(model_name=final_model_name, predictor=True)

    def third_word_predict(self, inputs):
        return self._third_model.predict_label(inputs)

    def test_word_predict(self, inputs):
        return self._test_model.predict_label(inputs)

    def train_word_predict(self, inputs):
        return self._train_model.predict_label(inputs)

    def final_predict(self, inputs):
        return self._final_model.predict_label(inputs)


predictor = BertPredictor()
