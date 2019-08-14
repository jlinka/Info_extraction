#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/5 上午9:55
# @Author  : jlinka
# @File    : berth.py

import config
from bert.tokenization import FullTokenizer
from model.mybert import MyBertModel

tokenizer = FullTokenizer(vocab_file=config.CORPUS_DIC + "/vocab.txt")
bert_holder = MyBertModel()