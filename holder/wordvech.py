#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/2/25 上午10:54
# @Author  : jlinka
# @File    : wordvech.py

import logging
import threading

from model.wordvec import WordVecModel


class WordVecHodler(object):
    def __init__(self):
        self._lock = threading.Lock()
        self._wordvec_model = WordVecModel().load_trained_model()

    def get(self, word: str):
        self._lock.acquire()
        try:
            vector = self._wordvec_model[word]
        except Exception as e:
            vector = self._wordvec_model["。"]
            logging.warning(e)
        self._lock.release()
        return vector


wordvec_holder = WordVecHodler()
