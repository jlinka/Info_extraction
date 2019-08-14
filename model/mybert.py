#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/5 上午10:09
# @Author  : jlinka
# @File    : mybert.py

import logging
import tensorflow as tf

import config
from bert.modeling import BertConfig, BertModel


class MyBertModel(object):
    def __init__(self, seq_length: int = config.SENTENCE_LEN, model_name: str = "bert_model.ckpt"):
        self._seq_length = seq_length
        self._model_name = model_name

        self._session, self._ph_input_ids, self._output = self.get_trained_model()

    def get_model(self):
        logging.info("get bert model")
        graph = tf.Graph()
        with graph.as_default():
            ph_input_ids = tf.placeholder(dtype=tf.int32, shape=[None, self._seq_length + 2], name="ph_input_ids")
            con = BertConfig.from_json_file(config.PROJECT_ROOT + "/bert_config.json")
            bert_model = BertModel(config=con, is_training=False, input_ids=ph_input_ids,
                                   use_one_hot_embeddings=True)
            output = bert_model.get_sequence_output()
            init = tf.global_variables_initializer()

        sess = tf.Session(graph=graph)
        sess.run(init)

        return sess, ph_input_ids, output

    def get_trained_model(self):
        logging.info("get trained bert model")
        sess, ph_input_ids, output = self.get_model()
        try:
            with sess.graph.as_default():
                saver = tf.train.Saver()
                saver.restore(sess, config.MODEL_DIC + "/" + self._model_name)
        except Exception as e:
            logging.error(e)
            exit(1)
        return sess, ph_input_ids, output

    def predict(self, inputs):
        # inputs shape [batch_size, self._seq_length + 2] [CLS] words... [SEP]
        embeddings = []
        with self._session.graph.as_default():
            for input in inputs:
                embedding = self._session.run(self._output, feed_dict={self._ph_input_ids: [input]})
                embeddings.append(embedding[0])
        return embeddings
        # embeddings shape [batch_size, seq_length + 2, word_size]


if __name__ == '__main__':
    MyBertModel()
