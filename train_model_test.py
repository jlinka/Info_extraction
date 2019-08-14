#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/2/25 下午4:50
# @Author  : jlinka
# @File    : train_model_test.py

import argparse
import logging
import config
import os
import numpy as np
from model.wordrec import WordRecModel
from model.wordrec import wpreprocess
from model.wordrec import defualt_model
from model.bert_wordrec import bert_default_model
from model.bert_wordrec import bert_wpreprocess




def deal_wordrec_data():
    train_x, train_y, test_x, test_y = wpreprocess.deal_tagdata(os.listdir(config.TAG_DIC + '/sr/seq/'))
    np.save(config.PREDATA_DIC + '/wtrain_x.npy', np.array(train_x))
    np.save(config.PREDATA_DIC + '/wtrain_y.npy', np.array(train_y))
    np.save(config.PREDATA_DIC + '/wtest_x.npy', np.array(test_x))
    np.save(config.PREDATA_DIC + '/wtest_y.npy', np.array(test_y))


def train_wordrec_model():
    defualt_model.train()
    defualt_model.test()


def deal_bert_wordred_data():
    bert_wtrain_x, bert_wtrain_y, bert_wtest_x, bert_wtest_y = bert_wpreprocess.deal_tagdata(
        os.listdir(config.TAG_WR_DIC))
    np.save(config.PREDATA_DIC + '/bert_wtrain_x.npy', np.array(bert_wtrain_x))
    np.save(config.PREDATA_DIC + '/bert_wtrain_y.npy', np.array(bert_wtrain_y))
    np.save(config.PREDATA_DIC + '/bert_wtest_x.npy', np.array(bert_wtest_x))
    np.save(config.PREDATA_DIC + '/bert_wtest_y.npy', np.array(bert_wtest_y))


def train_bert_wordrec_model():
    bert_default_model.train()
    bert_default_model.test()


def main():
    parse = argparse.ArgumentParser()
    parse.add_argument("-t", "--type", type=str,
                       help="sentence is for training sentence recognition model, word is for training word recognition model!")
    args = parse.parse_args()
    if args.type == "word":
        deal_wordrec_data()
        train_wordrec_model()
    else:
        logging.critical("type error")


if __name__ == '__main__':
    # main()
    # deal_wordrec_data()
    # train_wordrec_model()
    deal_bert_wordred_data()
    # train_bert_wordrec_model()

