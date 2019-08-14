#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/5 上午10:08
# @Author  : jlinka
# @File    : jiebah.py

import jieba
import logging

import config


class JiebaHolder(object):

    # # 加载词库
    # def _load_userdict(self):
    #     logging.debug("load userdict")
    #     jieba.load_userdict(config.CORPUS_DIC + "/major.txt")
    #     jieba.load_userdict(config.CORPUS_DIC + "/school.txt")
    #     jieba.load_userdict(config.CORPUS_DIC + "/name.txt")
    #
    # # 调整词频
    # def _suggest_freq(self):
    #     logging.debug("suggest freq")
    #     jieba.suggest_freq((u'生', u'于'), True)
    #     jieba.suggest_freq((u'日', u'于'), True)
    #     jieba.suggest_freq((u'月', u'于'), True)
    #     jieba.suggest_freq((u'年', u'于'), True)
    #
    #     jieba.suggest_freq((u'部', u'部长'), True)
    #     jieba.suggest_freq((u'科', u'科长'), True)
    #     jieba.suggest_freq((u'局', u'局长'), True)
    #     jieba.suggest_freq((u'处', u'处长'), True)
    #     jieba.suggest_freq((u'组', u'组长'), True)
    #     jieba.suggest_freq((u'班', u'组长'), True)
    #     jieba.suggest_freq((u'班', u'班长'), True)
    #     jieba.suggest_freq((u'会', u'会长'), True)
    #
    #     jieba.suggest_freq((u'起', u'任'), True)
    #     jieba.suggest_freq((u'现', u'任'), True)
    #
    #     jieba.suggest_freq(u'任职员', True)
    #
    #     jieba.suggest_freq((u'年', u'生'), True)
    #     jieba.suggest_freq((u'月', u'生'), True)
    #     jieba.suggest_freq((u'日', u'生'), True)
    #
    #     jieba.suggest_freq((u'年', u'任'), True)
    #     jieba.suggest_freq((u'月', u'任'), True)
    #     jieba.suggest_freq((u'日', u'任'), True)
    #
    #     jieba.suggest_freq(u'入职', True)
    #
    #     jieba.suggest_freq((u'年', u'入'), True)
    #     jieba.suggest_freq((u'月', u'入'), True)
    #     jieba.suggest_freq((u'日', u'入'), True)
    #
    #     jieba.suggest_freq((u'事务所', u'律师'), True)
    #
    #     jieba.suggest_freq((u'年', u'起'), True)
    #     jieba.suggest_freq((u'月', u'起'), True)
    #     jieba.suggest_freq((u'日', u'起'), True)
    #
    #     jieba.suggest_freq((u'部', u'副'), True)
    #     jieba.suggest_freq((u'科', u'副'), True)
    #     jieba.suggest_freq((u'局', u'副'), True)
    #     jieba.suggest_freq((u'处', u'副'), True)
    #     jieba.suggest_freq((u'组', u'副'), True)
    #     jieba.suggest_freq((u'会', u'副'), True)
    #
    #     jieba.suggest_freq((u'年', u'获'), True)
    #     jieba.suggest_freq((u'月', u'获'), True)
    #     jieba.suggest_freq((u'日', u'获'), True)
    #
    #     jieba.suggest_freq(u'人资', True)
    #     jieba.suggest_freq(u'软装设计', True)
    #     jieba.suggest_freq(u'娃哈哈', True)
    #     jieba.suggest_freq(u'哇哈哈', True)
    #     jieba.suggest_freq(u'财务负责人', True)
    #
    #     jieba.suggest_freq((u'公司', u'员工'), True)
    #     jieba.suggest_freq((u'公司', u'财务'), True)
    #
    #     jieba.suggest_freq((u'会计', u'职称'), True)
    #
    #     jieba.suggest_freq((u'协会', u'会员'), True)
    #     jieba.suggest_freq((u'协会', u'理事'), True)
    #
    #     jieba.suggest_freq(u'副班长', True)
    #     jieba.suggest_freq(u'副市长', True)
    #     jieba.suggest_freq(u'副处长', True)
    #     jieba.suggest_freq(u'副科长', True)
    #     jieba.suggest_freq(u'副部长', True)
    #     jieba.suggest_freq(u'副组长', True)
    #     jieba.suggest_freq(u'副经理', True)
    #     jieba.suggest_freq(u'副总经理', True)
    #     jieba.suggest_freq(u'副总裁', True)
    #     jieba.suggest_freq(u'副会长', True)
    #     jieba.suggest_freq(u'副教授', True)
    #     jieba.suggest_freq(u'副主任', True)
    #     jieba.suggest_freq(u'副院长', True)
    #     jieba.suggest_freq(u'副校长', True)
    #     jieba.suggest_freq(u'副董事长', True)
    #     jieba.suggest_freq(u'副厂长', True)
    #     jieba.suggest_freq(u'副董事', True)
    #
    #     jieba.suggest_freq((u'总裁', u'兼'), True)
    #
    #     jieba.suggest_freq((u'月', u'间'), True)
    #
    #     jieba.suggest_freq((u'管理', u'工作'), True)
    #
    #     jieba.suggest_freq((u'支部', u'书记'), True)
    #
    #     jieba.suggest_freq((u'获', u'工学'), True)
    #
    # # 其他处理
    # def _other_deal(self, words: list):
    #     new_words = list()
    #     for word in words:
    #         word_len = len(word)
    #         if word_len >= 2:
    #             if word.startswith(u'于'):
    #                 new_words.append(word[0])
    #                 new_words.append(word[1:])
    #                 break
    #
    #             elif word.endswith(u'于'):
    #                 if word == u'就读于' or word == u'就职于' or word == u'任职于' or word == u'毕业于' or word == u'出生于':
    #                     new_words.append(word)
    #                     break
    #                 else:
    #                     new_words.append(word[:-1])
    #                     new_words.append(word[-1])
    #
    #             elif word == u'任职员':
    #                 new_words.append(u'任')
    #                 new_words.append(u'职员')
    #
    #             elif word.startswith(u'任'):
    #                 if word == u'任职' or word == u'任职于' or word == u'任期' or word == u'任教' or word == u'任教于' or word == u'任命':
    #                     new_words.append(word)
    #                 else:
    #                     new_words.append(word[0])
    #                     new_words.append(word[1:])
    #
    #             elif word.endswith(u'任'):
    #                 if word == u'主任' or word == u'副主任' or word == u'担任' or word == u'历任' or word == u'现任' or word == u'兼任' or word == u'有限责任' or word == u'责任':
    #                     new_words.append(word)
    #                 else:
    #                     new_words.append(word[:-1])
    #                     new_words.append(word[-1])
    #
    #             elif word.endswith(u'专业') or word.endswith(u'学历') or word.endswith(u'毕业') or word.endswith(
    #                     u'硕士') or word.endswith(u'博士') or word.endswith(u'学位'):
    #                 if word[:-2]:
    #                     new_words.append(word[:-2])
    #                 new_words.append(word[-2:])
    #
    #             elif word.startswith(u'小学') or word.startswith(u'中学') or word.startswith(u'大学'):
    #                 new_words.append(word[:2])
    #                 if word[2:]:
    #                     new_words.append(word[2:])
    #
    #             else:
    #                 new_words.append(word)
    #         else:
    #             new_words.append(word)
    #     return new_words

    # 处理
    def lcut(self, sentence: str):
        words = jieba.lcut(sentence)
        # other deal
        return self._other_deal(words)


jieba_holder = JiebaHolder()
