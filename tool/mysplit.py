#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/5 下午12:57
# @Author  : jlinka
# @File    : mysplit.py

import re

number_pat = r"[0-9]"
word_pat = r"[a-z A-Z]"
reg_number = re.compile(number_pat)
reg_word = re.compile(word_pat)


def split(text):
    regchars = []
    chars = list(text)
    last_isnum = False
    numbers = []
    last_isword = False
    words = []
    for char in chars:
        # 是数字
        if re.fullmatch(reg_number, char) is not None:
            if last_isword:
                regchars.append("".join(words))
                words.clear()
                last_isword = False
            else:
                if last_isnum == False:
                    last_isnum = True
            numbers.append(char)
        # 是单词
        elif re.fullmatch(reg_word, char) is not None:
            if last_isnum:
                regchars.append("".join(numbers))
                numbers.clear()
                last_isnum = False
            else:
                if last_isword == False:
                    last_isword = True
            words.append(char)
        # 不是数字，也不是字符
        else:
            if last_isnum:
                regchars.append("".join(numbers))
                numbers.clear()
                last_isnum = False
            if last_isword:
                regchars.append("".join(words))
                words.clear()
                last_isword = False
            regchars.append(char)
    if len(numbers) > 0:
        regchars.append("".join(numbers))
    if len(words) > 0:
        regchars.append("".join(words))
    return regchars

if __name__ == '__main__':
    a = split('爱的说法sdfad')
    print(a)