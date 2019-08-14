#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/2/25 上午10:37
# @Author  : jlinka
# @File    : evaluate.py

# 用来计算多分类的准确率、精准率、召回率、F1值

import collections
import logging
from sklearn.metrics import confusion_matrix
import numpy as np
import csv


class Evaluate(object):

    # 计算混淆矩阵
    def _calculate_confusion_matrix(self, y_true, y_pred):
        if len(y_true) != len(y_pred):
            logging.error("y_true length and y_pred length are not equal")
            exit(1)
        return confusion_matrix(y_true, y_pred)

    # 计算准确率
    def calculate_accuracy(self, y_true, y_pred):
        if len(y_true) != len(y_pred):
            logging.error("y_true length and y_pred length are not equal")
            exit(1)

        true_count = 0
        false_count = 0
        true_false = np.equal(y_true, y_pred)
        for value in true_false:
            if value:
                true_count += 1
            else:
                false_count += 1

        accuracy = 0
        if (true_count + false_count) != 0:
            accuracy = true_count / (true_count + false_count) * 1.0

        return round(accuracy, 3)

    # 计算精准率
    def calculate_precision(self, y_true, y_pred):
        precisions = list()
        matrix = self._calculate_confusion_matrix(y_true, y_pred)
        classes = len(matrix)
        for i in range(classes):
            TP = matrix[i][i]
            FP = 0
            for j in range(classes):
                if j == i:
                    pass
                else:
                    FP += matrix[i][j]

            precision = 0.0
            if (TP + FP) != 0:
                precision = TP / (TP + FP) * 1.0

            precisions.append(precision)
        return precisions

    # 计算recall值
    def calculate_recall(self, y_true, y_pred):
        recalls = list()
        matrix = self._calculate_confusion_matrix(y_true, y_pred)
        classes = len(matrix)
        for i in range(classes):
            TP = matrix[i][i]
            FN = 0
            for j in range(classes):
                if j == i:
                    pass
                else:
                    FN += matrix[j][i]

            recall = 0.0
            if (TP + FN) != 0:
                recall = TP / (TP + FN) * 1.0

            recalls.append(recall)
        return recalls

    # 计算f1值
    def calculate_f1(self, y_true, y_pred):
        f1s = list()
        matrix = self._calculate_confusion_matrix(y_true, y_pred)
        classes = len(matrix)
        for i in range(classes):
            TP = matrix[i][i]
            FP = 0
            FN = 0
            for j in range(classes):
                if j == i:
                    pass
                else:
                    FP += matrix[i][j]
                    FN += matrix[j][i]

            precision = 0.0
            if (TP + FP) != 0:
                precision = TP / (TP + FP) * 1.0

            recall = 0.0
            if (TP + FN) != 0:
                recall = TP / (TP + FN) * 1.0

            f1 = 0.0
            if (precision + recall) != 0:
                f1 = 2 * precision * recall / (precision + recall) * 1.0

            f1s.append(f1)

        return f1s

    def calculate_prf(self, y_true, y_pred):
        precisions = list()
        recalls = list()
        f1s = list()
        matrix = self._calculate_confusion_matrix(y_true, y_pred)
        classes = len(matrix)
        for i in range(classes):
            TP = matrix[i][i]
            FP = 0
            FN = 0
            for j in range(classes):
                if j == i:
                    pass
                else:
                    FP += matrix[i][j]
                    FN += matrix[j][i]

            precision = 0.0
            if (TP + FP) != 0:
                precision = TP / (TP + FP) * 1.0

            recall = 0.0
            if (TP + FN) != 0:
                recall = TP / (TP + FN) * 1.0

            f1 = 0.0
            if (precision + recall) != 0:
                f1 = 2 * precision * recall / (precision + recall) * 1.0

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

        return precisions, recalls, f1s

    def calculate_avg_prf(self, y_true, y_pred):
        precisions, recalls, f1s = self.calculate_prf(y_true, y_pred)
        return round(float(np.mean(precisions)), 3), round(float(np.mean(recalls)), 3), round(float(np.mean(f1s)), 3)

    def print_evaluate(self, y_true, y_pred, labels):
        """
        :param y_true: shape like [1,2,3,4,5,4]
        :param y_pred: shape is same as y_true
        :param labels: shape like ["lalal","sdf","dfh"]
        """

        label_dic = collections.OrderedDict()  # key:label value:index
        for index in range(len(labels)):
            label_dic[labels[index]] = index
        acc = self.calculate_accuracy(y_true, y_pred)
        precisions, recalls, f1s = self.calculate_prf(y_true, y_pred)
        if len(precisions) == len(recalls) == len(f1s):
            if len(labels) != len(precisions):
                unuse_labels = list()

                for label in label_dic.keys():
                    if label_dic[label] not in y_true and label_dic[label] not in y_pred:
                        unuse_labels.append(label)

                for label in unuse_labels:
                    label_dic.pop(label)

                if len(unuse_labels) > 0:
                    logging.warning("these labels are not used {}".format(unuse_labels))

            total_true_count = 0
            total_false_count = 0
            i = 0
            out = open('result.csv', 'a', newline='')
            csv_write = csv.writer(out, dialect='excel')
            for label in label_dic.keys():
                true_count = 0
                false_count = 0

                for true, pred in zip(y_true, y_pred):
                    if true == label_dic[label]:
                        if true == pred:
                            true_count += 1
                        else:
                            false_count += 1

                total_true_count += true_count
                total_false_count += false_count

                print('label:', label, 'precision:', round(precisions[i], 3), 'recall:', round(recalls[i], 3),
                      'f1_score:', round(f1s[i], 3),
                      'true count:', true_count, 'false count:', false_count)
                csv_write.writerow(['label', 'precision', 'recall', 'f1_score', 'true count', 'false count'])
                csv_write.writerow(
                    [label, round(precisions[i], 3), round(recalls[i], 3), round(f1s[i], 3), true_count, false_count])
                i += 1

            avg_precision = float(np.mean(precisions))
            avg_recall = float(np.mean(recalls))
            avg_f1 = float(np.mean(f1s))
            print('total label ', "acc:", round(acc, 3), 'precision:', round(avg_precision, 3), 'recall:',
                  round(avg_recall, 3),
                  'f1_score:', round(avg_f1, 3), 'total true count:', total_true_count, 'total false count:',
                  total_false_count)
            csv_write.writerow(['total label acc', 'precision', 'recall', 'f1_score', 'total true count', 'total false count'])
            csv_write.writerow(
                [round(acc, 3), round(avg_precision, 3), round(avg_recall, 3), round(avg_f1, 3), total_true_count, total_false_count])
            print()


            print("result write over")


        else:
            logging.error("precisions length,recalls length and f1s length are not equal")
            exit(1)


default_evaluate = Evaluate()

if __name__ == '__main__':
    y_true = [2, 3, 4, 3, 6, 1, 5, 0, 0, 3, 4, 3, 6, 1, 0, 3, 4, 3, 6, 1, 0, 3, 4, 3, 6, 1, 0, 3, 4, 3, 6, 1, 0, 1]
    y_pred = [2, 3, 4, 3, 5, 6, 0, 0, 5, 3, 4, 3, 5, 6, 0, 0, 5, 3, 4, 3, 5, 6, 0, 0, 5, 3, 4, 3, 5, 6, 0, 0, 5, 8]

    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8']

    print(default_evaluate.calculate_accuracy(y_true, y_pred))
    print(default_evaluate.calculate_avg_prf(y_true, y_pred))
    default_evaluate.print_evaluate(y_true, y_pred, labels)
