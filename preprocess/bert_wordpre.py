#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/5 下午4:04
# @Author  : jlinka
# @File    : bert_wordpre.py
import logging
import numpy as np
import os
import jieba
import config
from tool import shuffle, bigfile
# from holder.wordvech import wordvec_holder
import bert.modeling as modeling
import bert.tokenization as tokenization
from bert.graph import optimize_graph
import bert.args as args
from queue import Queue
from threading import Thread
import tensorflow as tf
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class InputExample(object):

    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


class BertVector:

    def __init__(self, batch_size=32):
        """
        init BertVector
        :param batch_size:     Depending on your memory default is 32
        """
        self.max_seq_length = args.max_seq_len
        self.layer_indexes = args.layer_indexes
        self.gpu_memory_fraction = 1
        self.graph_path = optimize_graph()
        self.tokenizer = tokenization.FullTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
        self.batch_size = batch_size
        self.estimator = self.get_estimator()
        self.input_queue = Queue(maxsize=1)
        self.output_queue = Queue(maxsize=1)
        self.predict_thread = Thread(target=self.predict_from_queue, daemon=True)
        self.predict_thread.start()
        self.sentence_len = 0

    def get_estimator(self):
        from tensorflow.python.estimator.estimator import Estimator
        from tensorflow.python.estimator.run_config import RunConfig
        from tensorflow.python.estimator.model_fn import EstimatorSpec

        def model_fn(features, labels, mode, params):
            with tf.gfile.GFile(self.graph_path, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())

            input_names = ['input_ids', 'input_mask', 'input_type_ids']

            output = tf.import_graph_def(graph_def,
                                         input_map={k + ':0': features[k] for k in input_names},
                                         return_elements=['final_encodes:0'])

            return EstimatorSpec(mode=mode, predictions={
                'encodes': output[0]
            })

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = self.gpu_memory_fraction
        config.log_device_placement = False
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

        return Estimator(model_fn=model_fn, config=RunConfig(session_config=config),
                         params={'batch_size': self.batch_size})

    def predict_from_queue(self):
        prediction = self.estimator.predict(input_fn=self.queue_predict_input_fn, yield_single_examples=False)
        for i in prediction:
            self.output_queue.put(i)

    def encode(self, sentence):
        self.sentence_len = len(sentence)
        self.input_queue.put(sentence)
        prediction = self.output_queue.get()['encodes']
        return prediction

    def queue_predict_input_fn(self):

        return (tf.data.Dataset.from_generator(
            self.generate_from_queue,
            output_types={'unique_ids': tf.int32,
                          'input_ids': tf.int32,
                          'input_mask': tf.int32,
                          'input_type_ids': tf.int32},
            output_shapes={
                'unique_ids': (self.sentence_len,),
                'input_ids': (None, self.max_seq_length),
                'input_mask': (None, self.max_seq_length),
                'input_type_ids': (None, self.max_seq_length)}).prefetch(10))

    def generate_from_queue(self):
        while True:
            features = list(self.convert_examples_to_features(seq_length=self.max_seq_length, tokenizer=self.tokenizer))
            yield {
                'unique_ids': [f.unique_id for f in features],
                'input_ids': [f.input_ids for f in features],
                'input_mask': [f.input_mask for f in features],
                'input_type_ids': [f.input_type_ids for f in features]
            }

    def input_fn_builder(self, features, seq_length):
        """Creates an `input_fn` closure to be passed to Estimator."""

        all_unique_ids = []
        all_input_ids = []
        all_input_mask = []
        all_input_type_ids = []

        for feature in features:
            all_unique_ids.append(feature.unique_id)
            all_input_ids.append(feature.input_ids)
            all_input_mask.append(feature.input_mask)
            all_input_type_ids.append(feature.input_type_ids)

        def input_fn(params):
            """The actual input function."""
            batch_size = params["batch_size"]

            num_examples = len(features)

            # This is for demo purposes and does NOT scale to large data sets. We do
            # not use Dataset.from_generator() because that uses tf.py_func which is
            # not TPU compatible. The right way to load data is with TFRecordReader.
            d = tf.data.Dataset.from_tensor_slices({
                "unique_ids":
                    tf.constant(all_unique_ids, shape=[num_examples], dtype=tf.int32),
                "input_ids":
                    tf.constant(
                        all_input_ids, shape=[num_examples, seq_length],
                        dtype=tf.int32),
                "input_mask":
                    tf.constant(
                        all_input_mask,
                        shape=[num_examples, seq_length],
                        dtype=tf.int32),
                "input_type_ids":
                    tf.constant(
                        all_input_type_ids,
                        shape=[num_examples, seq_length],
                        dtype=tf.int32),
            })

            d = d.batch(batch_size=batch_size, drop_remainder=False)
            return d

        return input_fn

    def model_fn_builder(self, bert_config, init_checkpoint, layer_indexes):
        """Returns `model_fn` closure for TPUEstimator."""

        def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
            """The `model_fn` for TPUEstimator."""

            unique_ids = features["unique_ids"]
            input_ids = features["input_ids"]
            input_mask = features["input_mask"]
            input_type_ids = features["input_type_ids"]

            jit_scope = tf.contrib.compiler.jit.experimental_jit_scope

            with jit_scope():
                model = modeling.BertModel(
                    config=bert_config,
                    is_training=False,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    token_type_ids=input_type_ids)

                if mode != tf.estimator.ModeKeys.PREDICT:
                    raise ValueError("Only PREDICT modes are supported: %s" % (mode))

                tvars = tf.trainable_variables()

                (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                           init_checkpoint)

                tf.logging.info("**** Trainable Variables ****")
                for var in tvars:
                    init_string = ""
                    if var.name in initialized_variable_names:
                        init_string = ", *INIT_FROM_CKPT*"
                    tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                                    init_string)

                all_layers = model.get_all_encoder_layers()

                predictions = {
                    "unique_id": unique_ids,
                }

                for (i, layer_index) in enumerate(layer_indexes):
                    predictions["layer_output_%d" % i] = all_layers[layer_index]

                from tensorflow.python.estimator.model_fn import EstimatorSpec

                output_spec = EstimatorSpec(mode=mode, predictions=predictions)
                return output_spec

        return model_fn

    def convert_examples_to_features(self, seq_length, tokenizer):
        """Loads a data file into a list of `InputBatch`s."""

        features = []
        input_masks = []
        examples = self._to_example(self.input_queue.get())
        for (ex_index, example) in enumerate(examples):
            tokens_a = tokenizer.tokenize(example.text_a)

            # if the sentences's length is more than seq_length, only use sentence's left part
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids: 0     0   0   0  0     0 0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambiguously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens = []
            input_type_ids = []
            tokens.append("[CLS]")
            input_type_ids.append(0)
            for token in tokens_a:
                tokens.append(token)
                input_type_ids.append(0)
            tokens.append("[SEP]")
            input_type_ids.append(0)

            # Where "input_ids" are tokens's index in vocabulary
            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)
            input_masks.append(input_mask)
            # Zero-pad up to the sequence length.
            while len(input_ids) < seq_length:
                input_ids.append(0)
                input_mask.append(0)
                input_type_ids.append(0)

            assert len(input_ids) == seq_length
            assert len(input_mask) == seq_length
            assert len(input_type_ids) == seq_length

            if ex_index < 5:
                tf.logging.info("*** Example ***")
                tf.logging.info("unique_id: %s" % (example.unique_id))
                tf.logging.info("tokens: %s" % " ".join(
                    [tokenization.printable_text(x) for x in tokens]))
                tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                tf.logging.info(
                    "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))

            yield InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids)

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    @staticmethod
    def _to_example(sentences):
        import re
        """
        sentences to InputExample
        :param sentences: list of strings
        :return: list of InputExample
        """
        unique_id = 0
        for ss in sentences:
            line = tokenization.convert_to_unicode(ss)
            if not line:
                continue
            line = line.strip()
            text_a = None
            text_b = None
            m = re.match(r"^(.*) \|\|\| (.*)$", line)
            if m is None:
                text_a = line
            else:
                text_a = m.group(1)
                text_b = m.group(2)
            yield InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b)
            unique_id += 1


class Bert_WordPreprocess(object):
    _total_labels = ['0', '1']

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
            wtrain_x = np.load(config.PREDATA_DIC + '/bert_wtrain_x.npy')
            wtrain_y = np.load(config.PREDATA_DIC + '/bert_wtrain_y.npy')
            return wtrain_x, wtrain_y
        except Exception as e:
            logging.error(e)
            exit(1)

    # 加载测试数据
    def load_testdata(self):
        try:
            wtest_x = np.load(config.PREDATA_DIC + '/bert_wtest_x.npy')
            wtest_y = np.load(config.PREDATA_DIC + '/bert_wtest_y.npy')
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
        bert = BertVector()
        logging.info("word to vector")
        docvecs_list = list()
        for words in words_list:
            docvecs = []
            try:
                print(words)
                result = bert.encode(words)
                tuple(result)
            except:
                result = bert.encode('。')
                pass
            # for word in words:
            #     wordvecs = []

            docvecs.append(result)
            docvecs_list.extend(docvecs)
            # docvecs_list.append(docvecs)
            b = np.array(docvecs_list)
        return docvecs_list

    # 测试词转句向量
    def test_word2vec(self, words_list: list, bert):
        logging.info("word to vector")
        docvecs_list = list()
        for words in words_list:
            docvecs = []
            try:
                print(words)
                result = bert.encode(words)
                tuple(result)
            except:
                result = bert.encode('。')
                pass
            # for word in words:
            #     wordvecs = []

            docvecs.append(result)
            docvecs_list.extend(docvecs)
            # docvecs_list.append(docvecs)
            b = np.array(docvecs_list)
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

        for news in datas:
            words = []
            labels = []
            for sentence in news:
                if sentence[6:] != '':
                    labels.append(sentence[:1])
                    words.append(sentence[6:])
            labels_list.append(labels)
            words_list.append(words)

        return words_list, labels_list


bert_wpreprocess = Bert_WordPreprocess()

if __name__ == '__main__':
    a, b, c, d = bert_wpreprocess.deal_tagdata(os.listdir(config.TAG_DIC + '/sr/seq/'))
    np.save(config.PREDATA_DIC + '/bert_wtrain_x.npy', np.array(a))
    np.save(config.PREDATA_DIC + '/bert_wtrain_y.npy', np.array(b))
    np.save(config.PREDATA_DIC + '/bert_wtest_x.npy', np.array(c))
    np.save(config.PREDATA_DIC + '/bert_wtest_y.npy', np.array(d))
