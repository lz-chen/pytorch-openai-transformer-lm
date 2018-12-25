import os
import sys
import json
import time
from functools import partial
import numpy as np
# import tensorflow as tf
# from tensorflow.python.framework import function
from tqdm import tqdm
from typing import Tuple, List, Dict
from collections import namedtuple
from numpy import ndarray
from text_utils import TextEncoder

Dataset = namedtuple('Dataset', 'TrainInstance, ValInstance , TestInstance ')
TrainInstance = namedtuple('TrainInstance', 'first_four_sents, first_choice, second_choice, true_choice')
ValInstance = namedtuple('ValInstance', 'first_four_sents, first_choice, second_choice, true_choice')
TestInstance = namedtuple('TestInstance', 'first_four_sents, first_choice, second_choice, true_choice')

# cloze_test_val__spring2016\ -\ cloze_test_ALL_val.csv : 1871 records, split to train and valid set
# cloze_test_test__spring2016\ -\ cloze_test_ALL_test.csv : 1871 records, used as test set

def encode_dataset(*splits: Tuple[
    # the four lists are first_four_sentences (len=1497), first_choice(len=1497), second_choice(len=1497), true_choice(len=1497)
    Tuple[List[str], List[str], List[str], ndarray],  # each list of len 1497, train instances,
    Tuple[List, List, List, List],  # each list of len 374, val instances
    Tuple[List, List, List, List]  # each list of len 1871, test instances
], encoder: TextEncoder):
    encoded_splits = []
    for split in splits:  # loop over trainInstances, valInstances and testInstances
        fields = []
        for field in split: #  a field is one list of str (sentences) or int (true answers)
            if isinstance(field[0], str): # check first element in field to see if str
                # each str element in the field list is encoded as a list of int, hence field becomes List[List[int]]
                field = encoder.encode(field)  # only encode sentences, not encoding true answers (type int choice: {0,1})
            fields.append(field)
        encoded_splits.append(fields)
    return encoded_splits


def stsb_label_encoding(labels, nclass=6):
    """
    Label encoding from Tree LSTM paper (Tai, Socher, Manning)
    """
    Y = np.zeros((len(labels), nclass)).astype(np.float32)
    for j, y in enumerate(labels):
        for i in range(nclass):
            if i == np.floor(y) + 1:
                Y[j, i] = y - np.floor(y)
            if i == np.floor(y):
                Y[j, i] = np.floor(y) - y + 1
    return Y


def np_softmax(x, t=1):
    x = x / t
    x = x - np.max(x, axis=-1, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=-1, keepdims=True)


def make_path(f):
    d = os.path.dirname(f)
    if d and not os.path.exists(d):
        os.makedirs(d)
    return f


def _identity_init(shape, dtype, partition_info, scale):
    n = shape[-1]
    w = np.eye(n) * scale
    if len([s for s in shape if s != 1]) == 2:
        w = w.reshape(shape)
    return w.astype(np.float32)


def identity_init(scale=1.0):
    return partial(_identity_init, scale=scale)


def _np_init(shape, dtype, partition_info, w):
    return w


def np_init(w):
    return partial(_np_init, w=w)


class ResultLogger(object):
    def __init__(self, path, *args, **kwargs):
        if 'time' not in kwargs:
            kwargs['time'] = time.time()
        self.f_log = open(make_path(path), 'w')
        self.f_log.write(json.dumps(kwargs) + '\n')

    def log(self, **kwargs):
        if 'time' not in kwargs:
            kwargs['time'] = time.time()
        self.f_log.write(json.dumps(kwargs) + '\n')
        self.f_log.flush()

    def close(self):
        self.f_log.close()


def flatten(outer):
    return [el for inner in outer for el in inner]


def remove_none(l):
    return [e for e in l if e is not None]


def iter_data(*datas, n_batch=128, truncate=False, verbose=False, max_batches=float("inf")):
    n = len(datas[0])
    if truncate:
        n = (n // n_batch) * n_batch
    n = min(n, max_batches * n_batch)
    n_batches = 0
    if verbose:
        f = sys.stderr
    else:
        f = open(os.devnull, 'w')
    for i in tqdm(range(0, n, n_batch), total=n // n_batch, file=f, ncols=80, leave=False):
        if n_batches >= max_batches: raise StopIteration
        if len(datas) == 1:
            yield datas[0][i:i + n_batch]
        else:
            yield (d[i:i + n_batch] for d in datas)
        n_batches += 1
