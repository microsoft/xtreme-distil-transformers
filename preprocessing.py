"""
Author: Subho Mukherjee (submukhe@microsoft.com)
Code for XtremeDistil for distilling massive multi-lingual models.
"""

from collections import defaultdict
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer

import csv
import logging
import numpy as np
import os
import six
import tensorflow as tf


logger = logging.getLogger('xtremedistil')

#set random seeds
GLOBAL_SEED = int(os.getenv("PYTHONHASHSEED"))

def convert_to_unicode(text):
  """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return text.decode("utf-8", "ignore")
    elif isinstance(text, unicode):
      return text
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")


def sequence_process(tokenizer, words, labels, special_tokens, MAX_SEQUENCE_LENGTH):

    trimmed_tokens = []
    trimmed_labels = []

    prev = []
    for i in range(1, len(words)+1):
        cur = tokenizer.tokenize(' '.join(words[:i]))
        diff = cur[len(prev):]
        trimmed_tokens.extend(diff)
        for j,_ in enumerate(diff):
            if j==0:
                trimmed_labels.append(labels[i-1])
            else:
                trimmed_labels.append("X")
        prev = cur

    try:
        assert len(trimmed_tokens) == len(trimmed_labels)
    except AssertionError:
        logger.error ("Dimension mismatch {} {}".format(len(trimmed_tokens), len(trimmed_labels)))
        raise

    if len(trimmed_tokens) > MAX_SEQUENCE_LENGTH - 2:
        trimmed_tokens = trimmed_tokens[:MAX_SEQUENCE_LENGTH - 2]
        trimmed_labels = trimmed_labels[:MAX_SEQUENCE_LENGTH - 2]
        trimmed_tokens.insert(0, special_tokens["bos_token"])
        trimmed_tokens.extend([special_tokens["eos_token"]])
        trimmed_labels.insert(0, special_tokens["bos_token"])
        trimmed_labels.extend([special_tokens["eos_token"]])
    else:
        diff = MAX_SEQUENCE_LENGTH  - 2 - len(trimmed_tokens)
        trimmed_tokens = [special_tokens["bos_token"]] + trimmed_tokens + [special_tokens["eos_token"]] + [special_tokens["pad_token"]]*diff
        trimmed_labels = [special_tokens["bos_token"]] + trimmed_labels + [special_tokens["eos_token"]] + [special_tokens["pad_token"]]*diff

    try:
        assert len(trimmed_tokens) == MAX_SEQUENCE_LENGTH
        assert len(trimmed_labels) == MAX_SEQUENCE_LENGTH
    except AssertionError:
        logger.error ("Dimension mismatch {} {}".format(len(trimmed_tokens), len(trimmed_labels)))
        raise

    return(trimmed_tokens, trimmed_labels)


def get_labels(label_file, special_tokens=None):
    label_list = []
    with open(label_file) as f:
        for line in f:
            if len(line.strip()) == 0:
                continue
            line = line.strip()
            label_list.append(line)
    if special_tokens is not None:
        label_list.extend([special_tokens["pad_token"], "X", special_tokens["eos_token"], special_tokens["bos_token"]])
    return label_list


def generate_sequence_data(MAX_SEQUENCE_LENGTH, input_file, tokenizer, label_list=None, unlabeled=False, special_tokens=None, do_pairwise=False, do_NER=False):
    
    X1 = []
    X2 = []
    y = []

    label_count = defaultdict(int)
    with tf.io.gfile.GFile(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=None)
      for line in reader:
        if len(line) == 0:
          continue
        x1 = convert_to_unicode(line[0])
        if do_pairwise:
          X2.append(convert_to_unicode(line[1]))
        if do_NER:
            if not unlabeled:
                label = convert_to_unicode(line[1]).split(" ")
            else:
                label = ['O']*len(x1.split(" "))
            x1, label = sequence_process(tokenizer, x1.split(" "), label, special_tokens, MAX_SEQUENCE_LENGTH)
            x1 = " ".join(x1)
            if not unlabeled:
                label = [label_list.index(v) for v in label]
            else:
                label = np.zeros(MAX_SEQUENCE_LENGTH)
        else:
            if not unlabeled:
                if do_pairwise:
                    label = int(convert_to_unicode(line[2]))
                else:
                    label = int(convert_to_unicode(line[1]))
            else:
                label = -1
        X1.append(x1)
        y.append(label)
        if not do_NER:
            label_count[label] += 1
        else:
            for v in label:
                label_count[v] += 1
    
    if do_NER:
        X = {}
        X["input_ids"] = np.array([tokenizer.convert_tokens_to_ids(val.split(" ")) for val in X1])
        #set attention mask
        X["attention_mask"] = np.ones((len(X["input_ids"]), MAX_SEQUENCE_LENGTH))
        pad_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        X["attention_mask"][X["input_ids"]==pad_id] = 0
    else:
        if do_pairwise:
          X =  tokenizer(X1, X2, padding=True, truncation=True, max_length = MAX_SEQUENCE_LENGTH)
        else:
          X =  tokenizer(X1, padding=True, truncation=True, max_length = MAX_SEQUENCE_LENGTH)

    for key in label_count.keys():
        logger.info ("Count of instances with label {} is {}".format(key, label_count[key]))

    if "token_type_ids" not in X:
        token_type_ids = np.zeros((len(X["input_ids"]), MAX_SEQUENCE_LENGTH))
    else:
        token_type_ids = np.array(X["token_type_ids"])

    X["input_ids"], token_type_ids, X["attention_mask"], y = shuffle(X["input_ids"], token_type_ids, X["attention_mask"], y, random_state=GLOBAL_SEED)

    return {"input_ids": np.array(X["input_ids"]), "token_type_ids": token_type_ids, "attention_mask": np.array(X["attention_mask"])}, np.array(y)
