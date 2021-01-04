"""
Author: Subho Mukherjee (submukhe@microsoft.com)
Code for XtremeDistil for distilling massive multi-lingual models.
"""

from preprocessing import convert_to_unicode
from tensorflow.keras.layers import Embedding, Input, LSTM, Bidirectional, Dropout, Dense, Lambda
from tensorflow.keras.models import Model
from transformers import BertConfig, TFBertModel, TFBertForSequenceClassification
from huggingface_utils import get_output_state_indices

import csv
import logging
import numpy as np
import os
import random
import tensorflow as tf

logger = logging.getLogger('xtremedistil')

# set seeds for random number generator for reproducibility
GLOBAL_SEED = int(os.getenv("PYTHONHASHSEED"))
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
tf.random.set_seed(GLOBAL_SEED)

def gelu(x):
  """Gaussian Error Linear Unit.

  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    x: float Tensor to perform activation.

  Returns:
    `x` with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + tf.tanh(
      (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
  return x * cdf


def get_H_t(X):
    ans = X[-1, :, :, :]
    return ans


def construct_transformer_teacher_model(args, TFModelTeacher, teacher_config):

    encoder = TFModelTeacher.from_pretrained(args["pt_teacher_checkpoint"], config=teacher_config, from_pt=True, name="tf_model")
    input_ids = Input(shape=(args["seq_len"],), dtype=tf.int32, name="input_ids")
    attention_mask = Input(shape=(args["seq_len"],), dtype=tf.int32, name="attention_mask")
    token_type_ids = Input(shape=(args["seq_len"],), dtype=tf.int32, name="token_type_ids")
    encode = encoder(input_ids, token_type_ids=token_type_ids,  attention_mask=attention_mask)

    output_hidden_state_indx, output_attention_state_indx =  get_output_state_indices(TFModelTeacher)

    classes = len(args["label_list"])

    embedding = []
    if args["distil_multi_hidden_states"]:
        if args["do_NER"]:
            #add hidden states
            for i in range(1, args["num_hidden_layers"]+2):
                embedding.append(encode[output_hidden_state_indx][-i])
        else:
            for i in range(1, args["num_hidden_layers"]+2):
                embedding.append(encode[output_hidden_state_indx][-i][:,0])
        if args["distil_attention"]:
            #add attention states
            for i in range(1, args["num_hidden_layers"]+1):
                embedding.append(encode[output_attention_state_indx][-i])
    else:
        if args["do_NER"]:
            embedding.append(encode[0])
        else:
            embedding.append(encode[0][:,0])

    logger.info(teacher_config)
    outputs = Dropout(teacher_config.hidden_dropout_prob)(embedding[0])
    outputs = Dense(classes, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=teacher_config.initializer_range), name="final_logits")(outputs)
    teacher_model = tf.keras.Model(inputs=[input_ids, token_type_ids, attention_mask], outputs=outputs)
    teacher_intermediate_layer = tf.keras.Model(inputs=teacher_model.input, outputs=embedding)
    return teacher_model, teacher_intermediate_layer


def construct_transformer_student_model(args, stage, word_emb=None):

    input_ids = Input(shape=(args["seq_len"],), dtype=tf.int32, name="input_ids")
    attention_mask = Input(shape=(args["seq_len"],), dtype=tf.int32, name="attention_mask")
    token_type_ids = Input(shape=(args["seq_len"],), dtype=tf.int32, name="token_type_ids")

    classes = len(args["label_list"])

    #construct student models for different stages
    if args["pt_student_checkpoint"]:
        student_config = BertConfig.from_pretrained(args["pt_student_checkpoint"], output_hidden_states=args["distil_multi_hidden_states"], output_attentions=args["distil_attention"])
        student_encoder = TFBertModel.from_pretrained(args["pt_student_checkpoint"], from_pt=True, config=student_config, name="student_{}".format(stage))
    else:
        student_config = BertConfig(num_hidden_layers=args["num_hidden_layers"], num_attention_heads=args["num_attention_heads"], hidden_size=args["hidden_size"], output_hidden_states=args["distil_multi_hidden_states"], output_attentions=args["distil_attention"])
        student_encoder = TFBertModel(config=student_config, name="student_{}".format(stage))

    logger.info (student_config)

    if word_emb is not None:
        if args["freeze_word_embedding"]:
            student_encoder.set_input_embeddings(word_emb)
        else:
            student_encoder.set_input_embeddings(tf.Variable(word_emb))

    encode = student_encoder(input_ids, token_type_ids=token_type_ids,  attention_mask=attention_mask)

    output_hidden_state_indx, output_attention_state_indx =  get_output_state_indices(TFBertModel)

    embedding = []
    if args["distil_multi_hidden_states"]:
        if args["do_NER"]:
            #add hidden states
            for i in range(1, args["num_hidden_layers"]+2):
                embedding.append(encode[output_hidden_state_indx][-i])
        else:
            for i in range(1, args["num_hidden_layers"]+2):
                embedding.append(encode[output_hidden_state_indx][-i][:,0])
        if args["distil_attention"]:
            #add attention states
            for i in range(1, args["num_hidden_layers"]+1):
                embedding.append(encode[output_attention_state_indx][-i])
    else:
        if args["do_NER"]:
            embedding.append(encode[0])
        else:
            embedding.append(encode[0][:,0])

    if args["teacher_hidden_size"] > args["hidden_size"]:
        if args["distil_multi_hidden_states"]:
            dense = []
            dropout = []
            for i in range(args["num_hidden_layers"]+1):
                if i == 0:
                    dense.append(Dense(args["teacher_hidden_size"], name="dense_{}".format(stage), kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=student_config.initializer_range)))
                else:
                    dense.append(Dense(args["teacher_hidden_size"], name="dense_{}_{}".format(stage, i), kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=student_config.initializer_range)))
                dropout.append(Dropout(student_config.hidden_dropout_prob, name="dropout_{}".format(i)))

            embedding = [dense[i](dropout[i](embedding[i])) if (i < args["num_hidden_layers"]+1) else embedding[i] for i in range(len(embedding))]
        else:
            embedding = [Dense(args["teacher_hidden_size"], name="dense_{}".format(stage), kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=student_config.initializer_range))(Dropout(student_config.hidden_dropout_prob)(embedding[0]))]

    if stage == 1:
        return Model(inputs=[input_ids, token_type_ids, attention_mask], outputs=embedding)
    else:
        outputs = Dropout(student_config.hidden_dropout_prob)(embedding[0])
        logits = Dense(classes, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=student_config.initializer_range))(outputs)
        return Model(inputs=[input_ids, token_type_ids, attention_mask], outputs=logits)


def compile_model(model, args, strategy, stage):

    #construct student models for different stages
    with strategy.scope():
        if stage == 1 or stage == 2:
            if args["distil_attention"] and args["distil_multi_hidden_states"]:
                loss = ['mse'] * (2 * args["num_hidden_layers"] + 1)
            elif args["distil_multi_hidden_states"]:
                loss = ['mse'] * (args["num_hidden_layers"] + 1)
            else:
                loss = ['mse']
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08), loss=loss)
        else:
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")])

    return model