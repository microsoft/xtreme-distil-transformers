"""
Author: Subho Mukherjee (submukhe@microsoft.com)
Code for XtremeDistil for distilling massive multi-lingual models.
"""

from evaluation import ner_evaluate
from huggingface_utils import MODELS, get_special_tokens_from_teacher
from preprocessing import generate_sequence_data, get_labels
from tensorflow.keras.models import Model
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from transformers import *

import argparse
import json
import logging
import models
import numpy as np
import os
import random
import sys
import tensorflow as tf

#logging
logger = logging.getLogger('xtremedistil')
logging.basicConfig(level = logging.INFO)

#set random seeds
GLOBAL_SEED = int(os.getenv("PYTHONHASHSEED"))
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
tf.random.set_seed(GLOBAL_SEED)
logger.info ("Global seed {}".format(GLOBAL_SEED))

if __name__ == '__main__':

	# construct the argument parse and parse the arguments
	parser = argparse.ArgumentParser()

	#required arguments
	parser.add_argument("--batch_size", nargs="?", type=int, default=128, help="train batch size")
	parser.add_argument("--model_dir", required=True, help="path of model directory")
	parser.add_argument("--ft_epochs", nargs="?", type=int, default=100, help="epochs for fine-tuning")
	parser.add_argument("--patience", nargs="?", type=int, default=5, help="number of iterations for early stopping.")

	#mixed precision
	parser.add_argument("--opt_policy", nargs="?", default=False, help="mixed precision policy")

	args = vars(parser.parse_args())
	logger.info(args)

	logger.info ("Directory of script ".format(os.path.dirname(os.path.abspath(__file__))))

	#load xtreme distil config
	distil_args = json.load(open(os.path.join(args["model_dir"], "xtremedistil-config.json"), 'r'))
	label_list = distil_args["label_list"]

	#get pre-trained model, tokenizer and config
	for indx, model in enumerate(MODELS):
		if model[0].__name__ == distil_args["pt_teacher"]:
			TFModelTeacher, Tokenizer, TeacherConfig = MODELS[indx]

	#get pre-trained tokenizer and special tokens
	pt_tokenizer = Tokenizer.from_pretrained(distil_args["pt_teacher_checkpoint"])
	special_tokens = get_special_tokens_from_teacher(Tokenizer, pt_tokenizer)

	#generate sequence data for fine-tuning pre-trained teacher
	X_train, y_train = generate_sequence_data(distil_args["seq_len"], os.path.join(distil_args["task"], "train.tsv"), pt_tokenizer, label_list=label_list, special_tokens=special_tokens, do_pairwise=distil_args["do_pairwise"], do_NER=distil_args["do_NER"])
	logger.info("X Shape {}".format(X_train["input_ids"].shape))

	X_test, y_test = generate_sequence_data(distil_args["seq_len"], os.path.join(distil_args["task"], "test.tsv"), pt_tokenizer, label_list=label_list, special_tokens=special_tokens, do_pairwise=distil_args["do_pairwise"], do_NER=distil_args["do_NER"])
	logger.info("X Shape {}".format(X_test["input_ids"].shape))

	#initialize word embedding
	word_emb = None
	if distil_args["compress_word_embedding"]:
		if distil_args["freeze_word_embedding"]:
			word_emb = np.load(open(os.path.join(args["model_dir"], "word_embedding.npy"), "rb"))
		else:
			word_emb = np.zeros((pt_tokenizer.vocab_size, distil_args["hidden_size"]))

	strategy = tf.distribute.MirroredStrategy()
	if args["opt_policy"]:
		policy = mixed_precision.Policy(args["opt_policy"])
		mixed_precision.set_policy(policy)
	gpus = strategy.num_replicas_in_sync
	logger.info('Number of devices: {}'.format(gpus))

	with strategy.scope():
		model = models.construct_transformer_student_model(distil_args,  stage=2, word_emb=word_emb)
		model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")])
		logger.info (model.summary())
		model.load_weights(os.path.join(args["model_dir"], "xtremedistil.h5"))

	model.fit(X_train, y_train, batch_size=args["batch_size"]*gpus, verbose=2, shuffle=True, epochs=args["ft_epochs"], callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=args["patience"], restore_best_weights=True)], validation_split=0.1)
	model.save_weights(os.path.join(args["model_dir"], "xtremedistil-ft.h5"))

	if distil_args["do_NER"]:
		cur_eval = ner_evaluate(model, X_test, y_test, label_list, special_tokens, distil_args["seq_len"], batch_size=args["batch_size"]*gpus)
	else:
		y_pred = np.argmax(model.predict(X_test, batch_size=args["batch_size"]*gpus), axis=-1)
		cur_eval = (y_pred == y_test).sum() / len(y_test)
	logger.info ("Test score {}".format(cur_eval))

