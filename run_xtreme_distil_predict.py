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
	parser.add_argument("--pred_file", nargs="?", help="file for prediction")
	parser.add_argument("--batch_size", nargs="?", type=int, default=256, help="predict batch size")
	parser.add_argument("--model_dir", required=True, help="path of model directory")
	parser.add_argument("--do_eval", action="store_true", default=False, help="whether to evaluate model performance on test data")
	parser.add_argument("--do_predict", action="store_true", default=False, help="whether to make model predictions")

	#mixed precision
	parser.add_argument("--opt_policy", nargs="?", default="mixed_float16", help="mixed precision policy")

	args = vars(parser.parse_args())
	logger.info(args)

	logger.info ("Directory of script ".format(os.path.dirname(os.path.abspath(__file__))))

	if not args["do_eval"] and not args["do_predict"]:
		logger.info ("Select one of do_eval or do_predict flags")
		sys.exit(1)

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
	if args["do_predict"]:
		X, _ = generate_sequence_data(distil_args["seq_len"], args["pred_file"], pt_tokenizer, label_list=label_list, unlabeled=True, special_tokens=special_tokens, do_pairwise=distil_args["do_pairwise"], do_NER=distil_args["do_NER"])
		logger.info("X Shape {}".format(X["input_ids"].shape))

	if args["do_eval"]:
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
	policy = mixed_precision.Policy(args["opt_policy"])
	mixed_precision.set_policy(policy)
	gpus = strategy.num_replicas_in_sync
	logger.info('Number of devices: {}'.format(gpus))

	with strategy.scope():
		model = models.construct_transformer_student_model(distil_args,  stage=2, word_emb=word_emb)
		model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")])
		logger.info (model.summary())
		model.load_weights(os.path.join(args["model_dir"], "xtremedistil.h5"))

	if args["do_eval"]:
		if distil_args["do_NER"]:
			cur_eval = ner_evaluate(model, X_test, y_test, label_list, special_tokens, distil_args["seq_len"], batch_size=args["batch_size"]*gpus)
		else:
			y_pred = np.argmax(model.predict(X_test, batch_size=args["batch_size"]*gpus), axis=-1)
			cur_eval = (y_pred == y_test).sum() / len(y_test)
		logger.info ("Test score {}".format(cur_eval))


	if args["do_predict"]:		
		texts = [pt_tokenizer.convert_ids_to_tokens(seq, skip_special_tokens=True) for seq in X["input_ids"]]
		y = np.argmax(model.predict(X, batch_size=args["batch_size"]*gpus), axis=-1)
		if distil_args["do_NER"]:
			y = [[label_list[elem] for elem in seq] for seq in y]
		with open(os.path.join(args["model_dir"], "predictions.txt"), "w") as fw:
			for i in range(len(y)):
				if distil_args["do_NER"]:
					fw.write(" ".join(texts[i]) + "\t" + " ".join(y[i][1:len(texts[i])+1]) + "\n")
				else:
					fw.write(texts[i] + "\t" + str(y) + "\n")



