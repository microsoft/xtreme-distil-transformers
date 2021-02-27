"""
Author: Subho Mukherjee (submukhe@microsoft.com)
Code for XtremeDistil for distilling massive multi-lingual models.
"""

from evaluation import ner_evaluate
from huggingface_utils import MODELS, get_special_tokens_from_teacher, get_word_embedding, get_output_state_indices
from preprocessing import generate_sequence_data, get_labels
from sklearn.utils import shuffle
from tensorflow.keras.layers import Input, Dropout, Dense, Lambda
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras.models import Model
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
	parser.add_argument("--task", required=True, help="name of the task")
	parser.add_argument("--model_dir", required=True, help="path of model directory")
	parser.add_argument("--seq_len", required=True, type=int, help="sequence length")

	#task
	parser.add_argument("--multilingual", action="store_true", default=False, help="whether to perform multilingual task")
	parser.add_argument("--do_NER", action="store_true", default=False, help="whether to perform NER")
	parser.add_argument("--do_pairwise", action="store_true", default=False, help="whether to perform pairwise instance classification tasks")

	#model parameters (optional)
	parser.add_argument("--pt_model", nargs="?", default="TFBertModel",help="Pre-trained model")
	parser.add_argument("--pt_model_checkpoint", nargs="?", default="bert-base-multilingual-cased", help="model checkpoint to load to pre-trained weights")

	#mixed precision
	parser.add_argument("--opt_policy", nargs="?", default=False, help="mixed precision policy")

	#batch sizes and epochs (optional)
	parser.add_argument("--batch_size", nargs="?", type=int, default=128, help="batch size for distillation")
	parser.add_argument("--ft_epochs", nargs="?", type=int, default=100, help="epochs for fine-tuning")
	parser.add_argument("--patience", nargs="?", type=int, default=5, help="number of iterations for early stopping.")

	args = vars(parser.parse_args())
	logger.info(args)
	logger.info ("Directory of script ".format(os.path.dirname(os.path.abspath(__file__))))

	#get pre-trained model, tokenizer and config
	for indx, model in enumerate(MODELS):
		if model[0].__name__ == args["pt_model"]:
			TFModel, Tokenizer, Config = MODELS[indx]

	if args["multilingual"]:
		Tokenizer = XLMRobertaTokenizer

	#get pre-trained tokenizer and special tokens
	pt_tokenizer = Tokenizer.from_pretrained(args["pt_model_checkpoint"])

	special_tokens = get_special_tokens_from_teacher(Tokenizer, pt_tokenizer)

	model_config = Config.from_pretrained(args["pt_model_checkpoint"])

	#get labels for NER
	label_list=None
	if args["do_NER"]:
		label_list = get_labels(os.path.join(args["task"], "labels.tsv"), special_tokens)

	#generate sequence data for fine-tuning pre-trained model
	X_train, y_train = generate_sequence_data(args["seq_len"], os.path.join(args["task"], "train.tsv"), pt_tokenizer, label_list=label_list, special_tokens=special_tokens, do_pairwise=args["do_pairwise"], do_NER=args["do_NER"])

	X_test, y_test = generate_sequence_data(args["seq_len"], os.path.join(args["task"], "test.tsv"), pt_tokenizer, label_list=label_list, special_tokens=special_tokens, do_pairwise=args["do_pairwise"], do_NER=args["do_NER"])

	X_dev, y_dev = generate_sequence_data(args["seq_len"], os.path.join(args["task"], "dev.tsv"), pt_tokenizer, label_list=label_list, special_tokens=special_tokens, do_pairwise=args["do_pairwise"], do_NER=args["do_NER"])

	if not args["do_NER"]:
		label_list = [str(elem) for elem in set(y_train)]

	#logging data shapes
	logger.info("X Train Shape {} {}".format(X_train["input_ids"].shape, y_train.shape))
	logger.info("X Dev Shape {} {}".format(X_dev["input_ids"].shape, y_dev.shape))
	logger.info("X Test Shape {} {}".format(X_test["input_ids"].shape, y_test.shape))

	for i in range(3):
		logger.info ("Example {}".format(i))
		logger.info ("Input sequence: {}".format(pt_tokenizer.convert_ids_to_tokens(X_train["input_ids"][i])))
		logger.info ("Input ids: {}".format(X_train["input_ids"][i]))
		logger.info ("Attention mask: {}".format(X_train["attention_mask"][i]))
		logger.info ("Token type ids: {}".format(X_train["token_type_ids"][i]))
		if args["do_NER"]:
			logger.info ("Label sequence: {}".format(' '.join([label_list[v] for v in y_train[i]])))
		else:
			logger.info ("Label: {}".format(y_train[i]))

	#fine-tune pre-trained model
	strategy = tf.distribute.MirroredStrategy()

	gpus = strategy.num_replicas_in_sync
	logger.info('Number of devices: {}'.format(gpus))

	#set optimization policy
	if args["opt_policy"]:
		policy = mixed_precision.Policy(args["opt_policy"])
		mixed_precision.set_policy(policy)

	with strategy.scope():
		encoder = TFModel.from_pretrained(args["pt_model_checkpoint"], config=model_config, name="tf_model", from_pt=True)
		input_ids = Input(shape=(args["seq_len"],), dtype=tf.int32, name="input_ids")
		attention_mask = Input(shape=(args["seq_len"],), dtype=tf.int32, name="attention_mask")
		token_type_ids = Input(shape=(args["seq_len"],), dtype=tf.int32, name="token_type_ids")
		encode = encoder(input_ids, token_type_ids=token_type_ids,  attention_mask=attention_mask)
		if args["do_NER"]:
			embedding = encode[0]
		else:
			embedding = encode[0][:,0]
		embedding = Dropout(model_config.hidden_dropout_prob)(embedding)
		logits = Dense(len(label_list), kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=model_config.initializer_range))(embedding)

		model = tf.keras.Model(inputs=[input_ids, token_type_ids, attention_mask], outputs=logits)
		model = models.compile_model(model, args, strategy, stage=3)
		logger.info (model.summary())

	model.fit(x=X_train, y=y_train, batch_size=args["batch_size"]*gpus, shuffle=True, epochs=args["ft_epochs"], callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=args["patience"], restore_best_weights=True)], validation_data=(X_dev, y_dev))

	#evaluate fine-tuned model
	if args["do_NER"]:
		ner_evaluate(model, X_test, y_test, label_list, special_tokens, args["seq_len"], batch_size=args["batch_size"]*gpus)
	else:
		logger.info("Model accuracy {}".format(model.evaluate(X_test, y_test, batch_size=args["batch_size"]*gpus)))
