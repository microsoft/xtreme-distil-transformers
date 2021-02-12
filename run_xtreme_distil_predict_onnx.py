"""
Author: Subho Mukherjee (submukhe@microsoft.com)
Code for XtremeDistil for distilling massive multi-lingual models.
"""

import os
import psutil

# ATTENTION: these environment variables must be set before importing onnxruntime.
os.environ["OMP_NUM_THREADS"] = str(psutil.cpu_count(logical=True))
os.environ["OMP_WAIT_POLICY"] = 'ACTIVE'

from evaluation import ner_evaluate
from huggingface_utils import MODELS, get_special_tokens_from_teacher
from keras2onnx.proto import keras
from onnxruntime_tools import optimizer
from preprocessing import generate_sequence_data, get_labels
from tensorflow.keras.models import Model
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from transformers import *

import argparse
import conlleval
import json
import keras2onnx
import logging
import models
import numpy as np
import onnxruntime
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


def ner_evaluate(y_pred, X_test, y_test, labels, special_tokens, MAX_SEQUENCE_LENGTH):

    pred_tags_all = []
    true_tags_all = []
    for i, seq in enumerate(y_pred):
        for j in range(MAX_SEQUENCE_LENGTH):
            indx = y_test[i][j]
            true_label = labels[indx]
            if special_tokens["pad_token"] in true_label or special_tokens["bos_token"] in true_label or special_tokens["eos_token"] in true_label:
                continue
            true_tags_all.append(true_label)
            indx = np.argmax(seq[j])
            pred_label = labels[indx]
            pred_tags_all.append(pred_label)
    prec, rec, f1 = conlleval.evaluate(true_tags_all, pred_tags_all, special_tokens, verbose=True)
    logger.info ("Test scores {} {} {}".format(prec, rec, f1))

    return np.mean(f1)

def pad_zeros(X, seq_len):
	result = np.zeros((len(X), seq_len))
	result[:X.shape[0], :X.shape[1]] = X
	return np.array(result, dtype=np.int32)

if __name__ == '__main__':

	# construct the argument parse and parse the arguments
	parser = argparse.ArgumentParser()

	#required arguments
	parser.add_argument("--pred_file", nargs="?", help="file for prediction")
	parser.add_argument("--model_dir", required=True, help="path of model directory")
	parser.add_argument("--do_eval", action="store_true", default=False, help="whether to evaluate model performance on test data")
	parser.add_argument("--do_predict", action="store_true", default=False, help="whether to make model predictions")

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
		X_test, y_test = generate_sequence_data(distil_args["seq_len"], os.path.join(distil_args["task"], "test_small.tsv"), pt_tokenizer, label_list=label_list, special_tokens=special_tokens, do_pairwise=distil_args["do_pairwise"], do_NER=distil_args["do_NER"])
		logger.info("X Shape {}".format(X_test["input_ids"].shape))

	#initialize word embedding
	word_emb = None
	if distil_args["compress_word_embedding"]:
		if distil_args["freeze_word_embedding"]:
			word_emb = np.load(open(os.path.join(args["model_dir"], "word_embedding.npy"), "rb"))
		else:
			word_emb = np.zeros((pt_tokenizer.vocab_size, distil_args["hidden_size"]))

	strategy = tf.distribute.MirroredStrategy()
	gpus = strategy.num_replicas_in_sync
	logger.info('Number of devices: {}'.format(gpus))

	with strategy.scope():
		model = models.construct_transformer_student_model(distil_args,  stage=2, word_emb=word_emb)
		model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")])
		logger.info (model.summary())
		model.load_weights(os.path.join(args["model_dir"], "xtremedistil.h5"))

	model.get_layer("student_2")._saved_model_inputs_spec = None

	# convert to onnx model
	output_model_path = os.path.join(args["model_dir"], "xtremedistil.onnx")
	onnx_model = keras2onnx.convert_keras(model, model.name)
	keras2onnx.save_model(onnx_model, output_model_path)

	optimized_model_path = os.path.join(args["model_dir"], "xtremedistil_opt.onnx")
	optimized_model = optimizer.optimize_model(output_model_path, model_type='bert_keras', num_heads=distil_args["num_attention_heads"], hidden_size=distil_args["hidden_size"])
	optimized_model.use_dynamic_axes()
	optimized_model.save_model_to_file(optimized_model_path)

	sess_options = onnxruntime.SessionOptions()
	# intra_op_num_threads=1 can be used to enable OpenMP in OnnxRuntime 1.2.0.
	# For OnnxRuntime 1.3.0 or later, this does not have effect unless you are using onnxruntime-gpu package.
	# sess_options.intra_op_num_threads=1

	# Providers is optional. Only needed when you use onnxruntime-gpu for CPU inference.
	session = onnxruntime.InferenceSession(optimized_model_path, sess_options, providers=['CPUExecutionProvider'])


	if args["do_eval"]:
		#pad input_ids to max_length with 0's
		X_test["input_ids"] = pad_zeros(X_test["input_ids"], distil_args["seq_len"])
		X_test["token_type_ids"] = pad_zeros(X_test["token_type_ids"], distil_args["seq_len"])
		X_test["attention_mask"] = pad_zeros(X_test["attention_mask"], distil_args["seq_len"])

		batch_size = 1
		inputs_onnx = {k_: np.repeat(v_, batch_size, axis=0) for k_, v_ in X_test.items()}

		results = np.array(session.run(None, inputs_onnx)[0])

		if distil_args["do_NER"]:
			cur_eval = ner_evaluate(results, X_test, y_test, label_list, special_tokens, distil_args["seq_len"])
		else:
			y_pred = np.argmax(results, axis=-1)
			cur_eval = (y_pred == y_test).sum() / len(y_test)
		logger.info ("Test score {}".format(cur_eval))


	if args["do_predict"]:		
		texts = [pt_tokenizer.convert_ids_to_tokens(seq, skip_special_tokens=True) for seq in X["input_ids"]]
		#pad input_ids to max_length with 0's
		X["input_ids"] = pad_zeros(X["input_ids"], distil_args["seq_len"])
		X["token_type_ids"] = pad_zeros(X["token_type_ids"], distil_args["seq_len"])
		X["attention_mask"] = pad_zeros(X["attention_mask"], distil_args["seq_len"])

		inputs_onnx = {k_: np.repeat(v_, batch_size, axis=0) for k_, v_ in X.items()}

		results = session.run(None, inputs_onnx)[0]

		y = np.argmax(results, axis=-1)

		if distil_args["do_NER"]:
			y = [[label_list[elem] for elem in seq] for seq in y]

		with open(os.path.join(args["model_dir"], "predictions.txt"), "w") as fw:
			for i in range(len(y)):
				if distil_args["do_NER"]:
					fw.write(" ".join(texts[i]) + "\t" + " ".join(y[i][1:len(texts[i])+1]) + "\n")
				else:
					fw.write(" ".join(texts[i]) + "\t" + str(y[i]) + "\n")



