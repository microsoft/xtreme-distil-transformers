"""
Author: Subho Mukherjee (submukhe@microsoft.com)
Code for XtremeDistil for distilling massive multi-lingual models.
"""

from evaluation import ner_evaluate
from huggingface_utils import MODELS, get_special_tokens_from_teacher, get_word_embedding, get_output_state_indices
from preprocessing import generate_sequence_data, get_labels
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
	parser.add_argument("--transfer_file", required=True, help="transfer data for distillation")
	parser.add_argument("--teacher_model_dir", required=True, help="path of model directory")

	#task
	parser.add_argument("--do_NER", action="store_true", default=False, help="whether to perform NER")
	parser.add_argument("--do_pairwise", action="store_true", default=False, help="whether to perform pairwise instance classification tasks")

	#transformer student model parameters (optional)
	parser.add_argument("--hidden_size", nargs="?", type=int, default=384, help="hidden state dimension of the student model")
	parser.add_argument("--num_attention_heads", nargs="?", type=int, default=6, help="number of attention heads")
	parser.add_argument("--num_hidden_layers", nargs="?", type=int, default=6, help="number of layers in the student model")
	#optional student model checkpoint to load from
	parser.add_argument("--pt_student_checkpoint", nargs="?", default=False, help="student model checkpoint to initialize the distilled model with pre-trained weights.")

	#distillation features
	parser.add_argument("--distil_attention", action="store_true", default=False, help="whether to distil teacher attention")
	parser.add_argument("--distil_multi_hidden_states", action="store_true", default=False, help="whether to distil multiple hidden layers from teacher")
	parser.add_argument("--compress_word_embedding", action="store_true", default=False, help="whether to compress word embedding matrix")

	#teacher model parameters (optional)
	parser.add_argument("--pt_teacher", nargs="?", default="TFBertModel",help="Pre-trained teacher model to distil")
	parser.add_argument("--pt_teacher_checkpoint", nargs="?", default="bert-base-multilingual-cased", help="teacher model checkpoint to load to pre-trained weights")

	#mixed precision
	parser.add_argument("--opt_policy", nargs="?", default="mixed_float16", help="mixed precision policy")

	#batch sizes and epochs (optional)
	parser.add_argument("--student_batch_size", nargs="?", type=int, default=128, help="batch size for distillation")
	parser.add_argument("--teacher_batch_size", nargs="?", type=int, default=128, help="batch size for distillation")
	parser.add_argument("--ft_epochs", nargs="?", type=int, default=100, help="epochs for fine-tuning")
	parser.add_argument("--distil_epochs", nargs="?", type=int, default=500, help="epochs for distillation")
	parser.add_argument("--distil_chunk_size", nargs="?", type=int, default=100000, help="transfer data partition size (reduce if OOM)")
	parser.add_argument("--patience", nargs="?", type=int, default=5, help="number of iterations for early stopping.")

	args = vars(parser.parse_args())
	logger.info(args)
	logger.info ("Directory of script ".format(os.path.dirname(os.path.abspath(__file__))))

	#get pre-trained model, tokenizer and config
	for indx, model in enumerate(MODELS):
		if model[0].__name__ == args["pt_teacher"]:
			TFModelTeacher, Tokenizer, TeacherConfig = MODELS[indx]

	#get pre-trained tokenizer and special tokens
	pt_tokenizer = Tokenizer.from_pretrained(args["pt_teacher_checkpoint"])

	special_tokens = get_special_tokens_from_teacher(Tokenizer, pt_tokenizer)
	output_hidden_state_indx, output_attention_state_indx =  get_output_state_indices(TFModelTeacher)

	teacher_config = TeacherConfig.from_pretrained(args["pt_teacher_checkpoint"], output_hidden_states=args["distil_multi_hidden_states"], output_attentions=args["distil_attention"])

	if args["pt_student_checkpoint"]:
		student_config = BertConfig.from_pretrained(args["pt_student_checkpoint"], output_hidden_states=args["distil_multi_hidden_states"], output_attentions=args["distil_attention"])
		args["hidden_size"] = student_config.hidden_size
		args["num_hidden_layers"] = student_config.num_hidden_layers
		args["num_attention_heads"] = student_config.num_attention_heads

	if args["distil_attention"]:
		args["distil_multi_hidden_states"] = True

	args["teacher_hidden_size"] = teacher_config.hidden_size

	#get labels for NER
	label_list=None
	if args["do_NER"]:
		label_list = get_labels(os.path.join(args["task"], "labels.tsv"), special_tokens)

	#generate sequence data for fine-tuning pre-trained teacher
	X_train, y_train = generate_sequence_data(args["seq_len"], os.path.join(args["task"], "train.tsv"), pt_tokenizer, label_list=label_list, special_tokens=special_tokens, do_pairwise=args["do_pairwise"], do_NER=args["do_NER"])

	X_test, y_test = generate_sequence_data(args["seq_len"], os.path.join(args["task"], "test.tsv"), pt_tokenizer, label_list=label_list, special_tokens=special_tokens, do_pairwise=args["do_pairwise"], do_NER=args["do_NER"])

	X_dev, y_dev = generate_sequence_data(args["seq_len"], os.path.join(args["task"], "dev.tsv"), pt_tokenizer, label_list=label_list, special_tokens=special_tokens, do_pairwise=args["do_pairwise"], do_NER=args["do_NER"])

	X_unlabeled, _ = generate_sequence_data(args["seq_len"], args["transfer_file"], pt_tokenizer, label_list=label_list, unlabeled=True, special_tokens=special_tokens, do_pairwise=args["do_pairwise"], do_NER=args["do_NER"])

	if not args["do_NER"]:
		label_list = [str(elem) for elem in set(y_train)]

	args["label_list"] = label_list

	#logging teacher data shapes
	logger.info("X Train Shape {} {}".format(X_train["input_ids"].shape, y_train.shape))
	logger.info("X Dev Shape {} {}".format(X_dev["input_ids"].shape, y_dev.shape))
	logger.info("X Test Shape {} {}".format(X_test["input_ids"].shape, y_test.shape))
	logger.info("X Unlabeled Transfer Shape {}".format(X_unlabeled["input_ids"].shape))

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

	#fine-tune pre-trained teacher
	strategy = tf.distribute.MirroredStrategy()
	policy = mixed_precision.Policy(args["opt_policy"])
	mixed_precision.set_policy(policy)

	gpus = strategy.num_replicas_in_sync
	logger.info('Number of devices: {}'.format(gpus))

	with strategy.scope():
		teacher_model, teacher_intermediate_layer = models.construct_transformer_teacher_model(args, TFModelTeacher, teacher_config)
		logger.info (teacher_model.summary())
		teacher_model = models.compile_model(teacher_model, args, strategy, stage=3)

	model_file = os.path.join(args["teacher_model_dir"], "model-ft.h5")
	if os.path.exists(model_file):
		logger.info ("Loadings weights for fine-tuned model from {}".format(model_file))
		teacher_model.load_weights(model_file)
	else:
		teacher_model.fit(x=X_train, y=y_train, batch_size=args["teacher_batch_size"]*gpus, shuffle=True, epochs=args["ft_epochs"], callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=args["patience"], restore_best_weights=True)], validation_data=(X_dev, y_dev))
		teacher_model.save_weights(model_file)

	#evaluate fine-tuned teacher
	if args["do_NER"]:
		teacher_model.evaluate(X_test, y_test, batch_size=args["student_batch_size"]*gpus)
		ner_evaluate(teacher_model, X_test, y_test, label_list, special_tokens, args["seq_len"], batch_size=args["teacher_batch_size"]*gpus)
	else:
		logger.info("Teacher model accuracy {}".format(teacher_model.evaluate(X_test, y_test, batch_size=args["teacher_batch_size"]*gpus)))

	word_emb = None
	if args["compress_word_embedding"]:
		word_emb = get_word_embedding(teacher_model.get_layer("tf_model"), pt_tokenizer, args["hidden_size"])

	with strategy.scope():
		model_1 = models.construct_transformer_student_model(args, stage=1, word_emb=word_emb)
		model_1 = models.compile_model(model_1, args, strategy, stage=1)

		model_2 = models.construct_transformer_student_model(args, stage=2, word_emb=word_emb)
		model_2 = models.compile_model(model_2, args, strategy, stage=2)

	#get shared layers for student models
	shared_layers = set()
	for layer in model_1.layers:
		if len(layer.trainable_weights) > 0:
			shared_layers.add(layer.name.split("_")[0])
	#update parameters top down from the shared layers
	shared_layers = list(shared_layers)
	logger.info ("Shared layers {}".format(shared_layers))

	best_model = None
	best_eval = 0
	min_loss = np.inf
	min_ckpt = None

	for stage in range(1, 2*len(shared_layers)+4):

		logger.info ("*** Starting stage {}".format(stage))
		patience_counter = 0

		#stage = 1, optimize representation loss (transfer set) with end-to-end training
		#stage = 2, copy model from stage = 1, and optimize logit loss (transfer set) with all but last layer frozen
		#stage = [3, 4, .., 2+num_shared_layers], optimize logit loss (transfer set) with gradual unfreezing
		#stage == 3+num_shared_layers, optimize CE loss (labeled data) with all but last layer frozen
		#stage = [4+num_shared_layers, ...], optimize CE loss (labeled data)with gradual unfreezing

		if stage == 2:
			#copy weights from model_stage_1
			logger.info ("Copying weights from model stage 1")
			for layer in shared_layers:
				model_2.get_layer(layer+"_2").set_weights(model_1.get_layer(layer+"_1").get_weights())
				model_2.get_layer(layer+"_2").trainable = False
			model_2 = models.compile_model(model_2, args, strategy, stage=2)
			#resetting min loss
			min_loss = np.inf
		elif stage > 2 and stage < 3+len(shared_layers):
			logger.info ("Unfreezing layer {}".format(shared_layers[stage-3]))
			model_2.get_layer(shared_layers[stage-3]+"_2").trainable = True
			model_2 = models.compile_model(model_2, args, strategy, stage=2)
		elif stage == 3+len(shared_layers):
			for layer in shared_layers:
				model_2.get_layer(layer+"_2").trainable = False
			model_2 = models.compile_model(model_2, args, strategy, stage=3)
			#resetting min loss
			min_loss = np.inf
		elif stage > 3+len(shared_layers):
			logger.info ("Unfreezing layer {}".format(shared_layers[stage-4-len(shared_layers)]))
			model_2.get_layer(shared_layers[stage-4-len(shared_layers)]+"_2").trainable = True
			model_2 = models.compile_model(model_2, args, strategy, stage=3)

		start_teacher = 0

		while start_teacher < len(X_unlabeled["input_ids"]) and stage < 3+len(shared_layers):

			end_teacher = min(start_teacher + args["distil_chunk_size"], len(X_unlabeled["input_ids"]))
			logger.info ("Teacher indices from {} to {}".format(start_teacher, end_teacher))

			#get teacher logits
			input_data_chunk = {"input_ids": X_unlabeled["input_ids"][start_teacher:end_teacher], "attention_mask": X_unlabeled["attention_mask"][start_teacher:end_teacher], "token_type_ids": X_unlabeled["token_type_ids"][start_teacher:end_teacher]}

			if stage == 1:
				#get representation from intermediate teacher layer
				y_layer_teacher = teacher_intermediate_layer.predict(input_data_chunk, batch_size=args["teacher_batch_size"]*gpus)
			else:
				y_teacher = teacher_model.predict(input_data_chunk, batch_size=args["teacher_batch_size"]*gpus)

			model_file = os.path.join(args["model_dir"], "model-stage-{}-indx-{}.h5".format(stage, start_teacher))
			history_file = os.path.join(args["model_dir"], "model-stage-{}-indx-{}-history.json".format(stage, start_teacher))

			if stage == 1:
				if os.path.exists(model_file):
					logger.info ("Loadings weights for stage {} from {}".format(stage, model_file))
					model_1.load_weights(model_file)
					history = json.load(open(history_file, 'r'))
				else:
					logger.info (model_1.summary())
					model_history = model_1.fit(input_data_chunk, y_layer_teacher, shuffle=True, batch_size=args["student_batch_size"]*gpus, verbose=2, epochs=args["distil_epochs"], callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=args["patience"], restore_best_weights=True)], validation_split=0.1)
					history = model_history.history
					model_1.save_weights(model_file)
					json.dump(history, open(history_file, 'w'))
				val_loss = history['val_loss'][-1]
				if  val_loss < min_loss:
					min_loss = val_loss
					min_ckpt = model_1.get_weights()
					logger.info ("Checkpointing model weights with minimum validation loss {}".format(min_loss))
					patience_counter = 0
				else:
					patience_counter += 1
					logger.info ("Resetting model to best weights found so far corresponding to val_loss {}".format(min_loss))
					model_1.set_weights(min_ckpt)
					if patience_counter == args["patience"]:
						logger.info("Early stopping")
						break

			elif stage > 1 and stage < 3+len(shared_layers):
				if os.path.exists(model_file):
					logger.info ("Loadings weights for stage {} from {}".format(stage, model_file))
					model_2.load_weights(model_file)
					history = json.load(open(history_file, 'r'))
				else:
					logger.info (model_2.summary())
					model_history = model_2.fit(input_data_chunk, y_teacher, shuffle=True, batch_size=args["student_batch_size"]*gpus, verbose=2, epochs=args["distil_epochs"], callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=args["patience"], restore_best_weights=True)], validation_split=0.1)
					history = model_history.history
					model_2.save_weights(model_file)
					json.dump(history, open(history_file, 'w'))
				val_loss = history['val_loss'][-1]
				if  val_loss < min_loss:
					min_loss = val_loss
					min_ckpt = model_2.get_weights()
					logger.info ("Checkpointing model weights with minimum validation loss {}".format(min_loss))
					patience_counter = 0
				else:
					patience_counter += 1
					logger.info ("Resetting model to best weights found so far corresponding to val_loss {}".format(min_loss))
					model_2.set_weights(min_ckpt)
					if patience_counter == args["patience"]:
						logger.info("Early stopping")
						break

			start_teacher = end_teacher

		if stage > 1 and stage < 3+len(shared_layers):
			if args["do_NER"]:
				cur_eval = ner_evaluate(model_2, X_test, y_test, label_list, special_tokens, args["seq_len"], batch_size=args["student_batch_size"]*gpus)
			else:
				y_pred = np.argmax(model_2.predict(X_test, batch_size=args["student_batch_size"]*gpus), axis=-1)
				cur_eval = (y_pred == y_test).sum() / len(y_test)
			if cur_eval >= best_eval:
				best_eval = cur_eval
				best_model_weights = model_2.get_weights()

		if stage >= 3+len(shared_layers):
			model_file = os.path.join(args["model_dir"], "model-stage-{}.h5".format(stage))
			history_file = os.path.join(args["model_dir"], "model-stage-{}-history.json".format(stage))
			if os.path.exists(model_file):
				logger.info ("Loadings weights for stage 3 from {}".format(model_file))
				model_2.load_weights(model_file)
			else:
				logger.info (model_2.summary())
				model_history = model_2.fit(X_train, y_train, batch_size=args["teacher_batch_size"]*gpus, verbose=2, shuffle=True, epochs=args["ft_epochs"], callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=args["patience"], restore_best_weights=True)], validation_data=(X_dev, y_dev))
				history = model_history.history
				model_2.save_weights(model_file)
				json.dump(history, open(history_file, 'w'))

			if args["do_NER"]:
				cur_eval = ner_evaluate(model_2, X_test, y_test, label_list, special_tokens, args["seq_len"], batch_size=args["student_batch_size"]*gpus)
			else:
				y_pred = np.argmax(model_2.predict(X_test, batch_size=args["student_batch_size"]*gpus), axis=-1)
				cur_eval = (y_pred == y_test).sum() / len(y_test)

			if cur_eval >= best_eval:
				best_eval = cur_eval
				best_model_weights = model_2.get_weights()

	model_2.set_weights(best_model_weights)
	logger.info ("Best eval score {}".format(best_eval))

	#save xtremedistil training config and final model weights
	json.dump(args, open(os.path.join(args["model_dir"], "xtremedistil-config.json"), 'w'))
	model_2.save_weights(os.path.join(args["model_dir"], "xtremedistil.h5"))
	logger.info ("Model and config saved to {}".format(args["model_dir"]))
