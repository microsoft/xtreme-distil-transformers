# XtremeDistilTransformers for Distilling Massive Multilingual Neural Networks
## Microsoft Research [[Task-specific]](https://www.microsoft.com/en-us/research/publication/xtremedistil/) [[Task-agnostic with Task Transfer]](https://arxiv.org/pdf/2106.04563.pdf)

Releasing [**XtremeDistilTransformers**] with Tensorflow 2.3 and [HuggingFace Transformers](https://huggingface.co/transformers) with an unified API with the following features:
* Distil any supported [pre-trained language models](https://huggingface.co/transformers/pretrained_models.html) as teachers (e.g, Bert, Electra, Roberta)
* Initialize student model with any pre-trained model (e.g, MiniLM, DistilBert, TinyBert), or initialize from scratch
* Multilingual text classification and sequence tagging 
* Distil multiple hidden states from teacher
* Distil deep attention networks from teacher
* Pairwise and instance-level classification tasks (e.g, MNLI, MRPC, SST)
* Progressive knowledge transfer with gradual unfreezing
* Fast mixed precision training for distillation (e.g, mixed_float16, mixed_bfloat16)
* ONNX runtime inference

*Install requirements*
```pip install -r requirements.txt```

You can use the following *task-agnostic pre-distilled checkpoints* from XtremeDistilTransformers for (only) fine-tuning on labeled data from downstream tasks:
- [6/256 xtremedistil pre-trained checkpoint](https://huggingface.co/microsoft/xtremedistil-l6-h256-uncased)
- [6/384 xtremedistil pre-trained checkpoint](https://huggingface.co/microsoft/xtremedistil-l6-h384-uncased)
- [12/384 xtremedistil pre-trained checkpoint](https://huggingface.co/microsoft/xtremedistil-l12-h384-uncased)

For further performance improvement, initialize XtremeDistilTransformers with any of the above pre-distilled checkpoints for *task-specific distillation* with additional unlabeled data from the the downstream task for the best performance.

The following table shows the performance of the above checkpoints on GLUE dev set and SQuAD-v2.

| Models         | #Params | Speedup | MNLI | QNLI | QQP  | RTE  | SST  | MRPC | SQUAD2 | Avg   |
|----------------|--------|---------|------|------|------|------|------|------|--------|-------|
| BERT        | 109    | 1x       | 84.5 | 91.7 | 91.3 | 68.6 | 93.2 | 87.3 | 76.8   | 84.8 |
| DistilBERT  | 66     | 2x       | 82.2 | 89.2 | 88.5 | 59.9 | 91.3 | 87.5 | 70.7   | 81.3 |
| TinyBERT    | 66     | 2x       | 83.5 | 90.5 | 90.6 | 72.2 | 91.6 | 88.4 | 73.1   | 84.3 |
| MiniLM      | 66     | 2x       | 84.0   | 91.0   | 91.0   | 71.5 | 92.0   | 88.4 | 76.4   | 84.9  |
| MiniLM      | 22     | 5.3x     | 82.8 | 90.3 | 90.6 | 68.9 | 91.3 | 86.6 | 72.9   | 83.3 |
| XtremeDistil-l6-h256   | 13     | 8.7x     | 83.9 | 89.5 | 90.6   | 80.1 | 91.2 | 90.0   | 74.1   | 85.6 |
| XtremeDistil-l6-h384   | 22     | 5.3x     | 85.4 | 90.3 | 91.0   | 80.9 | 92.3 | 90.0   | 76.6   | 86.6 |
| XtremeDistil-l12-h384   | 33     | 2.7x     | 87.2 | 91.9 | 91.3   | 85.6 | 93.1 | 90.4   | 80.2   | 88.5 |

Tested with `tensorflow 2.3.1, transformers 4.1.1, torch 1.6.0, python 3.6.9 and CUDA 10.2`

*Sample usages for distilling different pre-trained language models*

***Training***

*Sequence Labeling for Wiki NER*
```
PYTHONHASHSEED=42 python run_xtreme_distil.py 
--task $$PT_DATA_DIR/datasets/NER 
--model_dir $$PT_OUTPUT_DIR 
--seq_len 32  
--transfer_file $$PT_DATA_DIR/datasets/NER/unlabeled.txt 
--do_NER 
--pt_teacher TFBertModel 
--pt_teacher_checkpoint bert-base-multilingual-cased 
--student_distil_batch_size 256 
--student_ft_batch_size 32
--teacher_batch_size 128  
--pt_student_checkpoint microsoft/xtremedistil-l6-h384-uncased 
--distil_chunk_size 10000 
--teacher_model_dir $$PT_OUTPUT_DIR 
--distil_multi_hidden_states 
--distil_attention 
--compress_word_embedding 
--freeze_word_embedding
--opt_policy mixed_float16
```

*Text Classification for MNLI*
```
PYTHONHASHSEED=42 python run_xtreme_distil.py 
--task $$PT_DATA_DIR/glue_data/MNLI 
--model_dir $$PT_OUTPUT_DIR 
--seq_len 128  
--transfer_file $$PT_DATA_DIR/glue_data/MNLI/train.tsv 
--do_pairwise 
--pt_teacher TFElectraModel 
--pt_teacher_checkpoint google/electra-base-discriminator 
--student_distil_batch_size 128  
--student_ft_batch_size 32
--pt_student_checkpoint microsoft/xtremedistil-l6-h384-uncased 
--teacher_model_dir $$PT_OUTPUT_DIR 
--teacher_batch_size 32
--distil_chunk_size 300000
--opt_policy mixed_float16
```

Alternatively, use TinyBert pre-trained student model checkpoint as `--pt_student_checkpoint nreimers/TinyBERT_L-4_H-312_v2`

*Arguments*

```- refer to code for detailed arguments

- task folder contains
	-- train/dev/test '.tsv' files with text and classification labels / token-wise tags (space-separated)
	--- Example 1: feel good about themselves <tab> 1
	--- Example 2: '' Atelocentra '' Meyrick , 1884 <tab> O B-LOC O O O O
	-- label files containing class labels for sequence labeling
	-- transfer file containing unlabeled data
	
- model_dir to store/restore model checkpoints

- task arguments
-- do_pairwise for pairwise classification tasks like MNLI and MRPC
-- do_NER for sequence labeling

- teacher arguments
-- pt_teacher for teacher model to distil (e.g., TFBertModel, TFRobertaModel, TFElectraModel)
-- pt_teacher_checkpoint for pre-trained teacher model checkpoints (e.g., bert-base-multilingual-cased, roberta-large, google/electra-base-discriminator)

- student arguments
-- pt_student_checkpoint to initialize from pre-trained small student models (e.g., MiniLM, DistilBert, TinyBert)
-- instead of pre-trained checkpoint, initialize a raw student from scratch with
--- hidden_size
--- num_hidden_layers
--- num_attention_heads

- distillation features
-- distil_multi_hidden_states to distil multiple hidden states from the teacher
-- distil_attention to distil deep attention network of the teacher
-- compress_word_embedding to initialize student word embedding with SVD-compressed teacher word embedding (useful for multilingual distillation)
-- freeze_word_embedding to keep student word embeddings frozen during distillation (useful for multilingual distillation)
-- opt_policy (e.g., mixed_float16 for GPU and mixed_bfloat16 for TPU)
-- distil_chunk_size for using transfer data in chunks during distillation (reduce for OOM issues, checkpoints are saved after every distil_chunk_size steps)
```

***Model Outputs***

The above training code generates intermediate model checkpoints to continue the training in case of abrupt termination instead of starting from scratch -- all saved in $$PT_OUTPUT_DIR. The final output of the model consists of (i) `xtremedistil.h5` with distilled model weights, (ii) `xtremedistil-config.json` with the training configuration, and (iii) `word_embedding.npy` for the input word embeddings from the student model.

***Prediction***

```
PYTHONHASHSEED=42 python run_xtreme_distil_predict.py 
--do_eval 
--model_dir $$PT_OUTPUT_DIR 
--do_predict 
--pred_file ../../datasets/NER/unlabeled.txt
--opt_policy mixed_float16
```

****ONNX Runtime Inference***

You can also use [ONXX Runtime](https://github.com/microsoft/onnxruntime) for inference speedup with the following script:

```
PYTHONHASHSEED=42 python run_xtreme_distil_predict_onnx.py 
--do_eval 
--model_dir $$PT_OUTPUT_DIR 
--do_predict 
--pred_file ../../datasets/NER/unlabeled.txt
```

For details on ONNX Runtime Inference, environment and arguments refer to this [Notebook](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/transformers/notebooks/Tensorflow_Keras_Bert-Squad_OnnxRuntime_CPU.ipynb)
The script is for online inference with batch_size=1.

****Continued Fine-tuning***

You can continue fine-tuning the distilled/compressed student model on more labeled data with the following script:

```
PYTHONHASHSEED=42 python run_xtreme_distil_ft.py --model_dir $$PT_OUTPUT_DIR 
```

If you use this code, please cite:
```
@misc{mukherjee2021xtremedistiltransformers,
      title={XtremeDistilTransformers: Task Transfer for Task-agnostic Distillation}, 
      author={Subhabrata Mukherjee and Ahmed Hassan Awadallah and Jianfeng Gao},
      year={2021},
      eprint={2106.04563},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
```
@inproceedings{mukherjee-hassan-awadallah-2020-xtremedistil,
    title = "{X}treme{D}istil: Multi-stage Distillation for Massive Multilingual Models",
    author = "Mukherjee, Subhabrata  and
      Hassan Awadallah, Ahmed",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.202",
    pages = "2221--2234",
}
```

Code is released under [MIT](https://github.com/MSR-LIT/XtremeDistil/blob/master/LICENSE) license.
