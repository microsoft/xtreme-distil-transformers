# XtremeDistil for Distilling Massive Multilingual Neural Networks
## ACL 2020 Microsoft Research [[Paper]](https://www.microsoft.com/en-us/research/publication/xtremedistil/) [[Video]](https://slideslive.com/38929189/xtremedistil-multistage-distillation-for-massive-multilingual-models)

***Update 12/27/2020***
Releasing [**XtremeDistil-v3**] with Tensorflow 2.3 and [HuggingFace Transformers](https://huggingface.co/transformers) for distilling all supported [pre-trained language models](https://huggingface.co/transformers/pretrained_models.html) with an unified API for multilingual text classification and sequence tagging tasks. 

*Install requirements*
```pip install -r requirements.txt```

*Sample usages for distilling different pre-trained language models (tested with Python 3.6.9 and CUDA 10.2)*

***Training***

*Sequence Labeling for Wiki NER*
```
PYTHONHASHSEED=42 python run_xtreme_distil.py --task $$PT_DATA_DIR/datasets/NER --model_dir $$PT_OUTPUT_DIR --seq_len 32  --transfer_file $$PT_DATA_DIR/datasets/NER/unlabeled.txt --do_NER --pt_teacher TFBertModel --pt_teacher_checkpoint --pt_teacher_checkpoint bert-base-multilingual-cased --student_batch_size 256 --teacher_batch_size 128  --pt_student_checkpoint minilm/minilm-l6-h384-uncased --distil_chunk_size 10000 --teacher_model_dir $$PT_OUTPUT_DIR --distil_multi_hidden_states --distil_attention --compress_word_embedding --freeze_word_embedding

```

*Text Classification for MNLI*
```
PYTHONHASHSEED=42 python run_xtreme_distil.py --task $$PT_DATA_DIR/glue_data/MNLI --model_dir $$PT_OUTPUT_DIR --seq_len 128  --transfer_file $$PT_DATA_DIR/glue_data/MNLI/train.tsv --do_pairwise --pt_teacher TFElectraModel --pt_teacher_checkpoint --pt_teacher_checkpoint google/electra-base-discriminator --student_batch_size 128  --pt_student_checkpoint minilm/minilm-l6-h384-uncased --teacher_model_dir $$PT_OUTPUT_DIR --teacher_batch_size 128 --distil_chunk_size 60000
```

***Model Outputs***

The above training code generates intermediate model checkpoints to continue the training in case of abrupt termination instead of starting from scratch -- all saved in $$PT_OUTPUT_DIR. The final output of the model consists of (i) `xtremedistil.h5` with distilled model weights, (ii) `xtremedistil-config.json` with the training configuration, and (iii) `word_embedding.npy` for the input word embeddings from the student model.

***Prediction***

```
PYTHONHASHSEED=42 python run_xtreme_distil_predict.py --do_eval --model_dir $$PT_OUTPUT_DIR --do_predict --pred_file ../../datasets/NER/unlabeled.txt
```

Arguments

```- refer to code for detailed arguments
- task folder contains
	-- train/dev/test '.tsv' files with text and classification labels / token-wise tags (space-separated)
	--- Example 1: feel good about themselves <tab> 1
	--- Example 2: '' Atelocentra '' Meyrick , 1884 <tab> O B-LOC O O O O
	-- label files containing class labels for sequence labeling
	-- transfer file containing unlabeled data
- model_dir to store/restore model checkpoints
```

If you use this code, please cite:
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
    abstract = "Deep and large pre-trained language models are the state-of-the-art for various natural language processing tasks. However, the huge size of these models could be a deterrent to using them in practice. Some recent works use knowledge distillation to compress these huge models into shallow ones. In this work we study knowledge distillation with a focus on multilingual Named Entity Recognition (NER). In particular, we study several distillation strategies and propose a stage-wise optimization scheme leveraging teacher internal representations, that is agnostic of teacher architecture, and show that it outperforms strategies employed in prior works. Additionally, we investigate the role of several factors like the amount of unlabeled data, annotation resources, model architecture and inference latency to name a few. We show that our approach leads to massive compression of teacher models like mBERT by upto 35x in terms of parameters and 51x in terms of latency for batch inference while retaining 95{\%} of its F1-score for NER over 41 languages.",
}
```

Code is released under [MIT](https://github.com/MSR-LIT/XtremeDistil/blob/master/LICENSE) license.
