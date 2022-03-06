## UCTopic: Unsupervised Contrastive Learning for Phrase Representations and Topic Mining

This repository contains the code and pre-trained models for our paper [UCTopic: Unsupervised Contrastive Learning for Phrase Representations and Topic Mining](https://arxiv.org/abs/2202.13469).

## To-Do:
We continue completing the following sections:

 * Using UCTopic with PyPI
 * Get Topical Phrases


## Quick Links

  - [Overview](#overview)
  - [Pretrained Model](#pretrained-model)
  - [Get Phrase Embeddings](#get-phrase-embeddings)
  - [Experiments](#experiments)
    - [Requirements](#requirements)
    - [Datasets](#datasets)
    - [Entity Clustering](#entity-clustering)
    - [Topic Mining](#topic-mining)
  - [Pretraining](#pretraining)
  - [Contact](#contact)
  - [Citation](#citation)

## Overview

We propose UCTopic, a novel unsupervised contrastive learning framework for context-aware phrase representations and topic mining. UCTopic is pretrained in a large scale to distinguish if the contexts of two phrase mentions have the same semantics. The key to pretraining is positive pair construction from our phrase-oriented assumptions. However, we find traditional in-batch negatives cause performance decay when finetuning on a dataset with small topic numbers. Hence, we propose cluster-assisted contrastive learning(CCL) which largely reduces noisy negatives by selecting negatives from clusters and further improves phrase representations for topics accordingly.

## Pretrained Model
Our released model:
|              Model              | Note|
|:-------------------------------|------|
|[uctopic-base](https://drive.google.com/file/d/1XQzi4E9ctdI373CK5O-pXQyBvOONssp1/view?usp=sharing)| Pretrained UCTopic model based on [LUKE-BASE](https://arxiv.org/abs/2010.01057)

Unzip to get `uctopic-base` folder.

## Get Phrase Embeddings
We downloaded the dependencies of UCTopic from HuggingFace's `transformers`, so you can get phrase embeddings after installing [Pytorch](https://pytorch.org/). Basically, our model inputs are same as [LUKE](https://huggingface.co/docs/transformers/model_doc/luke). Note: please input only <strong>ONE</strong> span each time, otherwise, will have performance decay according to our empirical results.

```python
from uctopic import UCTopicTokenizer, UCTopic

tokenizer = UCTopicTokenizer.from_pretrained('uctopic-base') # Path to your uctopic-base folder
model = UCTopic.from_pretrained('uctopic-base')

text = "Beyoncé lives in Los Angeles."
entity_spans = [(17, 28)] # character-based entity span corresponding to "Los Angeles"

inputs = tokenizer(text, entity_spans=entity_spans, add_prefix_space=True, return_tensors="pt")
outputs, phrase_repr = model(**inputs)
```
`phrase_repr` is the phrase embedding (size `[768]`) of the phrase `Los Angeles`. `outputs` has the same format as the outputs from `LUKE`.

## Experiments
In this section, we re-implement experiments in our paper.

### Requirements
First, install PyTorch by following the instructions from [the official website](https://pytorch.org). To faithfully reproduce our results, please use the correct `1.9.0` version corresponding to your platforms/CUDA versions.

Then run the following script to install the remaining dependencies,
```bash
pip install -r requirements.txt
```

Download `en_core_web_sm` model from spacy,
```bash
python -m spacy download en_core_web_sm
```

### Datasets
The downstream datasets used in our experiments can be downloaded from [here](https://drive.google.com/file/d/1dVIp9li1Wdh0JgU8slsWm0ObcitbQtSL/view?usp=sharing).

### Entity Clustering
The config file of entity clustering is `clustering/consts.py` and most arguments are self-explained. Please setup `--gpu` and `--data_path` before running. The clustering scores will be printed.

Clustering with our pre-trained phrase embeddings.
```bash
python clustering.py --gpu 0
```
Clustering with our pre-trained phrase embeddings and Cluster-Assisted Constrastive Learning (CCL) proposed in our paper.
```bash
python clustering_ccl_finetune.py --gpu 0
```

### Topic Mining
The config file of entity clustering is `topic_modeling/consts.py`.

**Key Argument Table**
|    Arguments     | Description |
|:-----------------|:-----------:|
| --num_classes     |**Min** and **Max** number of classes, e.g., `[5, 15]`. Our model will find the class number by [silhouette_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html).|
| --sample_num_cluster   |Number of sampled phrases to confirm class number.|
| --sample_num_finetune|Number of sampled phrases for CCL finetuning.|
| --contrastive_num|Number of negative classes for CCL finetuning.|
| --finetune_step | CCL finetuning steps (maximum global steps for finetuning).|

**Tips**: Please tune `--batch_size` or `--contrastive_num` for suitable GPU memory usage.

Topic mining with our pre-trained phrase embeddings and Cluster-Assisted Constrastive Learning (CCL) proposed in our paper.
```bash
python find_topic.py --gpu 0
```
**Outputs**

We output three files under `topic_results`:
|    File Name     | Description |
|:-----------------|:-----------:|
| `merged_phraes_pred_prob.pickle` |A dictionary of phrases and their topic number and prediction probability. A topic of a phrase is merged from all phrase mentioins. `{phrase: [topic_id, probability]}`, e.g., {'fair prices': [0, 0.34889686]}|
| `phrase_instances_pred.json`| A list of all mined phrase mentions. Each element is `[[doc_id, start, end, phrase_mention], topic_id]`.|
| `topics_phrases.json`|A dictionary of topics and corresponding phrases sorted by probability. `{'topic_id': [[phrase1, prob1], [phrase2, prob2]]}`|

### Pretraining

**Data**

For unsupervised pretraining of UCTopic, we use article and span with links from English Wikipedia and Wikidata. Our processed dataset can be downloaded from [here](https://drive.google.com/file/d/1wflsmhPI9J0ZA6aVRl2mQjHIE6JIvzAv/view?usp=sharing).

**Training scripts**

We provide example training scripts and our default training parameters for unsupervised training of UCTopic in `run_example.sh`.

```bash
bash run_example.sh
```

Arguments description can be found in `pretrain.py`. All the other arguments are standard Huggingface's `transformers` training arguments.

**Convert models**

Our pretrained checkpoints are slightly different from the checkpoint `uctopic-base`. Please refer `convert_uctopic_parameters.py` to convert it.

## Contact

If you have any questions related to the code or the paper, feel free to email Jiacheng (`j9li@eng.ucsd.edu`). If you encounter any problems when using the code, or want to report a bug, you can open an issue. Please try to specify the problem with details so we can help you better and quicker!

## Citation

Please cite our paper if you use UCTopic in your work:

```bibtex
@article{Li2022UCTopicUC,
  title={UCTopic: Unsupervised Contrastive Learning for Phrase Representations and Topic Mining},
  author={Jiacheng Li and Jingbo Shang and Julian McAuley},
  journal={ArXiv},
  year={2022},
  volume={abs/2202.13469}
}
```