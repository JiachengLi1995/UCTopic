# UCTopic

This repository contains the code of model UCTopic and an easy-to-use tool UCTopicTool used for <strong>Topic Mining</strong>, <strong>Unsupervised Aspect Extractioin</strong> or <strong>Phrase Retrieval</strong>.

Our ACL 2022 paper [UCTopic: Unsupervised Contrastive Learning for Phrase Representations and Topic Mining](https://arxiv.org/abs/2202.13469).

# Quick Links

  - [Overview](#overview)
  - [Pretrained Model](#pretrained-model)
  - [Getting Started](#getting-started)
    - [UCTopic Model](#uctopic-model)
    - [UCTopicTool](#uctopictool)
  - [Experiments in Paper](#experiments)
    - [Requirements](#requirements)
    - [Datasets](#datasets)
    - [Entity Clustering](#entity-clustering)
    - [Topic Mining](#topic-mining)
  - [Pretraining](#pretraining)
  - [Contact](#contact)
  - [Citation](#citation)

# Overview

We propose UCTopic, a novel unsupervised contrastive learning framework for context-aware phrase representations and topic mining. UCTopic is pretrained in a large scale to distinguish if the contexts of two phrase mentions have the same semantics. The key to pretraining is positive pair construction from our phrase-oriented assumptions. However, we find traditional in-batch negatives cause performance decay when finetuning on a dataset with small topic numbers. Hence, we propose cluster-assisted contrastive learning(CCL) which largely reduces noisy negatives by selecting negatives from clusters and further improves phrase representations for topics accordingly.

# Pretrained Model
Our released model:
|              Model              | Note|
|:-------------------------------|------|
|[uctopic-base](https://drive.google.com/file/d/1XQzi4E9ctdI373CK5O-pXQyBvOONssp1/view?usp=sharing)| Pretrained UCTopic model based on [LUKE-BASE](https://arxiv.org/abs/2010.01057)

Unzip to get `uctopic-base` folder.

# Getting Started
We provide an easy-to-use phrase representation tool based on our UCTopic model. To use the tool, first install the uctopic package from PyPI
```bash
pip install uctopic
```
Or directly install it from our code
```bash
python setup.py install
```

## UCTopic Model
After installing the package, you can load our model by just two lines of code
```python
from uctopic import UCTopic
model = UCTopic.from_pretrained('JiachengLi/uctopic-base')
```
The model will automatically download pre-trained parameters from [HuggingFace's models](https://huggingface.co/models). If you encounter any problem when directly loading the models by HuggingFace's API, you can also download the models manually from the above table and use `model = UCTopic.from_pretrained({PATH TO THE DOWNLOAD MODEL})`.

To get pre-trained <strong>phrase representations</strong>, our model inputs are same as [LUKE](https://huggingface.co/docs/transformers/model_doc/luke). Note: please input only <strong>ONE</strong> span each time, otherwise, will have performance decay according to our empirical results.

```python
from uctopic import UCTopicTokenizer, UCTopic

tokenizer = UCTopicTokenizer.from_pretrained('JiachengLi/uctopic-base')
model = UCTopic.from_pretrained('JiachengLi/uctopic-base')

text = "Beyoncé lives in Los Angeles."
entity_spans = [(17, 28)] # character-based entity span corresponding to "Los Angeles"

inputs = tokenizer(text, entity_spans=entity_spans, add_prefix_space=True, return_tensors="pt")
outputs, phrase_repr = model(**inputs)
```
`phrase_repr` is the phrase embedding (size `[768]`) of the phrase `Los Angeles`. `outputs` has the same format as the outputs from `LUKE`.

## UCTopicTool
We provide a tool `UCTopicTool` built on `UCTopic` for efficient phrase encoding, topic mining (or unsupervised aspect extraction) or phrase retrieval.

### Initialization

`UCTopicTool` is initialized by giving the `model_name_or_path` and `device`.
```python
from uctopic import UCTopicTool

topic_tool = UCTopicTool('JiachengLi/uctopic-base', device='cuda:0')
```

### Phrase Encoding

Phrases are encoded by our method `UCTopicTool.encode` in batches, which is more efficient than `UCTopic`.
```python
phrases = [["This place is so much bigger than others!", (0, 10)],
           ["It was totally packed and loud.", (15, 21)],
           ["Service was on the slower side.", (0, 7)],
           ["I ordered 2 mojitos: 1 lime and 1 mango.", (12, 19)],
           ["The ingredient weren't really fresh.", (4, 14)]]

embeddings = topic_tool.encode(phrases) # len(embeddings) is equal to len(phrases)
```
**Note**: Each instance in `phrases` contains only one sentence and one span (character-level position) in format `[sentence, span]`.

Arguments for `UCTopicTool.encode` are as follows,
* **phrase** (List) - A list of `[sentence, span]` to be encoded.
* **return_numpy** (bool, *optional*, defaults to `False`) - Return `numpy.array` or `torch.Tensor`.
* **normalize_to_unit** (bool, *optional*, defaults to `True`) - Normalize all embeddings to unit vectors.
* **keepdim** (bool, *optional*, defaults to `True`) - Keep dimension size `[instance_number, hidden_size]`.
* **batch_size** (int, *optional*, defaults to `64`) - The size of mini-batch in the model.

### Topic Mining and Unsupervised Aspect Extraction

The method `UCTopicTool.topic_mining` can mine topical phrases or conduct aspect extraction from sentences with or without spans.

```python
sentences = ["This place is so much bigger than others!",
             "It was totally packed and loud.",
             "Service was on the slower side.",
             "I ordered 2 mojitos: 1 lime and 1 mango.",
             "The ingredient weren't really fresh."]

spans = [[(0, 10)],                       # This place
         [(15, 21), (26, 30)],            # packed; loud
         [(0, 7)],                        # Service
         [(12, 19), (21, 27), (32, 39)],  # mojitos; 1 lime; 1 mango
         [(4, 14)]]                       # ingredient
# len(sentences) is equal to len(spans)
output_data, topic_phrase_dict = tool.topic_mining(sentences, spans, \
                                                   n_clusters=[15, 25])

# predict topic for new phrases
phrases = [["The food here is amazing!", (4, 8)],
           ["Lovely ambiance with live music!", (21, 31)]]

topics = tool.predict_topic(phrases)
```
**Note**: If `spans` is not given, `UCTopicTool` will extract noun phrases by [spaCy](https://spacy.io/).

Arguments for `UCTopicTool.topic_mining` are as follows,

Data arguments:
* **sentences** (List) - A List of sentences for topic mining.
* **spans** (List, *optional*, defaults to `None`) - A list of span list corresponding sentences, e.g., `[[(0, 9), (5, 7)], [(1, 2)]]` and `len(sentences)==len(spans)`. If None, automatically mine phrases from noun chunks.

Clustering arguments:
* **n_clusters** (int or List, *optional*, defaults to `2`) - The number of topics. When `n_clusters` is a list, `n_clusters[0]` and `n_clusters[1]` will be the minimum and maximum numbers to search, `n_clusters[2]` is the search step length (if not provided, default to 1).
* **meric** (str, *optional*, defaults to `"cosine"`) - The metric to measure the distance between vectors. `"cosine"` or `"euclidean"`.
* **batch_size** (int, *optional*, defaults to `64`) - The size of mini-batch for phrase encoding.
* **max_iter** (int, *optional*, defaults to `300`) - The maximum iteration number of kmeans.
        
CCL-finetune arguments:
* **ccl_finetune** (bool, *optional*, defaults to `True`) - Whether to conduct CCL-finetuning in the paper.
* **batch_size_finetune** (int, *optional*, defaults to `8`) - The size of mini-batch for finetuning.
* **max_finetune_num** (int, *optional*, defaults to `100000`) - The maximum number of training instances for finetuning.
* **finetune_step** (int, *optional*, defaults to `2000`) - The number of training steps for finetuning.
* **contrastive_num** (int, *optional*, defaults to `5`) - The number of negatives in contrastive learning.
* **positive_ratio** (float, *optional*, defaults to `0.1`) - The ratio of the most confident instances for finetuning.
* **n_sampling** (int, *optional*, defaults to `10000`) - The number of sampled examples for cluster number confirmation and finetuning. Set to `-1` to use the whole dataset.
* **n_workers** (int, *optional*, defaults to `8`) - The number of workers for preprocessing data.

Returns for `UCTopicTool.topic_mining` are as follows,
* **output_data** (List) - A list of sentences and corresponding phrases and topic numbers. Each element is `[sentence, [[start1, end1, topic1], [start2, end2, topic2]]]`.
* **topic_phrase_dict** (Dict) - A dictionary of topics and the list of phrases under a topic. The phrases are sorted by their confidence scores. E.g., `{topic: [[phrase1, score1], [phrase2, score2]]}`.


The method `UCTopicTool.predict_topic` predicts the topic ids for new phrases based on your training results from `UCTopicTool.topic_mining`. The inputs of `UCTopicTool.predict_topic` are same as `UCTopicTool.encode` and returns a list of topic ids (int).


### Phrase Similarities and Retrieval

The method `UCTopicTool.similarity` compute the cosine similarities between two groups of phrases:

```python
phrases_a = [["This place is so much bigger than others!", (0, 10)],
           ["It was totally packed and loud.", (15, 21)]]

phrases_b = [["Service was on the slower side.", (0, 7)],
           ["I ordered 2 mojitos: 1 lime and 1 mango.", (12, 19)],
           ["The ingredient weren't really fresh.", (4, 14)]]

similarities = tool.similarity(phrases_a, phrases_b)
```
Arguments for `UCTopicTool.similarity` are as follows,
* **queries** (List) - A list of `[sentence, span]` as queries.
* **keys** (List or `numpy.array`) - A list of `[sentence, span]` as keys or phrase representations (`numpy.array`) from `UCTopicTool.encode`.
* **batch_size** (int, *optional*, defaults to `64`) - The size of mini-batch in the model.

`UCTopicTool.similarity` returns a `numpy.array` contains the similarities between phrase pairs in two groups.


The methods `UCTopicTool.build_index` and `UCTopicTool.search` are used for phrase retrieval:
```python
phrases = [["This place is so much bigger than others!", (0, 10)],
           ["It was totally packed and loud.", (15, 21)],
           ["Service was on the slower side.", (0, 7)],
           ["I ordered 2 mojitos: 1 lime and 1 mango.", (12, 19)],
           ["The ingredient weren't really fresh.", (4, 14)]]

# query multiple phrases
query1 = [["The food here is amazing!", (4, 8)],
           ["Lovely ambiance with live music!", (21, 31)]]  

# query single phrases
query2 = ["The food here is amazing!", (4, 8)]

tool.build_index(phrases)
results = tool.search(query1, top_k=3)
# or
results = tool.search(query2, top_k=3)
```
We also support [faiss](https://github.com/facebookresearch/faiss), an efficient similarity search library. Just install the package following [instructions](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md) here and `UCTopicTool` will automatically use `faiss` for efficient search.

`UCTopicTool.search` returns the ranked top k phrases for each query.


### Save and Load finetuned UCTopicTool

The methods `UCTopicTool.save` and `UCTopicTool.load` are used for save and load all paramters of `UCTopicTool`.

Save:
```python
tool = UCTopicTool('JiachengLi/uctopic-base', 'cuda:0')
# finetune UCTopic with CCL
output_data, topic_phrase_dict = tool.topic_mining(sentences, spans, \
                                                   n_clusters=[15, 25])

tool.save(**your directory**)
```

Load:
```python
tool = UCTopicTool('JiachengLi/uctopic-base', 'cuda:0')
tool.load(**your directory**)
```
The loaded parameters will be used for all methods (for encoding, topic mining, phrase similarities and retrieval) introduced above.

# Experiments
In this section, we re-implement experiments in our paper.

## Requirements
First, install PyTorch by following the instructions from [the official website](https://pytorch.org). To faithfully reproduce our results, please use the correct `1.9.0` version corresponding to your platforms/CUDA versions.

Then run the following script to install the remaining dependencies,
```bash
pip install -r requirements.txt
```

Download `en_core_web_sm` model from spacy,
```bash
python -m spacy download en_core_web_sm
```

## Datasets
The downstream datasets used in our experiments can be downloaded from [here](https://drive.google.com/file/d/1dVIp9li1Wdh0JgU8slsWm0ObcitbQtSL/view?usp=sharing).

## Entity Clustering
The config file of entity clustering is `clustering/consts.py` and most arguments are self-explained. Please setup `--gpu` and `--data_path` before running. The clustering scores will be printed.

Clustering with our pre-trained phrase embeddings.
```bash
python clustering.py --gpu 0
```
Clustering with our pre-trained phrase embeddings and Cluster-Assisted Constrastive Learning (CCL) proposed in our paper.
```bash
python clustering_ccl_finetune.py --gpu 0
```

## Topic Mining
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

# Contact

If you have any questions related to the code or the paper, feel free to email Jiacheng (`j9li@eng.ucsd.edu`). If you encounter any problems when using the code, or want to report a bug, you can open an issue. Please try to specify the problem with details so we can help you better and quicker!

# Citation

Please cite our paper if you use UCTopic in your work:

```bibtex
@inproceedings{Li2022UCTopicUC,
    title = "{UCT}opic: Unsupervised Contrastive Learning for Phrase Representations and Topic Mining",
    author = "Li, Jiacheng  and
      Shang, Jingbo  and
      McAuley, Julian",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.426",
    doi = "10.18653/v1/2022.acl-long.426",
    pages = "6159--6169"
}
```
