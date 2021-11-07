import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import argparse
import os
import random
from tqdm import tqdm
import numpy as np
import spacy
import json
import pickle
from multiprocessing import Pool
from collections import defaultdict, Counter
from transformers import LukeTokenizer, LukeModel, AdamW, BertTokenizer, BertModel
#from sentence_transformers import SentenceTransformer
from clustering.utils import get_rankings, set_logger, update_logger
from clustering.kmeans import get_kmeans, get_metric, get_kmeans_score
from topic_modeling.dataloader import get_train_loader
from clustering.trainer import ClusterLearner
from topic_modeling.consts import TOKENIZER, NLP, ARGS, DEVICE
from topic_modeling.utils import read_data, get_features, get_probs
from uctopic.models import UCTopicConfig, UCTopicCluster


# def get_glove_features(data, label_dict):

#     all_features = []
#     all_labels = []

#     word_feature_dict = dict()
#     with open('glove.6B.300d.txt') as f:
#         for line in tqdm(f, desc='Reading Glove.', ncols=100):
#             line = line.strip().split(' ')
#             assert len(line) == 301
#             word_feature_dict[line[0]] = np.array([float(number) for number in line[1:]])
#     total = 0
#     zero = 0
#     for batch in tqdm(data, ncols=100, desc='Generate all features...'):

#         sentence, spans, ner_labels = batch

#         for span, label in zip(spans, ner_labels):
#             if label in label_dict:
#                 all_labels.append(label_dict[label])
#             else:
#                 continue

#             total += 1
#             start, end = span
#             phrase_tokens = [token.lower() for token in sentence[start:end+1]]
#             phrase_repr = [word_feature_dict[token] for token in phrase_tokens if token in word_feature_dict]
#             if len(phrase_repr)==0:
#                 phrase_repr=np.zeros(300)
#                 zero += 1
#             else:
#                 phrase_repr = np.array(phrase_repr).mean(axis=0)
#             assert phrase_repr.shape[0] == 300
#             all_features.append(phrase_repr)
            
#     print(zero/total)
#     all_features = torch.FloatTensor(all_features)
#     all_labels = torch.LongTensor(all_labels)

#     return all_features, all_labels

class NounPhraser:
    @staticmethod
    def process_data(data):

        sentence_dict = dict()
        phrase_list = []
        for line in data:
            doc_id = line['doc_id']
            text = line['text']
            sentence_dict[doc_id] = text

        pool = Pool(processes=ARGS.num_workers)
        pool_func = pool.imap(func=NounPhraser.rule_based_noun_phrase, iterable=data)
        doc_tuples = list(tqdm(pool_func, total=len(data), ncols=100, desc=f'Process data and extract phrases'))
        for phrases in doc_tuples:
            phrase_list += phrases
        pool.close()
        pool.join()

        return sentence_dict, phrase_list

    @staticmethod    
    def rule_based_noun_phrase(line):

        definite_articles = {'a', 'the', 'an', 'this', 'those', 'that', 'these', \
                                  'my', 'his', 'her', 'your', 'their', 'our'}
        text = line['text']
        if not text:
            return []
        doc_id = line['doc_id']
        doc = NLP(text)
        if len(doc) > ARGS.max_length:
            return []

        phrases = []
        for chunk in doc.noun_chunks:
            start, end = chunk.start, chunk.end ## token-level idx
            if len(chunk.text.split()) > 1:
                left_p = '(' in chunk.text
                right_p = ')' in chunk.text
                if left_p == right_p:
                    ps = chunk.text
                    if ps.split(" ")[0].lower() in definite_articles:
                        new_ps = " ".join(ps.split(" ")[1:])
                        start_char = chunk.start_char + len(ps) - len(new_ps)

                        span_lemma = ' '.join([doc[i].lemma_.lower() for i in range(start+1, end)])
                        assert doc.text[start_char:chunk.end_char] == new_ps
                        phrases.append((doc_id, start_char, chunk.end_char, span_lemma))

                    else:
                        span_lemma = ' '.join([doc[i].lemma_.lower() for i in range(start, end)])
                        phrases.append((doc_id, chunk.start_char, chunk.end_char, span_lemma))

            else:
                if doc[chunk.start].pos_ != 'PRON':
                    span_lemma = ' '.join([doc[i].lemma_.lower() for i in range(start, end)])
                    phrases.append((doc_id, chunk.start_char, chunk.end_char, span_lemma))

        return phrases

# def get_features(sentence_dict, phrase_list_sampled, model):

#     all_features = []
#     with torch.no_grad():

#         for batch in tqdm(batchify(sentence_dict, phrase_list_sampled, ARGS.batch_size), ncols=100, desc='Generate all features...'):

#             text_batch, span_batch = batch
            
#             inputs = TOKENIZER(text_batch, entity_spans=span_batch, padding=True, add_prefix_space=True, return_tensors="pt")

#             for k,v in inputs.items():
#                 inputs[k] = v.to(DEVICE)

#             luke_outputs, entity_pooling = model(**inputs)
            
#             all_features.append(entity_pooling.squeeze().detach().cpu())

#     all_features = torch.cat(all_features, dim=0)

#     return all_features

def main():

    config = UCTopicConfig.from_pretrained("studio-ousia/luke-base")
    model = UCTopicCluster(config, ARGS)
    model.load_state_dict(torch.load('result/pytorch_model.bin'), strict= False)
    model.to(DEVICE)
    model.eval()

    data_path = os.path.join(ARGS.data_path)
    ARGS.num_classes = eval(ARGS.num_classes)

    data = read_data(data_path)
    sentence_dict, phrase_list = NounPhraser.process_data(data)

    # To make sure the number of topics, we randomly sample part of phrases first
    phrase_list_sampled = random.sample(phrase_list, min(ARGS.sample_num_cluster, len(phrase_list)))    
    features = get_features(sentence_dict, phrase_list_sampled, model)
    #features, labels = get_phrase_bert_features(data)

    kmeans_scores = []
    for num_class in range(ARGS.num_classes[0], ARGS.num_classes[1]+1):
        score = get_kmeans_score(features, num_class)
        print('For n_clusters = ', num_class, 'The silhouette_score is: ', score)
        kmeans_scores.append((num_class, score))

    kmeans_scores = sorted(kmeans_scores, key=lambda x: x[1], reverse=True)
    num_class = kmeans_scores[0][0]
    print('We select the number of topics: ', num_class)
    #num_class = 22

    ## To finetune, we randomly sample part of phrases
    phrase_list_sampled = random.sample(phrase_list, min(ARGS.sample_num_finetune, len(phrase_list)))    
    features = get_features(sentence_dict, phrase_list_sampled, model)
    score_factor, score_cosine, cluster_centers = get_kmeans(features, None, num_class)

    rankings = get_rankings(score_cosine, positive_ratio=0.1)

    pseudo_label_dict = defaultdict(list)

    for i in range(len(rankings)):
        for j in range(len(rankings[i])):
            pseudo_label_dict[phrase_list[rankings[i][j]][-1]].append(j)
        

    ## majority vote
    for phrase, predictions in pseudo_label_dict.items():
        pseudo_label_dict[phrase] = Counter(predictions).most_common()[0][0]

    model.update_cluster_centers(cluster_centers)

    # dataset loader
    train_loader = get_train_loader(sentence_dict, phrase_list_sampled, ARGS, pseudo_label_dict)

    # optimizer 
    optimizer = torch.optim.Adam(model.parameters(), lr=ARGS.lr)

    print(optimizer)

    # set up logger
    #logger = set_logger(ARGS.save_path)
    global_step = 0
    # set up the trainer
    learner = ClusterLearner(model, optimizer)
    model.train()
    ret = False
    for epoch in range(ARGS.epoch):
        tqdm_dataloader = tqdm(train_loader, ncols=100)
        for features in tqdm_dataloader:

            for feature in features:
                for k, v in feature.items():
                    feature[k] = v.to(DEVICE)

            loss = learner.forward(features)

            tqdm_dataloader.set_description(
                'Epoch{}, Global Step {}, CL-loss {:.5f}'.format(
                    epoch, global_step,  loss['Instance-CL_loss']
                    ))

            #update_logger(logger, loss, global_step)
            global_step+=1
            if global_step >= ARGS.finetune_step:
                ret = True
                break
        if ret:
            break

    model.eval()

    all_prob = get_probs(sentence_dict, phrase_list, model)
    #all_embeddings, all_prob = get_features(sentence_dict, phrase_list, model, return_prob=True)
    all_pred = all_prob.max(1)[1].tolist()
    all_prob = all_prob.numpy()

    assert len(phrase_list) == len(all_pred)

    phrase_pred = []
    merge_phrase_dict = defaultdict(list)
    topic_phrase_dict = defaultdict(list)
    
    for phrase, pred, prob in zip(phrase_list, all_pred, all_prob):

        phrase_pred.append([phrase, pred])
        merge_phrase_dict[phrase[-1]].append(prob)

    for phrase, prob_list in merge_phrase_dict.items():

        prob_mean = np.array(prob_list).mean(axis=0)
        pred = prob_mean.argmax()
        merge_phrase_dict[phrase] = [pred, prob_mean[pred]]
        topic_phrase_dict[str(pred)].append((phrase, prob_mean[pred]))

    for topic, v in topic_phrase_dict.items():

        topic_phrase_dict[str(topic)] = [(line[0], str(round(line[1], 4))) for line in sorted(v, key=lambda x: x[1], reverse=True)]


    results_path = os.path.join(ARGS.save_path, ARGS.dataset)
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    with open(os.path.join(results_path, 'phrase_instances_pred.json'), 'w', encoding='utf8') as f:

        json.dump(phrase_pred, f)     

    with open(os.path.join(results_path, 'merged_phrase_pred_prob.pickle'), 'wb') as f:

        pickle.dump(merge_phrase_dict, f)    

    with open(os.path.join(results_path, 'topics_phrases.json'), 'w', encoding='utf8') as f:

        json.dump(topic_phrase_dict, f)      
    
 
if __name__ == '__main__':

    main()