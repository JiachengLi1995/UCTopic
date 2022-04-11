import logging
import random
import os
from tqdm import tqdm
import torch
from torch import Tensor
import numpy as np
from numpy import ndarray
from collections import defaultdict, Counter
from .models import UCTopicCluster
from .tokenizer import UCTopicTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Type, Union

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class UCTopicTool(object):
    def __init__(self, model_name_or_path: str, 
                       device: str = None,
                       num_cells: int = 100,
                       num_cells_in_search: int = 10):
        
        self.tokenizer = UCTopicTokenizer.from_pretrained(model_name_or_path)
        self.model = UCTopicCluster.from_pretrained(model_name_or_path)
        self.config = self.model.config

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device_type = device
            self.device_num = 0
        else:
            self.device_type = device.split(':')[0]
            self.device_num = int(device.split(':')[1])
        self.device = torch.device(device)

        self.num_cells = num_cells
        self.num_cells_in_search = num_cells_in_search

        self.do_topic_mining = False
        self.centers = None
        self.merge_phrase_dict = None


    def encode(self, phrase: List,
                return_numpy: bool = False,
                normalize_to_unit: bool = True,
                keepdim: bool = True,
                batch_size: int = 64) -> Union[ndarray, Tensor]:

        '''
        phrase: [sentence, span] or [[sentence, span], [sentence, span]].
                e.g., [sentence, span]: ["Beyoncé lives in Los Angeles.", (17, 28)] -> Los Angeles
        return_numpy: return numpy.array or torch.Tensor
        nomralize_to_unit: normalize all embeddings to unit vector.
        keepdim: keep dimension size [instance number, hidden_size]
        batch_size: batch size of data in model.
        '''

        self.model = self.model.to(self.device)

        single_instance = False
        if isinstance(phrase[0], str):
            phrase = [phrase]
            single_instance = True

        embedding_list = []

        with torch.no_grad():
            total_batch = len(phrase) // batch_size + (1 if len(phrase) % batch_size > 0 else 0)
            iterator = range(total_batch) if total_batch==1 else tqdm(range(total_batch))
            for batch_id in iterator:

                batch = phrase[batch_id*batch_size:(batch_id+1)*batch_size]

                text_batch = []
                span_batch = []
                for instance in batch:
                    text_batch.append(instance[0])
                    span_batch.append([instance[1]])

                inputs = self.tokenizer(text_batch, entity_spans=span_batch, padding=True, truncation=True, add_prefix_space=True, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                luke_outputs, phrase_repr = self.model(**inputs)
                phrase_repr = phrase_repr.view([len(batch), -1])

                if normalize_to_unit:
                    phrase_repr = phrase_repr / phrase_repr.norm(dim=1, keepdim=True)
                embedding_list.append(phrase_repr.cpu())

        embeddings = torch.cat(embedding_list, 0)

        if single_instance and not keepdim:
            embeddings = embeddings[0]

        if return_numpy and not isinstance(embeddings, ndarray):
            return embeddings.numpy()
        return embeddings

    def _prepare_encode_data(self, sentence_dict, phrase_list):

        data = []
        for phrase in phrase_list:
            doc_id, start, end, span_lemma = phrase
            sentence = sentence_dict[doc_id]

            data.append([sentence, (start, end)])

        return data

    def _get_probs(self, data, batch_size):

        self.model = self.model.to(self.device)

        all_probs = []

        with torch.no_grad():
            total_batch = len(data) // batch_size + (1 if len(data) % batch_size > 0 else 0)
            for batch_id in tqdm(range(total_batch)):

                batch = data[batch_id*batch_size:(batch_id+1)*batch_size]

                text_batch = []
                span_batch = []
                for instance in batch:
                    text_batch.append(instance[0])
                    span_batch.append([instance[1]])

                inputs = self.tokenizer(text_batch, entity_spans=span_batch, padding=True, truncation=True, add_prefix_space=True, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                luke_outputs, phrase_repr = self.model(**inputs)
                phrase_repr = phrase_repr.view([len(batch), -1])
                model_prob = self.model.get_cluster_prob(phrase_repr)

                all_probs.append(model_prob.detach().cpu())

        all_probs = torch.cat(all_probs, dim=0)
        return all_probs

    def _check_data(self, sentences, spans):

        logger.info('Check sentence length and span indices.')
        indices_filted = []
        for idx, sentence in enumerate(sentences):
            tokens = self.tokenizer.tokenize(sentence)
            
            is_valid_span = True
            if spans is not None:
                for span in spans[idx]:
                    if span[0] >= len(sentence) or span[1] > len(sentence):
                        is_valid_span = False

            if len(tokens) <= 510 and is_valid_span:
                indices_filted.append(idx)

        sentences_filted = [sentences[idx] for idx in indices_filted]
        if spans is not None:
            spans_filted = [spans[idx] for idx in indices_filted]
        else:
            spans_filted = spans
        
        number_filted = len(sentences) - len(indices_filted)
        logger.info(f'{number_filted} sentences are filted because of length or invalid span indices.')

        return sentences_filted, spans_filted

    def topic_mining(self, sentences: List,
                        spans: List = None,
                        n_clusters: Union[int, List] = 2,
                        metric: str = "cosine",
                        batch_size: int = 64,
                        max_iter: int = 300,
                        ccl_finetune: bool = True,
                        batch_size_finetune: int = 8,
                        max_finetune_num: int = 100000,
                        finetune_step: int = 2000,
                        contrastive_num: int = 5,
                        positive_ratio: float = 0.1,
                        n_sampling: int = 10000,
                        n_workers: int = 8
                        ):

        '''
        Data arguments:
        sentences: A list of sentences.
        spans: A list of spans corresponding sentences, len(spans)=len(sentences) e.g., [[(0, 9), (5, 7)], [(1, 2)]]. If None, automatically mine phrases from noun chunks.

        Clustering arguments:
        n_clusters: The number of topics. int or List. When n_clusters is a list, n_clusters[0] and n_clusters[1] will be the minimum and maximum numbers to search, n_clusters[2] is the step length.
        meric: The metric to measure the distance between vectors. "cosine" or "euclidean". Default to "cosine".
        batch_size: The size of minibatch for phrase encoding.
        max_iter: Maximum iteration number of kmeans.
        
        CCL-finetune arguments:
        ccl_finetune: Whether to conduct CCL-finetune in the paper.
        batch_size_finetune: The size of minibatch for finetuning.
        max_finetune_num: The maximum number of pairs for finetuning.
        finetune_step: Training steps of finetuning.
        contrastive_num: The number of negatives of contrastive learning.
        positive_ratio: The ratio of the most confident instances for finetuning.
        n_sampling: The number of sampled examples for cluster number confirmation and finetuning.
        n_workers: The number of works for preprocessing data.
        '''

        from .kmeans import get_silhouette_score, get_kmeans
        from .utils import NounPhraser, Lemmatizer, get_rankings
        from .dataloader import get_train_loader
        from .trainer import ClusterLearner

        assert metric in ["cosine", "euclidean"], "metric should be \"cosine\" or \"euclidean\""

        sentences, spans = self._check_data(sentences, spans)

        if spans is None:
            logger.info("Phrase mining from spaCy.")
            sentence_dict, phrase_list = NounPhraser.process_data(sentences, num_workers=n_workers)

        else:

            assert len(sentences) == len(spans), "Sentences and spans do not have the same length."
            sentence_dict, phrase_list = Lemmatizer.process_data(sentences, spans, num_workers=n_workers)


        if n_sampling > 0:

            phrase_list_sampled = random.sample(phrase_list, min(n_sampling, len(phrase_list)))
        
        else:

            phrase_list_sampled = phrase_list

        phrase = self._prepare_encode_data(sentence_dict, phrase_list_sampled)
        phrase_embeddings = self.encode(phrase,
                                        return_numpy=False,
                                        normalize_to_unit=True,
                                        keepdim=True,
                                        batch_size=batch_size)

        if isinstance(n_clusters, int):
            n_clusters = [n_clusters, n_clusters, 1]
        
        if len(n_clusters) == 2: n_clusters.append(1)

        s_scores, prob_scores, center_list = [], [], []
        class_list = []
        for num_class in range(n_clusters[0], n_clusters[1]+1, n_clusters[2]):

            s_score, p_score, centers = get_silhouette_score(phrase_embeddings, n_clusters=num_class, max_iter=max_iter)
            class_list.append(num_class)
            s_scores.append(s_score)
            prob_scores.append(p_score)
            center_list.append(centers)
            logger.info(f'When n_clusters = {num_class}, silhouette_score is: {s_score}')


        max_index = np.argmax(s_scores)
        s_score = s_scores[max_index]
        probs = prob_scores[max_index]
        centers = center_list[max_index]
        num_class = class_list[max_index]

        print(f'The number of topics: {num_class}, maximum silhouette_score is: {s_score}.')

        if ccl_finetune:

            # if len(sentences) < 1000:
            #     logger.warning("We do not recommend finetuning on a small dataset.")

            rankings = get_rankings(probs, positive_ratio=positive_ratio)

            pseudo_label_dict = defaultdict(list)

            for i in range(len(rankings)):
                for j in range(len(rankings[i])):
                    pseudo_label_dict[phrase_list_sampled[rankings[i][j]][-1]].append(j)

            ## majority vote
            for phrase, predictions in pseudo_label_dict.items():
                pseudo_label_dict[phrase] = Counter(predictions).most_common()[0][0]

            self.model.update_cluster_centers(centers)

            train_loader = get_train_loader(sentence_dict=sentence_dict, 
                                            phrase_list_sampled=phrase_list_sampled, 
                                            pseudo_label_dict=pseudo_label_dict, 
                                            tokenizer=self.tokenizer, 
                                            num_workers=n_workers,
                                            max_finetune_num=max_finetune_num,
                                            contrastive_num=contrastive_num,
                                            batch_size=batch_size_finetune)

            # optimizer 
            optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)

            # set up logger
            global_step = 0
            epoch = 0
            # set up the trainer
            learner = ClusterLearner(self.model, optimizer)
            self.model.train()

            ret = False

            while True:
                tqdm_dataloader = tqdm(train_loader, total=finetune_step)
                for features in tqdm_dataloader:

                    for feature in features:
                        for k, v in feature.items():
                            feature[k] = v.to(self.device)

                    loss = learner.forward(features)

                    tqdm_dataloader.set_description(
                        'Epoch{}, Global Step {}, CL-loss {:.5f}'.format(
                            epoch, global_step,  loss['Instance-CL_loss']
                            ))
                    global_step+=1
                    if global_step >= finetune_step:
                        ret = True
                        break
                epoch+=1
                if ret:
                    break

            self.model.eval()
            logger.info("Predicting all phrases.")
            all_phrase = self._prepare_encode_data(sentence_dict, phrase_list)
            all_probs = self._get_probs(all_phrase, batch_size)

            all_probs = all_probs.numpy()

            self.centers = self.model.cluster_centers.detach().cpu().numpy()

        else:

            if n_sampling > 0:
                
                logger.info("Encode all phrases.")
                all_phrase = self._prepare_encode_data(sentence_dict, phrase_list)
                phrase_embeddings = self.encode(all_phrase,
                                        return_numpy=False,
                                        normalize_to_unit=True,
                                        keepdim=True,
                                        batch_size=batch_size)
                logger.info("Clustering.")
                p_score, centers = get_kmeans(phrase_embeddings, n_clusters=num_class, max_iter=max_iter)

                all_probs = p_score.numpy()
                self.centers = centers

                logger.info("Done.")

            else:

                all_probs = probs.numpy()
                self.centers = centers

        
        assert len(phrase_list) == len(all_probs)

        merge_phrase_dict = defaultdict(list)
        topic_phrase_dict = defaultdict(list)

        for prob, data_line in zip(all_probs, phrase_list):

            merge_phrase_dict[data_line[-1]].append(prob)

        for phrase, prob_list in merge_phrase_dict.items():
            prob_mean = np.array(prob_list).mean(axis=0)
            pred = prob_mean.argmax()
            merge_phrase_dict[phrase] = str(pred)
            topic_phrase_dict[str(pred)].append((phrase, prob_mean[pred]))

        for topic, v in topic_phrase_dict.items():

            topic_phrase_dict[topic] = [(line[0], str(round(line[1], 4))) for line in sorted(v, key=lambda x: x[1], reverse=True)]


        doc_id_phrase_pred = defaultdict(list)
        for phrase in phrase_list:

            doc_id, start, end, span_lemma = phrase
            doc_id_phrase_pred[doc_id].append([start, end, merge_phrase_dict[span_lemma]])
        
        output_data = []

        for doc_id, sentence in sentence_dict.items():
            if doc_id in doc_id_phrase_pred:
                output_data.append([sentence, doc_id_phrase_pred[doc_id]])

        self.merge_phrase_dict = merge_phrase_dict
        self.do_topic_mining = True

        return output_data, topic_phrase_dict

    def save(self, path: str):
        '''
        Save model parameters and clustering centers.
        path: the directory of the saved files.
        '''
        if not self.do_topic_mining:
            assert 0, 'Please run topic_mining before save the model.'

        if not os.path.isdir(path):
            assert 0, f'{path} is not an existing directory.'

        save_dict = {'uctopic': self.model.state_dict(),
                    'cluster_centers': self.centers,
                    'phrase_dict': self.merge_phrase_dict
                    }

        torch.save(save_dict, os.path.join(path, 'uctopic_model.bin'))

    def load(self, path: str):
        '''
        Load model parameters and centers from topic modeling from checkpoint.
        path: the directory of the saved files.
        '''

        if not os.path.isdir(path):
            assert 0, f'{path} is not an existing directory.'

        save_dict = torch.load(os.path.join(path, 'uctopic_model.bin'))

        self.centers = save_dict['cluster_centers']
        if 'cluster_centers' in save_dict['uctopic']:
            del save_dict['uctopic']['cluster_centers']
        self.model.load_state_dict(save_dict['uctopic'])
        self.model.update_cluster_centers(self.centers)

        self.merge_phrase_dict = save_dict['phrase_dict']

        self.do_topic_mining = True

        logger.info('Load checkpoint successfully.')

    
    def similarity(self, queries: List, keys: Union[List, ndarray], batch_size: int = 64) -> Union[float, ndarray]:
        
        query_vecs = self.encode(queries, return_numpy=True, batch_size=batch_size) # suppose N queries
        
        if not isinstance(keys, ndarray):
            key_vecs = self.encode(keys, return_numpy=True, batch_size=batch_size) # suppose M keys
        else:
            key_vecs = keys

        # check whether N == 1 or M == 1
        single_query, single_key = query_vecs.shape[0] == 1, key_vecs.shape[0] == 1 
        if single_query:
            query_vecs = query_vecs.reshape(1, -1)
        if single_key:
            key_vecs = key_vecs.reshape(1, -1)
        
        # returns an N*M similarity array
        similarities = cosine_similarity(query_vecs, key_vecs)
        
        if single_query:
            similarities = similarities[0]
            if single_key:
                similarities = float(similarities[0])
        
        return similarities

    
    def build_index(self, phrases: List, 
                        use_faiss: bool = None,
                        faiss_fast: bool = False,
                        batch_size: int = 64):

        if use_faiss is None or use_faiss:
            try:
                import faiss
                assert hasattr(faiss, "IndexFlatIP")
                use_faiss = True 
            except:
                logger.warning("Fail to import faiss. If you want to use faiss, install faiss through PyPI. Now the program continues with brute force search.")
                use_faiss = False
        
        logger.info("Encoding embeddings for sentences...")
        embeddings = self.encode(phrases, batch_size=batch_size, normalize_to_unit=True, return_numpy=True)

        logger.info("Building index...")
        self.index = {"phrases": phrases}
        
        if use_faiss:
            quantizer = faiss.IndexFlatIP(embeddings.shape[1])  
            if faiss_fast:
                index = faiss.IndexIVFFlat(quantizer, embeddings.shape[1], min(self.num_cells, len(phrases))) 
            else:
                index = quantizer

            if self.device_type == "cuda":
                if hasattr(faiss, "StandardGpuResources"):
                    logger.info("Use GPU-version faiss")
                    res = faiss.StandardGpuResources()
                    res.setTempMemory(20 * 1024 * 1024 * 1024)
                    index = faiss.index_cpu_to_gpu(res, self.device_num, index)
                else:
                    logger.info("Use CPU-version faiss")
            else: 
                logger.info("Use CPU-version faiss")

            if faiss_fast:            
                index.train(embeddings.astype(np.float32))
            index.add(embeddings.astype(np.float32))
            index.nprobe = min(self.num_cells_in_search, len(phrases))
            self.is_faiss_index = True
        else:
            index = embeddings
            self.is_faiss_index = False
        self.index["index"] = index
        logger.info("Finished")


    def search(self, queries: List, top_k: int = 5):
        
        if not self.is_faiss_index:
            
            combined_results = []
            similarities = self.similarity(queries, self.index["index"]).tolist()

            single_query = False
            if isinstance(queries[0], str):
                similarities = [similarities]
                single_query = True

            for line in similarities:
                id_and_score = []
                for i, s in enumerate(line):
                    id_and_score.append((i, s))
                id_and_score = sorted(id_and_score, key=lambda x: x[1], reverse=True)[:top_k]
                results = [[self.index["phrases"][idx], score] for idx, score in id_and_score]
                combined_results.append(results)
            
            if single_query:
                return combined_results[0]
            else:
                return combined_results
        else:
            query_vecs = self.encode(queries,normalize_to_unit=True, keepdim=True, return_numpy=True)

            distance, idx = self.index["index"].search(query_vecs.astype(np.float32), top_k)
            
            def pack_single_result(dist, idx):
                results = [[self.index["phrases"][i], s] for i, s in zip(idx, dist)]
                return results
            
            if isinstance(queries[0], list):
                combined_results = []
                for i in range(len(queries)):
                    results = pack_single_result(distance[i], idx[i])
                    combined_results.append(results)
                return combined_results
            else:
                return pack_single_result(distance[0], idx[0])

    
    def predict_topic(self, phrases: List) -> List:

        if not self.do_topic_mining:
            assert 0, 'Please run topic_mining before predict the topic.'

        from .utils import Lemmatizer

        if isinstance(phrases[0], str):
            phrases = [phrases]

        topics = []

        for line in phrases:

            text, span = line
            span_lemma = Lemmatizer.normalize(text[span[0]:span[1]])

            if span_lemma in self.merge_phrase_dict:
                topics.append(int(self.merge_phrase_dict[span_lemma]))
            else:
                similarities = self.similarity(line, self.centers)
                pred = similarities.argmax(axis=-1)
                topics.append(pred)

        return topics


# if __name__=="__main__":


#     model_name = 'JiachengLi/uctopic-base'
#     ## Encoding test
#     phrases = [
#         ["We came for a birthday brunch and this place is so much bigger than it looks from the outside!", (3, 7)],
#         ["It was totally packed and loud.", (15, 21)],
#         ["Service was on the slower side.", (0, 7)],
#         ["I ordered 2 mojitos: 1 lime and 1 mango.", (12, 19)],
#         ["The ingredient weren\u2019t really fresh.", (4, 14)]
#     ]

#     tool = UCTopicTool(model_name, device='cuda:5')
#     # embeddings = tool.encode(phrases, keepdim=False)
#     # print(embeddings.shape)
    
    
#     import json
#     sentences = []
#     with open('data/topic_data/google_restaurant.json') as f:
#         for line in f:
#             line = json.loads(line)
#             sentences.append(line["text"])

#     sentences = sentences[:100000]

#     output_data, topic_phrase_dict = tool.topic_mining(sentences, n_clusters=[18, 28], ccl_finetune=True, n_workers=8, n_sampling=10000, batch_size_finetune=4, finetune_step=1000)

#     with open('tocal_phrase_1.json', 'w') as fout:

#         json.dump(topic_phrase_dict, fout)

#     print(output_data[:20])