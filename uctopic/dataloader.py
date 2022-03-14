import copy
import collections
import random
from tqdm import tqdm
import torch
from typing import Dict, List, Optional
import torch.utils.data as util_data
from itertools import combinations
from torch.utils.data import Dataset
from multiprocessing import Pool
from .tokenizer import UCTopicTokenizer


TOKENIZER = None

class ContrastClusteringDataset(Dataset):
    
    def __init__(self, sentence_dict: Dict, 
                phrase_list_sampled: List, 
                pseudo_label_dict: Dict, 
                tokenizer: UCTopicTokenizer, 
                num_workers: int,
                max_finetune_num: int,
                contrastive_num: int
                ):

        global TOKENIZER
        TOKENIZER = tokenizer

        self.pseudo_label_dict = pseudo_label_dict
        self.contrastive_num = contrastive_num

        data_dict = dict()
        for phrase_instance in phrase_list_sampled:
            doc_id, start, end, phrase_lemma = phrase_instance
            if doc_id in data_dict:
                data_dict[doc_id]['phrases'].append([phrase_lemma, start, end])

            else:
                data_dict[doc_id] = {'text': sentence_dict[doc_id], 'phrases':[[phrase_lemma, start, end]]}

        data = list(data_dict.values())
        
        pool = Pool(processes=num_workers)
        pool_func = pool.imap(func=ContrastClusteringDataset._par_tokenize_doc, iterable=data)
        doc_tuples = list(tqdm(pool_func, total=len(data), desc=f'[Tokenize]'))
        self.tokenized_corpus = [tokenized_sent for tokenized_sent, entity_ids in doc_tuples]
        doc_entity_list = [entity_ids for tokenized_sent, entity_ids in doc_tuples]
        pool.close()
        pool.join()

        self.label_instance_dict = collections.defaultdict(list)
        entity_position_dict = collections.defaultdict(list)
        for sent_idx, sent_entity_list in tqdm(enumerate(doc_entity_list), desc='Extract entity positions'):
            for entity_idx, entity in enumerate(sent_entity_list):

                if entity in pseudo_label_dict:
                    entity_position_dict[entity].append((sent_idx, entity_idx, entity))
                    self.label_instance_dict[pseudo_label_dict[entity]].append((sent_idx, entity_idx, entity))

        self.label_list = list(self.label_instance_dict.keys())

        entity_position_filter_dict = dict()
        for key, value in entity_position_dict.items():
            if len(value) >= 2:  ## have at least one pair
                entity_position_filter_dict[key] = value

        pool = Pool(processes=num_workers)
        pool_func = pool.imap(func=ContrastClusteringDataset._par_sample_pairs, iterable=entity_position_filter_dict.values())
        pair_tuples = list(tqdm(pool_func, total=len(entity_position_filter_dict), desc=f'Pairing....'))
        self.entity_pairs = []
        for sent_entity_pairs in pair_tuples:
            self.entity_pairs += sent_entity_pairs
        pool.close()
        pool.join()
        print(f'Toal number of pairs: {len(self.entity_pairs)}')

        pair_index = list(range(len(self.entity_pairs)))
        random.shuffle(pair_index)
        pair_index = pair_index[:max_finetune_num]
        self.entity_pairs = [self.entity_pairs[idx] for idx in pair_index]

        print(f'The number of sampled pairs: {len(self.entity_pairs)}')

    def __len__(self):
        return len(self.entity_pairs)

    def __getitem__(self, index):
        
        return self.entity_pairs[index]

    def collate_fn(self, samples):

        anchor_feature, cl_feature = self.extract_features(samples)
        
        anchor_batch = TOKENIZER.pad(
            anchor_feature,
            padding=True,
            return_tensors="pt",
        )

        cl_batch = TOKENIZER.pad(
            cl_feature,
            padding=True,
            return_tensors="pt",
        )

        anchor_batch['input_ids'], cl_batch['input_ids'] = self.mask_entity(anchor_feature, cl_feature)

        return anchor_batch, cl_batch

    def extract_features(self, features):

        entity_features_name = ['entity_ids', 'entity_position_ids', 'entity_attention_mask']

        anchor_feature = []
        cl_feature = []

        for feature in features:
            
            entity1, entity2 = feature
            sent_idx1, entity_idx1, entity_id1 = entity1
            sent_idx2, entity_idx2, entity_id2 = entity2
            ## sample negative instances
            assert entity_id1 == entity_id2
            label = self.pseudo_label_dict[entity_id1]

            entity_list = [entity1, entity2]

            neg_list = []
            while len(neg_list) < min(self.contrastive_num, len(self.label_list)):
                neg_label = random.choice(self.label_list)
                while neg_label == label:
                    neg_label = random.choice(self.label_list)
                neg_list.append(neg_label)

            for neg_label in neg_list:
                
                neg_entity = random.choice(self.label_instance_dict[neg_label])
                entity_list.append(neg_entity)

            for idx, entity in enumerate(entity_list):
                sent_idx, entity_idx, entity_id = entity
                sent_feature = copy.deepcopy(self.tokenized_corpus[sent_idx])
                for name in entity_features_name:
                    sent_feature[name] = [sent_feature[name][entity_idx]]

                if idx == 0: ## anchor
                    anchor_feature.append(sent_feature)

                else:
                    cl_feature.append(sent_feature)

        return anchor_feature, cl_feature

    def mask_entity(self, anchor_feature, cl_feature):

        anchor_input_ids = []
        cl_input_ids = []

        for feature in anchor_feature:
            anchor_input_ids.append(self._mask_entity(feature, prob=1))

        for feature in cl_feature:
            cl_input_ids.append(self._mask_entity(feature, prob=0.5))

        return self._collate_batch(anchor_input_ids), self._collate_batch(cl_input_ids)

    
    def _mask_entity(self, feature, prob=0.5):

        input_ids = feature['input_ids']
        if random.random() < prob:
            position_ids = feature['entity_position_ids'][0]

            for pos in position_ids:
                if pos == -1:
                    break

                input_ids[pos] = TOKENIZER.convert_tokens_to_ids(TOKENIZER.mask_token)

            return input_ids

        else:
            return input_ids

    def _collate_batch(self, examples, pad_to_multiple_of: Optional[int] = None):
        """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
        # Tensorize if necessary.
        if isinstance(examples[0], (list, tuple)):
            examples = [torch.tensor(e, dtype=torch.long) for e in examples]

        # Check if padding is necessary.
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
            return torch.stack(examples, dim=0)

        # If yes, check if we have a `pad_token`.
        if TOKENIZER._pad_token is None:
            raise ValueError(
                "You are attempting to pad samples but the tokenizer you are using"
                f" ({TOKENIZER.__class__.__name__}) does not have a pad token."
            )

        # Creating the full tensor and filling it with our data.
        max_length = max(x.size(0) for x in examples)
        if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
        result = examples[0].new_full([len(examples), max_length], TOKENIZER.pad_token_id)
        for i, example in enumerate(examples):
            if TOKENIZER.padding_side == "right":
                result[i, : example.shape[0]] = example
            else:
                result[i, -example.shape[0] :] = example
        return result


    @staticmethod
    def _par_tokenize_doc(doc):
        
        sentence = doc['text']
        phrases = doc['phrases']
        phrase_spans = [(phrase[1], phrase[2]) for phrase in phrases]
        entity_ids = [phrase[0] for phrase in phrases]
        tokenized_sent = TOKENIZER(sentence,
                                    entity_spans=phrase_spans,
                                    add_prefix_space=True)

        return dict(tokenized_sent), entity_ids
    @staticmethod
    def _par_sample_pairs(entity_list):
    
        return list(combinations(entity_list, 2))


def get_train_loader(sentence_dict: Dict, 
                    phrase_list_sampled: List, 
                    pseudo_label_dict: Dict, 
                    tokenizer: UCTopicTokenizer, 
                    num_workers: int,
                    max_finetune_num: int,
                    contrastive_num: int,
                    batch_size: int):
    '''
    pseudo_label_dict: phrase lemma --> clustering results
    '''

    train_dataset = ContrastClusteringDataset(sentence_dict=sentence_dict, 
                                            phrase_list_sampled=phrase_list_sampled, 
                                            pseudo_label_dict = pseudo_label_dict, 
                                            tokenizer = tokenizer, 
                                            num_workers = num_workers,
                                            max_finetune_num = max_finetune_num,
                                            contrastive_num = contrastive_num)
    train_loader = util_data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)

    return train_loader
        