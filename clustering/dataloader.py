import json
import copy
import collections
import random
from tqdm import tqdm
import torch
from typing import Optional, Union, List, Dict, Tuple
import torch.utils.data as util_data
from itertools import combinations
from torch.utils.data import Dataset
from consts import TOKENIZER, LEMMATIZER
from multiprocessing import Pool

def data_interface(data_line):
    '''
    Convert different dataset to identical format.
    '''
    sentence = data_line['sentences'][0]
    phrases = data_line['ner'][0]

    phrase_name_list = []
    phrase_lemma_list=  []

    for phrase in phrases:
        start, end, _ = phrase

        phrase_name = ' '.join(sentence[start:end+1])
        phrase_name_list.append(phrase_name)

        phrase_lemma = ' '.join([LEMMATIZER.lemmatize(word) for word in sentence[start:end+1]])
        phrase_lemma_list.append(phrase_lemma)

    text = ' '.join(sentence)

    phrases = []
    for phrase, phrase_lemma in zip(phrase_name_list, phrase_lemma_list):

        start = text.find(phrase)
        end = start + len(phrase)
        phrases.append([phrase, phrase_lemma, start, end])

    doc = dict()

    doc['text'] = text
    doc['phrases'] = phrases
    return doc

class ContrastClusteringDataset(Dataset):
    
    def __init__(self, data_path, data_args):
        dataset = []
        with open(data_path, encoding='utf8') as f:
            for line in tqdm(f, ncols=100, desc='Reading dataset...'):
                line = json.loads(line)
                dataset.append(data_interface(line))

        pool = Pool(processes=data_args.preprocessing_num_workers)
        pool_func = pool.imap(func=ContrastClusteringDataset._par_tokenize_doc, iterable=dataset)
        doc_tuples = list(tqdm(pool_func, total=len(dataset), ncols=100, desc=f'[Tokenize]'))
        self.tokenized_corpus = [tokenized_sent for tokenized_sent, entity_ids in doc_tuples]
        doc_entity_list = [entity_ids for tokenized_sent, entity_ids in doc_tuples]
        pool.close()
        pool.join()

        entity_position_dict = collections.defaultdict(list)
        for sent_idx, sent_entity_list in tqdm(enumerate(doc_entity_list), ncols=100, desc='Extract entity positions'):
            for entity_idx, entity in enumerate(sent_entity_list):
                entity_position_dict[entity].append((sent_idx, entity_idx))

        entity_position_filter_dict = dict()
        for key, value in entity_position_dict.items():
            if len(value) >= 2:  ## have at least one pair
                entity_position_filter_dict[key] = value

        pool = Pool(processes=data_args.preprocessing_num_workers)
        pool_func = pool.imap(func=ContrastClusteringDataset._par_sample_pairs, iterable=entity_position_filter_dict.values())
        pair_tuples = list(tqdm(pool_func, total=len(entity_position_filter_dict), ncols=100, desc=f'Pairing....'))
        self.entity_pairs = []
        for sent_entity_pairs in pair_tuples:
            self.entity_pairs += sent_entity_pairs
        pool.close()
        pool.join()
        print(f'Toal number of pairs: {len(self.entity_pairs)}')

    def __len__(self):
        return len(self.entity_pairs)

    def __getitem__(self, index):
        
        return self.entity_pairs[index]

    def collate_fn(self, samples):

        entity_feature1, entity_feature2 = self.extract_features(samples)
        
        batch1 = TOKENIZER.pad(
            entity_feature1,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch2 = TOKENIZER.pad(
            entity_feature2,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch0 = copy.deepcopy(batch1)
        batch1['input_ids'], batch2['input_ids'] = self.mask_entity(entity_feature1, entity_feature2)

        return [batch0, batch1, batch2]

    def extract_features(self, features):

        entity_features_name = ['entity_ids', 'entity_position_ids', 'entity_attention_mask']

        entity_feature1 = []
        entity_feature2 = []

        for feature in features:
            
            entity1, entity2 = feature['entity_pairs']
            sent_idx1, entity_idx1 = entity1
            sent_idx2, entity_idx2 = entity2

            sent_feature1 = copy.deepcopy(self.tokenized_corpus[sent_idx1])
            sent_feature2 = copy.deepcopy(self.tokenized_corpus[sent_idx2])

            for name in entity_features_name:
                sent_feature1[name] = [sent_feature1[name][entity_idx1]]
                sent_feature2[name] = [sent_feature2[name][entity_idx2]]

            entity_feature1.append(sent_feature1)
            entity_feature2.append(sent_feature2)

        return entity_feature1, entity_feature2

    def mask_entity(self, entity_feature1, entity_feature2):

        masked_input_ids_1 = []
        masked_input_ids_2 = []
        for feature1, feature2 in zip(entity_feature1, entity_feature2):
            
            masked_input_ids_1.append(self._mask_entity(feature1, prob=1))
            masked_input_ids_2.append(self._mask_entity(feature2, prob=0.5))

        return self._collate_batch(masked_input_ids_1), self._collate_batch(masked_input_ids_2)

    
    def _mask_entity(self, feature, prob=0.5):

        input_ids = feature['input_ids']
        if random.random() < prob:
            position_ids = feature['entity_position_ids'][0]

            for pos in position_ids:
                if pos == -1:
                    break

                input_ids[pos] = TOKENIZER.convert_tokens_to_ids(TOKENIZER.mask_token)

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
        entities = doc['phrases']

        entity_spans = [(entity[2], entity[3]) for entity in entities]
        entity_ids = [entity[1] for entity in entities]
        tokenized_sent = TOKENIZER(sentence,
                                    entity_spans=entity_spans,
                                    add_prefix_space=True)

        return dict(tokenized_sent), entity_ids
    @staticmethod
    def _par_sample_pairs(entity_list):
    
        return list(combinations(entity_list, 2))


def get_train_loader(args):

    train_dataset = ContrastClusteringDataset(args.data_path, args)
    train_loader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    return train_loader
        