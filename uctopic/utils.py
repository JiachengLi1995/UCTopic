from typing import List
import spacy
from numpy import ndarray
from tqdm import tqdm
from multiprocessing import Pool

try:
    NLP = spacy.load('en_core_web_sm', disable=['ner', 'token2vec'])
except:
    import os
    os.system("python -m spacy download en_core_web_sm")
    NLP = spacy.load('en_core_web_sm', disable=['ner', 'token2vec'])

class NounPhraser:
    @staticmethod
    def process_data(data: List,
                    num_workers: int = 8):

        sentence_dict = dict()
        phrase_list = []
        for doc_id, sentence in enumerate(data):
            sentence_dict[doc_id] = sentence

        pool = Pool(processes=num_workers)
        pool_func = pool.imap(func=NounPhraser.rule_based_noun_phrase, iterable=sentence_dict.items())
        doc_tuples = list(tqdm(pool_func, total=len(sentence_dict), desc=f'Extract phrases'))
        for phrases in doc_tuples:
            phrase_list += phrases
        pool.close()
        pool.join()

        return sentence_dict, phrase_list

    @staticmethod    
    def rule_based_noun_phrase(line):

        definite_articles = {'a', 'the', 'an', 'this', 'those', 'that', 'these', \
                                  'my', 'his', 'her', 'your', 'their', 'our'}
        doc_id, text = line
        if not text:
            return []
        doc = NLP(text)
        
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

                        span_lemma = ' '.join([doc[i].lemma_.lower().strip() for i in range(start+1, end)])
                        assert doc.text[start_char:chunk.end_char] == new_ps
                        phrases.append((doc_id, start_char, chunk.end_char, span_lemma))

                    else:
                        span_lemma = ' '.join([doc[i].lemma_.lower().strip() for i in range(start, end)])
                        phrases.append((doc_id, chunk.start_char, chunk.end_char, span_lemma))

            else:
                if doc[chunk.start].pos_ != 'PRON':
                    span_lemma = ' '.join([doc[i].lemma_.lower().strip() for i in range(start, end)])
                    phrases.append((doc_id, chunk.start_char, chunk.end_char, span_lemma))

        return phrases


class Lemmatizer:
    @staticmethod
    def process_data(sentences: List,
                    spans: List,
                    num_workers: int = 8):

        instance_dict = dict()
        phrase_list = []
        for doc_id, instance in enumerate(zip(sentences, spans)):
            instance_dict[doc_id] = instance

        pool = Pool(processes=num_workers)
        pool_func = pool.imap(func=Lemmatizer._process, iterable=instance_dict.items())
        doc_tuples = list(tqdm(pool_func, total=len(instance_dict), desc=f'Normalize phrases'))
        for phrases in doc_tuples:
            phrase_list += phrases
        pool.close()
        pool.join()

        sentence_dict = dict()
        for doc_id, instance in instance_dict.items():

            sentence = instance[0]
            sentence_dict[doc_id] = sentence


        return sentence_dict, phrase_list

    @staticmethod
    def _process(line):
        
        doc_id, (sentence, spans) = line

        phrases = []
        for span in spans:
            phrase = sentence[span[0]: span[1]]
            span_lemma = Lemmatizer.normalize(phrase)
            phrases.append((doc_id, span[0], span[1], span_lemma))

        return phrases

    @staticmethod
    def normalize(text):
        doc = NLP(text)
        return ' '.join([token.lemma_.lower().strip() for token in doc])


def get_rankings(scores: ndarray, positive_ratio: float = 0.8):
	'''
	scores: (samples, class_num)
	'''
	class_num = scores.shape[-1]
	rankings = (-scores).argsort(axis=0) #(samples, class_num)
	rankings = rankings[:int(len(rankings) * 1.0 / class_num * positive_ratio)]

	return rankings
