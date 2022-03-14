import os
import torch
import spacy
import argparse
from . import UCTopicTokenizer
from nltk import WordNetLemmatizer


def get_device(gpu):
	return torch.device('cpu' if gpu is None else f'cuda:{gpu}')

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--gpu", type=int, default=None)
	parser.add_argument("--data_path", type=str, default='data/topic_data/')
	parser.add_argument("--dataset", type=str, default='google_restaurant')
	parser.add_argument("--save_path", type=str, default='topic_results')
	parser.add_argument("--num_classes", type=str, default='[10, 25]', help='Min and Max number of classes.')
	parser.add_argument("--sample_num_cluster", type=int, default=5000)
	parser.add_argument("--sample_num_finetune", type=int, default=100000)
	parser.add_argument("--contrastive_num", type=int, default=10)
	parser.add_argument("--finetune_step", type=int, default=2000)
	parser.add_argument("--num_workers", type=int, default=8)
	parser.add_argument("--epoch", type=int, default=10)
	parser.add_argument("--max_length", type=int, default=32)
	parser.add_argument("--batch_size", type=int, default=16)
	parser.add_argument("--lr", type=float, default=1e-5)
	parser.add_argument("--temp", type=float, default=0.05)
	parser.add_argument('--alpha', type=float, default=1.0)
	args = parser.parse_args()
	return args

ARGS = parse_args()
ARGS.data_path = os.path.join(ARGS.data_path, ARGS.dataset+'.json')
DEVICE = get_device(ARGS.gpu)

def get_device(gpu):
	return torch.device('cpu' if gpu is None else f'cuda:{gpu}')

TOKENIZER = UCTopicTokenizer.from_pretrained('studio-ousia/luke-base')
LEMMATIZER = WordNetLemmatizer()
NLP = spacy.load('en_core_web_sm', disable=['ner'])


