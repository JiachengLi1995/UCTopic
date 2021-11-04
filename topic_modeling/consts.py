import torch
import spacy
import argparse
from transformers import AutoTokenizer
from nltk import WordNetLemmatizer


def get_device(gpu):
	return torch.device('cpu' if gpu is None else f'cuda:{gpu}')

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--gpu", type=int, default=None)
	parser.add_argument("--data_path", type=str, default='data/topic_data/google_restaurant.json')
	parser.add_argument("--save_path", type=str, default='topic_results')
	parser.add_argument("--num_classes", type=str, default='[15, 30]', help='Min and Max number of classes.')
	parser.add_argument("--sample_num_cluster", type=int, default=5000)
	parser.add_argument("--sample_num_finetune", type=int, default=5000)
	parser.add_argument("--contrastive_num", type=int, default=5)
	parser.add_argument("--num_workers", type=int, default=8)
	parser.add_argument("--epoch", type=int, default=10)
	parser.add_argument("--max_length", type=int, default=128)
	parser.add_argument("--batch_size", type=int, default=16)
	parser.add_argument("--lr", type=float, default=1e-5)
	parser.add_argument("--temp", type=float, default=0.05)
	parser.add_argument('--alpha', type=float, default=1.0)
	args = parser.parse_args()
	return args

ARGS = parse_args()
DEVICE = get_device(ARGS.gpu)

def get_device(gpu):
	return torch.device('cpu' if gpu is None else f'cuda:{gpu}')

TOKENIZER = AutoTokenizer.from_pretrained('studio-ousia/luke-base')
LEMMATIZER = WordNetLemmatizer()
NLP = spacy.load('en_core_web_sm', disable=['ner'])

