import argparse
import torch
from transformers import AutoTokenizer
from nltk import WordNetLemmatizer

def get_device(gpu):
	return torch.device('cpu' if gpu is None else f'cuda:{gpu}')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default=None)
    parser.add_argument("--data_path", type=str, default='data/mitmovie/all_data.json')
    parser.add_argument("--save_path", type=str, default='clustering_results/')
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--max_training_examples", type=int, default=100000)
    parser.add_argument("--steps_per_eval", type=int, default=50)
    parser.add_argument("--preprocessing_num_workers", type=int, default=4)
    parser.add_argument("--use_luke", action='store_true')
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lr_scale", type=int, default=1)
    parser.add_argument("--temp", type=float, default=0.05)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--use_perturbation', action='store_true', help="")
    args = parser.parse_args()
    return args

ARGS = parse_args()
DEVICE = get_device(ARGS.gpu)
TOKENIZER = AutoTokenizer.from_pretrained('studio-ousia/luke-base')
LEMMATIZER = WordNetLemmatizer()


