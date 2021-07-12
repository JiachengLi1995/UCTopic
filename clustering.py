import torch
import torch.nn as nn
import argparse
import os
from tqdm import tqdm
from transformers.models import luke
from clustering.utils import conll2003_reader, get_device, batchify
from clustering.kmeans import get_kmeans
from uctopic.models import UCTopicConfig, UCTopic
from transformers import LukeTokenizer, LukeModel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default=None)
    parser.add_argument("--data_path", type=str, default='data/conll2003/')
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--use_luke", action='store_true')
    args = parser.parse_args()
    return args

ARGS = parse_args()
DEVICE = get_device(ARGS.gpu)

def get_features(data, tokenizer, model):

    all_features = []
    all_labels = []

    for batch in tqdm(batchify(data), ncols=100, desc='Generate all features...'):

        text_batch, span_batch, label_batch = batch

        inputs = tokenizer(text_batch, entity_spans=span_batch, padding=True, add_prefix_space=True, return_tensors="pt")

        for k,v in inputs.items():
            inputs[k] = v.to(DEVICE)

        luke_outputs, entity_pooling = model(**inputs)
        if ARGS.use_luke:
            all_features.append(luke_outputs.entity_last_hidden_state.squeeze().detach().cpu())
        else:
            all_features.append(entity_pooling.squeeze().detach().cpu())

        all_labels += label_batch

    all_features = torch.cat(all_features, dim=0).numpy()
    all_labels = torch.LongTensor(all_labels)

    return all_features, all_labels

class ProjectHead(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()

        self.dense1 = nn.Linear(hidden_size, hidden_size)
        self.dense2 = nn.Linear(hidden_size, output_size)
        self.activation = nn.GELU()

    def forward(self, features):

        x = self.dense1(features)
        x = self.activation(x)
        x = self.dense2(x)

        return x

def main():

    config = UCTopicConfig.from_pretrained("studio-ousia/luke-base")
    tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-base")
    model = UCTopic(config)
    model.load_state_dict(torch.load('result/pytorch_model.bin'))
    if ARGS.use_luke:
        print('Using LUKE pre-trained model.')
        model.luke = LukeModel.from_pretrained("studio-ousia/luke-base")
    model.to(DEVICE)
    model.eval()

    train_path = os.path.join(ARGS.data_path, 'train.json')
    dev_path = os.path.join(ARGS.data_path, 'dev.json')
    test_path = os.path.join(ARGS.data_path, 'test.json')
    train_data = conll2003_reader(train_path)
    dev_data = conll2003_reader(dev_path)
    test_data = conll2003_reader(test_path)

    features, labels = get_features(train_data, tokenizer, model)
    score_factor, score_cosine = get_kmeans(features, labels, ARGS.num_classes)
 
if __name__ == '__main__':

    main()