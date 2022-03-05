import torch
import argparse
import os
from tqdm import tqdm
from clustering.utils import dataset_reader, get_device, batchify
from clustering.kmeans import get_kmeans
from uctopic.models import UCTopicConfig, UCTopic
from uctopic.tokenizer import UCTopicTokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--data_path", type=str, default='data/mitmovie/')
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--use_luke", action='store_true')
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--temp", type=float, default=0.05)
    parser.add_argument("--negative_numbers", type=int, default=10)
    parser.add_argument("--in_batch", action='store_true')
    args = parser.parse_args()
    return args

ARGS = parse_args()
DEVICE = get_device(ARGS.gpu)

def get_features(data, tokenizer, model):

    all_features = []
    all_labels = []

    with torch.no_grad():

        for batch in tqdm(batchify(data, ARGS.batch_size), ncols=100, desc='Generate all features...'):

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

    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.LongTensor(all_labels)

    return all_features, all_labels

def main():

    config = UCTopicConfig.from_pretrained("studio-ousia/luke-base")
    tokenizer = UCTopicTokenizer.from_pretrained("studio-ousia/luke-base")
    model = UCTopic(config)
    model.load_state_dict(torch.load('result/pytorch_model.bin'))
    model.to(DEVICE)
    model.eval()

    train_path = os.path.join(ARGS.data_path, 'train.json')
    dev_path = os.path.join(ARGS.data_path, 'dev.json')
    test_path = os.path.join(ARGS.data_path, 'test.json')

    if 'conll2003' in ARGS.data_path:
        label_dict = {'PER':0, 'LOC':1, 'ORG':2}
    elif 'bc5cdr' in ARGS.data_path:
        label_dict = {'Chemical': 0, 'Disease': 1}
    elif 'mitmovie' in ARGS.data_path:
        label_dict = {'person': 0, 'title': 1}
    elif 'wnut2017' in ARGS.data_path:
        label_dict = {'corporation': 0, 'creative_work':1, 'group': 2,
                      'location': 3, 'person': 4, 'product': 5}
    else:
        raise NotImplementedError

    ARGS.num_classes = len(label_dict)

    train_data = dataset_reader(train_path, label_dict, token_level=False)
    dev_data = dataset_reader(dev_path, label_dict, token_level=False)
    test_data = dataset_reader(test_path, label_dict, token_level=False)

    data = train_data + dev_data + test_data

    features, labels = get_features(data, tokenizer, model)
    score_factor, score_cosine, cluster_centers = get_kmeans(features, labels, ARGS.num_classes)
 
if __name__ == '__main__':

    main()