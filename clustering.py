import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import argparse
import os
import random
from tqdm import tqdm
from transformers.models import luke
from clustering.utils import dataset_reader, get_device, batchify, get_data, get_rankings, Confusion
from clustering.kmeans import get_kmeans, get_metric
from uctopic.models import UCTopicConfig, UCTopic, Similarity
from transformers import LukeTokenizer, LukeModel, AdamW, BertTokenizer, BertModel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default=None)
    parser.add_argument("--data_path", type=str, default='data/wnut2017/')
    parser.add_argument("--save_path", type=str, default='result/project_head.bin')
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--use_luke", action='store_true')
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--temp", type=float, default=0.05)
    parser.add_argument("--negative_numbers", type=int, default=10)
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

def get_bert_features(data, pooling = 'ending'):

    all_features = []
    all_labels = []
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()
    model.to(DEVICE)

    def convert_span(text_batch, span_batch):

        word_spans = []

        for text, span in zip(text_batch, span_batch):

            span = span[0]

            start = len(tokenizer.tokenize(text[:span[0]]))
            end = len(tokenizer.tokenize(text[span[0]:span[1]])) + start
            word_spans.append((start, end))

        return word_spans

    def mask_span(text_batch, span_batch):

        word_spans = []
        new_text_batch = []
        for text, span in zip(text_batch, span_batch):

            span = span[0]

            start = len(tokenizer.tokenize(text[:span[0]]))
            new_text = text[:span[0]] + '[MASK]' + text[span[1]:]
            word_spans.append((start, start))

            assert tokenizer.tokenize(new_text)[start] == '[MASK]'
            new_text_batch.append(new_text)

        return new_text_batch, word_spans

    with torch.no_grad():

        for batch in tqdm(batchify(data, ARGS.batch_size), ncols=100, desc='Generate all features...'):

            text_batch, span_batch, label_batch = batch

            if pooling == 'mask':

                text_batch, span_batch = mask_span(text_batch, span_batch)
            
            else:

                span_batch = convert_span(text_batch, span_batch)

            inputs = tokenizer(text_batch, padding=True, return_tensors="pt")

            for k,v in inputs.items():
                inputs[k] = v.to(DEVICE)

            outputs = model(**inputs)
            last_hidden_state = outputs.last_hidden_state

            for i, (span, label) in enumerate(zip(span_batch, label_batch)):

                start, end = span
                if pooling == 'ending':
                    entity_pooling = torch.cat([last_hidden_state[i][start], last_hidden_state[i][end-1]], dim=0)

                elif pooling == 'mean':

                    entity_pooling = (last_hidden_state[i][start] + last_hidden_state[i][end-1]) / 2
                
                elif pooling == 'mask':

                    entity_pooling = last_hidden_state[i][start]

                else:

                    raise NotImplementedError()
                
                all_features.append(entity_pooling.squeeze().detach().cpu())

                all_labels.append(label)

    all_features = torch.stack(all_features, dim=0)
    all_labels = torch.LongTensor(all_labels)

    return all_features, all_labels


def project_features(features, project_head):

    new_features = []

    for i in tqdm(range(0, len(features), ARGS.batch_size), ncols=100, desc='Projecting features...'):

        batch_features = features[i:i+ARGS.batch_size]
        new_features.append(project_head(batch_features).detach().cpu())

    new_features = torch.cat(new_features, dim=0)

    return new_features


def train(features, model, optimizer, dataset, labels=None):

    model.train()
    for i in tqdm(range(0, len(dataset), ARGS.batch_size), ncols=100, desc='Training...'):

        batch = torch.LongTensor(dataset[i:i+ARGS.batch_size]).to(DEVICE) # (batch_size, num_samples) [anchor, positive, negative, negative,...]
        batch_feature = features[batch]

        if labels is not None: ## classification
            batch_labels = torch.LongTensor(labels[i:i+ARGS.batch_size]).to(DEVICE)
            logits, loss = model(batch_feature, batch_labels)

        else:

            ## contrastive learning
            cos_sim, loss = model(batch_feature)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

        optimizer.step()
        optimizer.zero_grad()

def eval(features, model, dataset, labels=None, mode='cl'):

    model.eval()
    all_pred = []
    for i in tqdm(range(0, len(dataset), ARGS.batch_size), ncols=100, desc='Evaluating...'):

        batch = torch.LongTensor(dataset[i:i+ARGS.batch_size]).to(DEVICE) # (batch_size, num_samples) [anchor, positive, negative, negative,...]
        batch_feature = features[batch]

        if labels is not None:
            logits, loss = model(batch_feature)
            _, pred = logits.max(-1)

        else:
        
            cos_sim, loss = model(batch_feature)

            _, pred = cos_sim.max(-1)

        all_pred.append(pred.detach().cpu())

    all_pred = torch.cat(all_pred, dim=0)
    if labels is not None:
        accuracy = (all_pred == torch.LongTensor(labels)).float().mean().item()

    else:
        if mode == 'cl':
            accuracy = all_pred.eq(0).float().mean().item()
        else:
            accuracy = loss.detach().cpu().item()
            accuracy = 1.0 / accuracy
            # labels = torch.arange(cos_sim.size(0)).long().to(cos_sim.device)
            # accuracy = (torch.argmax(cos_sim, 1) == labels).float().mean().detach().cpu().item()

    print(f'Accuracy: {accuracy}')

    return accuracy, all_pred

def contrastive_learning(features, scores, labels, config):

    rankings = get_rankings(scores)

    cl_model = ContrastiveLearning(config, ARGS)
    cl_model.to(DEVICE)

    features = torch.tensor(features).to(DEVICE)
    optimizer = AdamW(cl_model.parameters(), lr=ARGS.learning_rate)

    best_acc = 0
    for _ in range(ARGS.epoch):

        dataset = get_data(rankings, negative_numbers=ARGS.negative_numbers)
        train_dataset = dataset[:int(0.8 * len(dataset))]
        eval_dataset = dataset[int(0.8 * len(dataset)):]

        train(features, cl_model, optimizer, train_dataset)

        acc, _ = eval(features, cl_model, eval_dataset)

        if acc > best_acc:
            best_acc = acc
            save_model(cl_model.head)
            print('Best checkpoint.')

    project_head = ProjectHead(config, ARGS.output_size)
    load_model(project_head)
    project_head.to(DEVICE)

    new_features = project_features(features, project_head)
    new_features = new_features.numpy()
    score_factor, score_cosine = get_kmeans(new_features, labels, ARGS.num_classes)

def classifier(features, scores, labels, config):

    rankings = get_rankings(scores)

    dataset = []
    weak_labels = []
    for i in range(rankings.shape[-1]):
        dataset += list(rankings[:, i])
        weak_labels += [i] * rankings.shape[0]

    idxs = list(range(len(dataset)))
    random.shuffle(idxs)

    dataset = [dataset[idx] for idx in idxs]
    weak_labels = [weak_labels[idx] for idx in idxs]

    train_dataset = dataset[:int(0.8 * len(dataset))]
    train_labels = weak_labels[:int(0.8 * len(dataset))]

    eval_dataset = dataset[int(0.8 * len(dataset)):]
    eval_labels = weak_labels[int(0.8 * len(dataset)):]

    model = Classifier(config, ARGS)
    model.to(DEVICE)

    features = torch.tensor(features).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=ARGS.learning_rate)

    best_acc = 0
    for _ in range(ARGS.epoch):

        train(features, model, optimizer, train_dataset, train_labels)

        acc, _ = eval(features, model, eval_dataset, eval_labels)

        if acc > best_acc:
            best_acc = acc
            save_model(model)
            print('Best checkpoint.')

    load_model(model)
    model.to(DEVICE)
    print('Test:')
    acc, pred = eval(features, model, list(range(len(features))), labels)

    confusion = Confusion(ARGS.num_classes)
    confusion.add(pred, labels)
    confusion.optimal_assignment(ARGS.num_classes)

    print(f'After confusion acc: {confusion.acc()}')
    print('Clustering scores:',confusion.clusterscores())

def autoencoder(features, labels, config, cluster_centers):

    dataset = list(range(features.shape[0]))
    #random.shuffle(dataset)
    train_dataset = dataset[:int(0.8 * len(dataset))]
    eval_dataset = dataset[int(0.8 * len(dataset)):]

    model = AutoEncoder(config, ARGS, init_class_emb=cluster_centers)
    model.to(DEVICE)

    features = torch.tensor(features).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=ARGS.learning_rate)

    best_acc = 0
    for _ in range(ARGS.epoch):

        random.shuffle(train_dataset)
        train_dataset = [[ele] + random.sample(dataset, 20) for ele in train_dataset]
        train(features, model, optimizer, train_dataset)

        eval_dataset = [[ele] + random.sample(dataset, 20) for ele in eval_dataset]
        acc, pred = eval(features, model, eval_dataset, mode='auto')

        print((labels[int(0.8 * len(dataset)):] == pred).float().mean().detach().cpu().item())
        train_dataset = dataset[:int(0.8 * len(dataset))]
        eval_dataset = dataset[int(0.8 * len(dataset)):]

        get_metric(features.detach().cpu().numpy(), model.class_emb.detach().cpu().numpy(), labels, ARGS.num_classes)

        if acc > best_acc:
            best_acc = acc
            save_model(model)
            print('Best checkpoint.')

    load_model(model)
    model.to(DEVICE)
    centers = model.class_emb
    print('Test:')

    get_metric(features.detach().cpu().numpy(), centers.detach().cpu().numpy(), labels, ARGS.num_classes)




def save_model(model):
    torch.save(model.state_dict(), ARGS.save_path)

def load_model(model):

    model.load_state_dict(torch.load(ARGS.save_path))
    model.eval()

class ProjectHead(nn.Module):
    def __init__(self, config, output_size):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, output_size)

    def forward(self, features):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class AutoEncoder(nn.Module):
    def __init__(self, config, args, init_class_emb=None):
        super().__init__()

        self.linear = ProjectHead(config, args.num_classes)#nn.Linear(config.hidden_size, args.num_classes)
        self.softmax= nn.Softmax(dim=-1)
        self.class_emb = Parameter(torch.empty(size=(args.num_classes, config.hidden_size)))
        self.num_classes = args.num_classes
        self.sim = Similarity(args.temp)

        if init_class_emb is None:
            torch.nn.init.xavier_uniform(self.class_emb)
        else:
            self.class_emb.data = torch.from_numpy(init_class_emb)

    def forward(self, features):

        anchor_features = features[:, 0] # (batch_size, hidden_size)
        contrastive_features = features[:, 1:]#torch.mean(features[:, 1:], dim=1) # (batch_size, sample_num, hidden_size)

        class_importrance = self.softmax(self.linear(anchor_features))  ## (batch_size, num_class)
        recovered_emb = torch.matmul(class_importrance, self.class_emb) ## (batch_size, hidden_size)

        
        reconstruction_triplet_loss = AutoEncoder._reconstruction_loss(anchor_features,
                                                                        recovered_emb,
                                                                        contrastive_features)

        loss = 0.1 * self._ortho_regularizer() + reconstruction_triplet_loss

        # cos_sim = self.sim(recovered_emb.unsqueeze(1), features.unsqueeze(0))
        # labels = torch.arange(cos_sim.size(0)).long().to(cos_sim.device)

        # loss_fct = nn.CrossEntropyLoss()
        # loss = loss_fct(cos_sim, labels)

        return class_importrance, loss

    @staticmethod
    def _reconstruction_loss(anchor_features, recovered_emb, contrastive_features):

        positive_dot_products = torch.sum(anchor_features * recovered_emb, dim=-1)
        negative_dot_products = torch.sum(contrastive_features * recovered_emb.unsqueeze(1), dim=-1)
        max_margin = torch.max(1 - positive_dot_products.unsqueeze(1) + negative_dot_products, torch.zeros_like(negative_dot_products))
        reconstruction_triplet_loss = max_margin.sum()

        return reconstruction_triplet_loss

    def _ortho_regularizer(self):
        return torch.norm(
            torch.matmul(self.class_emb, self.class_emb.t()) \
            - torch.eye(self.num_classes).to(DEVICE))


class ContrastiveLearning(nn.Module):
    def __init__(self, config, args):
        super().__init__()

        self.head = ProjectHead(config, args.output_size)
        self.sim = Similarity(args.temp)

        self.loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, features): # (batch_size, num_samples, hidden_size)

        features = self.head(features) # (batch_size, num_samples, output_size)

        anchor_features = features[:, 0].unsqueeze(1) # (batch_size, 1, hidden_size)
        contrastive_features = features[:, 1:] # (batch_size, num_samples-1, hidden_size)
        cos_sim = self.sim(anchor_features, contrastive_features) # (batch_size, num_samples-1)

        label_size = cos_sim.size(0)
        labels = torch.zeros(label_size, device=DEVICE, dtype=torch.long) # (batch_size)

        loss = self.loss_fn(cos_sim, labels)

        return cos_sim, loss

class Classifier(nn.Module):
    def __init__(self, config, args):
        super().__init__()

        self.head = ProjectHead(config, args.num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, features, labels=None):

        logits = self.head(features)

        loss = None

        if labels is not None:

            loss = self.loss_fn(logits, labels.view(-1))


        return logits, loss


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

    train_data = dataset_reader(train_path, label_dict)
    dev_data = dataset_reader(dev_path, label_dict)
    test_data = dataset_reader(test_path, label_dict)

    data = train_data + dev_data + test_data

    #features, labels = get_features(data, tokenizer, model)
    features, labels = get_bert_features(data, pooling='ending')
    score_factor, score_cosine, cluster_centers = get_kmeans(features, labels, ARGS.num_classes)

    #contrastive_learning(features, score_cosine, labels, config)
    #classifier(features, score_cosine, labels, config)
    #autoencoder(features, labels, config, cluster_centers)
    

    
 
if __name__ == '__main__':

    main()