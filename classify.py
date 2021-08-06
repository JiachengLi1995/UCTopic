import sys
import argparse
import torch
import torch.nn as nn
import random
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from transformers import LukeTokenizer, LukeModel, AdamW
from uctopic.models import UCTopicConfig, UCTopic, Similarity
from classify.utils import LabelField, TypingMetric, read_dataset

def get_device(gpu):
    return torch.device('cpu' if gpu is None else f'cuda:{gpu}')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default=None)
    parser.add_argument("--dataset", type=str, default='OpenEntity')
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--use_luke", action='store_true')
    parser.add_argument("--freeze", action='store_true')
    parser.add_argument("--save_path", type=str, default='./best_model.bin')
    args = parser.parse_args()
    return args

ARGS = parse_args()
DEVICE = get_device(ARGS.gpu)



def batchify(data, batch_size=32):

    batches = []
    pointer = 0
    total_num = len(data)
    while pointer < total_num:
        text_batch = []
        span_batch = []
        label_batch = []

        for data_line in data[pointer:pointer+batch_size]:

            text, span, label = data_line
            
            text_batch.append(text)
            span_batch.append(span)
            label_batch.append(label)

        batches.append((text_batch, span_batch, label_batch))
        pointer += batch_size

    return batches


class EntityClassification(nn.Module):
    def __init__(self, config, args):
        super().__init__()

        self.args = args
        if args.use_luke:
            self.model = LukeModel.from_pretrained("studio-ousia/luke-base")
        else:
            self.model = UCTopic(config)
            self.model.load_state_dict(torch.load('result/pytorch_model.bin'))

        if args.freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        self.num_labels = args.label_num
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.act = nn.ReLU()
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.metric = TypingMetric()

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        entity_ids=None,
        entity_attention_mask=None,
        entity_token_type_ids=None,
        entity_position_ids=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None
        ):
        if self.args.use_luke:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                entity_ids=entity_ids,
                entity_attention_mask=entity_attention_mask,
                entity_token_type_ids=entity_token_type_ids,
                entity_position_ids=entity_position_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )
            feature_vector = outputs.entity_last_hidden_state[:, 0, :]
        else:
            luke_output, entity_pooling = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                entity_ids=entity_ids,
                entity_attention_mask=entity_attention_mask,
                entity_token_type_ids=entity_token_type_ids,
                entity_position_ids=entity_position_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )

            feature_vector = entity_pooling[:, 0, :]

        feature_vector = self.dropout(feature_vector)
        feature_vector = self.act(self.linear(feature_vector))
        logits = self.classifier(feature_vector)

        loss = self.loss_fn(logits, labels)
        self.metric(logits, labels)

        return loss, self.metric


def train(tokenizer, model, optimizer, dataset):

    model.train()
    tqdm_dataloader = tqdm(batchify(dataset), ncols=100)

    for batch in tqdm_dataloader:

        text_batch, span_batch, label_batch = batch
        inputs = tokenizer(text_batch, entity_spans=span_batch, padding=True, add_prefix_space=True, return_tensors="pt")
        label_batch = torch.FloatTensor(label_batch).to(DEVICE)

        for k,v in inputs.items():
            inputs[k] = v.to(DEVICE)

        loss, metric = model(**inputs, labels=label_batch)

        metric_dict = metric.get_metric()
        micro_p, micro_r, micro_f = metric_dict['micro']['p'], metric_dict['micro']['r'], metric_dict['micro']['f1']

        tqdm_dataloader.set_description('Loss {0:.5f}, Micro F1 {1:2.4f}'.format(loss.detach().cpu().item(), micro_f))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

        optimizer.step()
        optimizer.zero_grad()

def eval(tokenizer, model, dataset):

    model.eval()
    model.metric.reset()
    for batch in tqdm(batchify(dataset), ncols=100, desc='Eval:'):

        text_batch, span_batch, label_batch = batch
        inputs = tokenizer(text_batch, entity_spans=span_batch, padding=True, add_prefix_space=True, return_tensors="pt")
        label_batch = torch.FloatTensor(label_batch).to(DEVICE)

        for k,v in inputs.items():
            inputs[k] = v.to(DEVICE)

        loss, metric = model(**inputs, labels=label_batch)
        
    metric_dict = metric.get_metric()
    accuracy = metric_dict['accuracy']
    micro_p, micro_r, micro_f = metric_dict['micro']['p'], metric_dict['micro']['r'], metric_dict['micro']['f1']
    macro_p, macro_r, macro_f = metric_dict['macro']['p'], metric_dict['macro']['r'], metric_dict['macro']['f1']

    sys.stdout.write('loss: {0:2.6f},  Acc: {1:3.2f}% '.format(loss, 100 * accuracy) +'\n')
    sys.stdout.write('    \t F1 \t Precision \t Recall \n')
    sys.stdout.write('Micro \t {0:2.4f} \t {1:2.4f} \t {2:2.4f} '.format(micro_f, micro_p, micro_r) +'\n')
    sys.stdout.write('Macro \t {0:2.4f} \t {1:2.4f} \t {2:2.4f}'.format(macro_f, macro_p, macro_r) +'\n')
    sys.stdout.flush()

    return micro_f

def save_model(model):

    torch.save(model.state_dict(), ARGS.save_path)


def load_model(model):

    model.load_state_dict(torch.load(ARGS.save_path))
    model.eval()

def main():
    
    label_field = LabelField()
    train_data, val_data, test_data = read_dataset(ARGS.dataset, label_field)
    ARGS.label_num = label_field.label_num

    config = UCTopicConfig.from_pretrained("studio-ousia/luke-base")
    tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-base")
    model = EntityClassification(config, ARGS)
    model.to(DEVICE)

    named_parameters = model.named_parameters()
    no_decay = ["bias", "norm"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in named_parameters if not any(nd in n for nd in no_decay)],
            "weight_decay": ARGS.weight_decay,
        },
        {"params": [p for n, p in named_parameters if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=ARGS.learning_rate)

    best_score = 0

    for epoch in range(ARGS.epoch):

        print(f'Epoch: {epoch}')
        random.shuffle(train_data)
        train(tokenizer, model, optimizer, train_data)

        score = eval(tokenizer, model, val_data)
        print(f'Result: {score}')
        if score > best_score:
            print('Best checkpoint.')
            save_model(model)
            best_score = score

    print('Testing:')
    load_model(model)
    score = eval(tokenizer, model, test_data)
    print(f'Result: {score}')


if __name__ == '__main__':

    main()