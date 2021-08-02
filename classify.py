import json
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
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--use_luke", action='store_true')
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
        batches.append(data[pointer:pointer+batch_size])
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

        self.num_labels = args.num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.metric = TypingMetric()

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        entity_ids=None,
        entity_attention_mask=None,
        entity_token_type_ids=None,
        entity_position_ids=None,
        head_mask=None,
        inputs_embeds=None,
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
                position_ids=position_ids,
                entity_ids=entity_ids,
                entity_attention_mask=entity_attention_mask,
                entity_token_type_ids=entity_token_type_ids,
                entity_position_ids=entity_position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
            feature_vector = outputs.entity_last_hidden_state[:, 0, :]
        else:
            luke_output, entity_pooling = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                entity_ids=entity_ids,
                entity_attention_mask=entity_attention_mask,
                entity_token_type_ids=entity_token_type_ids,
                entity_position_ids=entity_position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )

            feature_vector = entity_pooling.entity_last_hidden_state[:, 0, :]

        feature_vector = self.dropout(feature_vector)
        logits = self.classifier(feature_vector)

        loss = self.loss_fn(logits, labels)
        self.metric(logits, labels)

        return loss, metrics


        

def train(model, optimizer, dataset, user_phrases, business_phrases, pad_token):

    model.train()
    tqdm_dataloader = tqdm(batchify(dataset), ncols=100)
    for batch in tqdm_dataloader:

        user_features, candidates, _ = generate_data(batch, user_phrases, business_phrases, pad_token)

        user_features = user_features.to(DEVICE)
        candidates = candidates.to(DEVICE)

        logits, loss = model(user_features, candidates)

        tqdm_dataloader.set_description('loss {:.5f} '.format(loss.detach().cpu().item()))


        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

        optimizer.step()
        optimizer.zero_grad()

def eval(model, dataset, user_phrases, business_phrases, pad_token):

    model.eval()
    all_ranks = []
    for batch in tqdm(batchify(dataset), ncols=100, desc='Eval:'):

        user_features, candidates, lengths = generate_data(batch, user_phrases, business_phrases, pad_token)

        user_features = user_features.to(DEVICE)
        candidates = candidates.to(DEVICE)

        logits = model(user_features, candidates, mode='eval')
        
        logits = logits.detach().cpu().tolist()
        for score, candidate, length in zip(logits, candidates.tolist(), lengths):
            
            filted_score = []
            for s, c in zip(score, candidate):
                if c!=pad_token:
                    filted_score.append(s)
            
            rank = np.argsort(-np.array(filted_score))
            all_ranks.append(list(rank[:length]))

    score = metric(all_ranks)

    return score

def save_model(model):

    torch.save(model.state_dict(), ARGS.save_path)


def load_model(model):

    model.load_state_dict(torch.load(ARGS.save_path))
    model.eval()

def main():
    
    label_field = LabelField()
    train_data, val_data, test_data = read_dataset(ARGS.dataset, label_field)

    model = EntityClassification(features, ARGS)
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

    optimizer = optim.Adam(optimizer_grouped_parameters, lr=ARGS.learning_rate)

    pad_token = features.shape[0]

    best_score = 0

    for epoch in range(ARGS.epoch):

        print(f'Epoch: {epoch}')

        train(model, optimizer, train_data, user_phrases, business_phrases, pad_token)

        score = eval(model, dev_data, user_phrases, business_phrases, pad_token)
        print(f'Result: {score}')
        if score > best_score:
            print('Best checkpoint.')
            save_model(model)
            best_score = score

    print('Testing:')
    load_model(model)
    score = eval(model, dev_data, user_phrases, business_phrases, pad_token)
    print(f'Result: {score}')


if __name__ == '__main__':

    main()