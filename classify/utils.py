import json
from tqdm import tqdm
import torch

class TypingMetric:
    
    def __init__(self) -> None:

        self.correct_count = 0.0
        self.total_count = 0.0
        self.pred = []
        self.true = []

    def __call__(
        self,
        logits: torch.Tensor, # (batch_size, span_num, span_label_num)
        gold_labels: torch.Tensor # (batch_size, span_num, span_label_num)
    ):

        label_num = logits.size(-1)

        logits = logits.view(-1, label_num).detach().cpu().tolist()
        gold_labels = gold_labels.view(-1, label_num).detach().cpu().tolist()

        correct_cnt, total_cnt, y1, y2 = self.accuracy(logits, gold_labels)

        self.correct_count += correct_cnt
        self.total_count += total_cnt
        self.pred += y1
        self.true += y2

    def accuracy(self, logits, labels):
        cnt = 0
        total_cnt = 0
        y1 = []
        y2 = []
        for x1, x2 in zip(logits, labels):
            yy1 = []
            yy2 = []
            top = max(x1)
            for i in range(len(x1)):
                if x1[i] > 0:
                    yy1.append(i)
                if x2[i] > 0:
                    yy2.append(i)
            y1.append(yy1)
            y2.append(yy2)
            cnt += set(yy1) == set(yy2)
            total_cnt += 1
        return cnt, total_cnt, y1, y2

    def f1(self, p, r):
        if r == 0.:
            return 0.
        return 2 * p * r / float( p + r )

    def loose_macro(self, true, pred):
        num_entities = len(true)
        p = 0.
        r = 0.
        for true_labels, predicted_labels in zip(true, pred):
            if len(predicted_labels) > 0:
                p += len(set(predicted_labels).intersection(set(true_labels))) / float(len(predicted_labels))
            if len(true_labels):
                r += len(set(predicted_labels).intersection(set(true_labels))) / float(len(true_labels))
        precision = p / num_entities
        recall = r / num_entities
        return precision, recall, self.f1(precision, recall)

    def loose_micro(self, true, pred):
        num_predicted_labels = 0.
        num_true_labels = 0.
        num_correct_labels = 0.
        for true_labels, predicted_labels in zip(true, pred):
            num_predicted_labels += len(predicted_labels)
            num_true_labels += len(true_labels)
            num_correct_labels += len(set(predicted_labels).intersection(set(true_labels))) 
        if num_predicted_labels > 0:
            precision = num_correct_labels / num_predicted_labels
        else:
            precision = 0.
        recall = num_correct_labels / num_true_labels
        return precision, recall, self.f1(precision, recall)

    def get_metric(self, reset: bool = False):
        """
        # Returns
        The accumulated accuracy.
        """
        return_dict = {'accuracy': 0.0, 'micro': {'p': 0.0, 'r': 0.0, 'f1': 0.0}, 'macro': {'p': 0.0, 'r': 0.0, 'f1': 0.0}}

        if self.total_count > 1e-12:
            return_dict['accuracy'] = float(self.correct_count) / float(self.total_count)
            return_dict['micro']['p'], return_dict['micro']['r'], return_dict['micro']['f1'] = self.loose_micro(self.true, self.pred)
            return_dict['macro']['p'], return_dict['macro']['r'], return_dict['macro']['f1'] = self.loose_macro(self.true, self.pred)
          
        if reset:
            self.reset()
            
        return return_dict

    def reset(self):
        self.correct_count = 0.0
        self.total_count = 0.0
        self.pred = []
        self.true = []


class LabelField:
    def __init__(self):
        self.label2id = dict()
        self.id2label = dict()
        self.label_num = 0

    def get_id(self, label):
        
        if label in self.label2id:
            return self.label2id[label]
        
        self.label2id[label] = self.label_num
        self.id2label[self.label_num] = label
        self.label_num += 1

        return self.label2id[label]

    def get_label(self, id):

        if id not in self.id2label:
            print(f'Cannot find label that id is {id}!!!')
            assert 0
        return self.id2label[id]
    
    def get_num(self):
        return self.label_num
    
    def all_labels(self):
        return list(self.label2id.keys())

    def update(self, label_field):

        self.label2id.update(label_field.label2id)
        self.id2label.update(label_field.id2label)
        self.label_num = label_field.label_num


def process_dataset(dataset, label_field):

    processed = []

    for line in tqdm(dataset, desc='Processing dataset.', ncols=100):

        sentence = line['sent']
        sentence = sentence.replace('-LRB-', '(')
        sentence = sentence.replace('-RRB-', ')')
        start = line['start']
        end = line['end']
        
        label = [label_field.get_id(label) for label in line['labels']]

        processed.append([sentence, [(start, end)], label])

    return processed



def read_dataset(dataset, label_field):

    train_file = f'data/{dataset}/train.json'
    val_file = f'data/{dataset}/dev.json'
    test_file = f'data/{dataset}/test.json'

    train_data = process_dataset(json.load(open(train_file, encoding='utf8')), label_field)
    val_data = process_dataset(json.load(open(val_file, encoding='utf8')), label_field)
    test_data = process_dataset(json.load(open(test_file, encoding='utf8')), label_field)
    
    return train_data, val_data, test_data

