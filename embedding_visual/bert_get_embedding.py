import pickle
import json
import torch
from tqdm import tqdm
from transformers import BertModel, BertTokenizerFast
from transformers.pipelines.zero_shot_classification import ZeroShotClassificationArgumentHandler

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.cuda()
model.eval()

path = './data/conll2003/train.json'
data = []
with open(path, "r") as f:
    lines = f.readlines()
    for line in lines:
        line = json.loads(line)
        sentence = line['sentences'][0]
        spans = []
        ner_labels = []
        for entity in line['ner'][0]:
            spans.append((entity[0], entity[1]))
            ner_labels.append(entity[2])

        if len(ner_labels)==0:
            continue
        data.append([sentence, spans, ner_labels])


data_processed = []
for line in tqdm(data):

    sentence, spans, ner_labels = line

    for span, label in zip(spans, ner_labels):

        new_sentence = sentence[:span[0]] + [tokenizer.mask_token] + sentence[span[1]+1:]
        inputs = tokenizer(new_sentence, is_split_into_words=True, return_tensors="pt")
        word_ids = inputs.word_ids(batch_index=0)
        position = word_ids.index(span[0])
        data_processed.append([inputs, position, label])


f = open('bert_embedding_conll2003.pickle', 'wb')
outputs = []

for line in tqdm(data_processed):

    inputs, position, label = line
    
    for k,v in inputs.items():
        inputs[k] = v.cuda()
    bert_outputs = model(**inputs)
    last_hidden_state = bert_outputs.last_hidden_state
    span_pool = last_hidden_state[0][position].detach().cpu().numpy()

    outputs.append([span_pool, label])

pickle.dump(outputs, f)
f.close()
