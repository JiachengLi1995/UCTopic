import torch
import pickle
import json
from tqdm import tqdm
from transformers import LukeTokenizer, LukeModel


model = LukeModel.from_pretrained("studio-ousia/luke-base")
tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-base")

model.cuda()
model.eval()

path = './data/OpenEntity/train.json'
# data = []
# with open(path, "r") as f:
#     lines = f.readlines()
#     for line in lines:
#         line = json.loads(line)
#         sentence = line['sentences'][0]
#         spans = []
#         ner_labels = []
#         for entity in line['ner'][0]:
#             spans.append((entity[0], entity[1]))
#             ner_labels.append(entity[2])

#         if len(ner_labels)==0:
#             continue
#         data.append([sentence, spans, ner_labels])

data_processed = []
with open(path, "r") as f:
    lines = json.load(f)
    for line in tqdm(lines):
        try:
            sentence = line['sent']
            span = [(line['start'], line['end'])]
            label = line['labels'][0]

        except:
            print(line)

        data_processed.append([sentence, span, label])
# data_processed = []
# for line in tqdm(data):

#     sentence, spans, ner_labels = line

#     text = ' '.join(sentence)

#     new_spans = []
#     for span in spans:
#         span_text = ' '.join(sentence[span[0]:span[1]+1])

#         span_start = text.find(span_text)
#         span_end = span_start+len(span_text)
#         new_spans.append((span_start, span_end))
    
#     data_processed.append([text, new_spans, ner_labels])


f = open('embedding_openentity.pickle', 'wb')
outputs = []
for line in tqdm(data_processed):

    text, new_spans, ner_labels = line
    inputs = tokenizer(text, entity_spans=new_spans, add_prefix_space=True, return_tensors="pt")
    
    for k,v in inputs.items():
        inputs[k] = v.cuda()
    model_outputs = model(**inputs)
    entity_last_hidden_state = model_outputs.entity_last_hidden_state

    span_pool = entity_last_hidden_state[0].detach().cpu().numpy()

    # for i, label in enumerate(ner_labels):

    #     outputs.append([span_pool[i], label])
    outputs.append([span_pool, ner_labels])

pickle.dump(outputs, f)
f.close()


    

