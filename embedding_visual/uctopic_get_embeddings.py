import pickle
import json
import torch
from tqdm import tqdm
from uctopic.models import UCTopic
from transformers import LukeTokenizer, AutoConfig

config = AutoConfig.from_pretrained("studio-ousia/luke-base")
tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-base")
model = UCTopic(config)
model.load_state_dict(torch.load('result/pytorch_model.bin'))
model.cuda()
model.eval()

path = './data/OpenEntity/train.json'
data = []
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

#     for span, label in zip(spans, ner_labels):
#         span_text = ' '.join(sentence[span[0]:span[1]+1])

#         span_start = text.find(span_text)
#         span_end = span_start+len(span_text)
    
#         data_processed.append([text, [(span_start, span_end)], label])

f = open('uctopic_embedding_openentity.pickle', 'wb')
f_pooling = open('uctopic_pooling_embedding_openentity.pickle', 'wb')
outputs = []
pooling_outputs = []
for line in tqdm(data_processed):

    text, new_spans, label = line
    inputs = tokenizer(text, entity_spans=new_spans, add_prefix_space=True, return_tensors="pt")
    
    for k,v in inputs.items():
        inputs[k] = v.cuda()
    luke_outputs, entity_pooling = model(**inputs)
    entity_last_hidden_state = luke_outputs.entity_last_hidden_state

    span_pool = entity_last_hidden_state[0].detach().cpu().numpy()
    entity_pooling = entity_pooling[0].detach().cpu().numpy()

    outputs.append([span_pool, label])
    pooling_outputs.append([entity_pooling, label])
pickle.dump(outputs, f)
f.close()

pickle.dump(pooling_outputs, f_pooling)
f_pooling.close()


    

