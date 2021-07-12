import json

f = open('./data/wiki_small.json')


for line in f:
    line = json.loads(line)
    sentence = line['text']
    entities = line['selected_entities']

    for entity in entities:

        start, end = entity[2], entity[3]
        if start>len(sentence) or end>len(sentence):
            print(line)
            break
            