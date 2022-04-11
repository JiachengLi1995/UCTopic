#%%
from collections import defaultdict
from uctopic import UCTopicTool
import json

#%%
data = []
with open('data/topic_data/data_before_aspect_clustering.json') as f:
    for line in f:
        line = json.loads(line)
        data.append(line)
#%%
sentence_span_dict = defaultdict(set)
for line in data:
    tokens = line['all_tokens']
    phrase = line['aspect_phrase']
    sentence = ' '.join(line['all_tokens'])
    phrase_position = line['all_position']
    start, end = phrase_position[0], phrase_position[-1]

    if start==0:
        char_start = 0
    else:
        char_start = len(' '.join(tokens[:start]))+1
    char_end = char_start+len(phrase)
    sentence_span_dict[sentence].add((char_start, char_end))

# %%
sentences = []
spans = []
for k, v in sentence_span_dict.items():
    sentences.append(k)
    spans.append(list(v))
# %%
tool = UCTopicTool('JiachengLi/uctopic-base', 'cuda:6')

# %%
data_with_aspects, topical_phrases = tool.topic_mining(
                                                sentences=sentences[:10000],
                                                ccl_finetune=True,
                                                spans=spans[:10000],
                                                n_clusters=[10],
                                                finetune_step=1000
                                            )

# %%
tool.save('./')
# %%
tool = UCTopicTool('JiachengLi/uctopic-base', 'cuda:6')
# %%
tool.load('./')
# %%
phrases = []

for sentence, span_list in zip(sentences[:10000], spans[:10000]):
    for span in span_list:
        phrases.append([sentence, span])

query_phrases = []

for sentence, span_list in zip(sentences[-5:], spans[-5:]):
    for span in span_list:
        query_phrases.append([sentence, span])

# %%
tool.build_index(phrases, use_faiss=False)
# %%
results = tool.search(query_phrases[0])
# %%
for i, result in enumerate(results):
    text, span = query_phrases[i]
    print("Retrieval results for query: {}".format(text[span[0]:span[1]]))
    for phrase, score in result:
        text, span = phrase
        print(text[span[0]:span[1]], "(cosine similarity: {:.4f})".format(score))
    print("")
# %%
tool.build_index(phrases, use_faiss=True)
# %%
results = tool.search(query_phrases[:2])
#%%
for i, result in enumerate(results):
    print("Retrieval results for query: {}".format(query_phrases[i]))
    for phrase, score in result:
        print(phrase, "(cosine similarity: {:.4f})".format(score))
    print("")
# %%
tool.predict_topic(query_phrases)
# %%
for phrase in query_phrases:

    text, span = phrase
    print(text[span[0]:span[1]])
# %%
