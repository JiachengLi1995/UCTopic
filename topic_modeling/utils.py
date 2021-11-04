import json
import torch
from tqdm import tqdm
from .consts import ARGS, DEVICE, TOKENIZER


def read_data(path):
    data = []
    with open(path, encoding='utf8') as f:
        for line in f:
            line = json.loads(line)
            data.append(line)

    return data

def batchify(sentence_dict, phrase_list_sampled, batch_size=32):

	batches = []
	pointer = 0
	total_num = len(phrase_list_sampled)
	while pointer < total_num:
		text_batch = []
		span_batch = []

		for data_line in phrase_list_sampled[pointer:pointer+batch_size]:

			sent_id, start, end, phrase_lemma = data_line
			text = sentence_dict[sent_id]

			text_batch.append(text)
			span_batch.append([(start, end)])

		batches.append((text_batch, span_batch))
		pointer += batch_size

	return batches


def get_features(sentence_dict, phrase_list, model, return_prob=False):

	all_features = []

	if return_prob:
		all_probs = []

	for batch in tqdm(batchify(sentence_dict, phrase_list, ARGS.batch_size), ncols=100, desc='Generate all features...'):

		text_batch, span_batch = batch

		inputs = TOKENIZER(text_batch, entity_spans=span_batch, padding=True, add_prefix_space=True, return_tensors="pt")

		for k,v in inputs.items():
			inputs[k] = v.to(DEVICE)

		with torch.no_grad():
			luke_outputs, entity_pooling = model(**inputs)

		if return_prob:
			model_prob = model.get_cluster_prob(entity_pooling)

			all_probs.append(model_prob.detach().cpu())

		
		all_features.append(entity_pooling.detach().cpu())

	all_features = torch.cat(all_features, dim=0)
	if return_prob:
		all_probs = torch.cat(all_probs, dim=0)
		return all_features, all_probs

	return all_features