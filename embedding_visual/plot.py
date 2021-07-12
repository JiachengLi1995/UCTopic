
#%%
import umap
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = '18'
plt.rcParams["figure.figsize"] = [6.4, 4.8]
#%%
def read_embedding(file_name):
    f_embedding = open(file_name)
    embeddings = []

    for line in f_embedding:
        line = np.array([float(ele) for ele in line.strip().split('\t')])
        embeddings.append(line)

    f_embedding.close()
    return embeddings

def read_labels(file_name):
    f_labels = open(file_name)
    labels = []

    for line in f_labels:

        line = line.strip()
        labels.append(line)
    f_labels.close()

    return labels

def convert_label_color(labels):

    label_dict = dict()
    for label in labels:
        if label not in label_dict:
            label_dict[label] = len(label_dict)
    labels = [label_dict[label] for label in labels]

    return labels

def convert_and_filter_labels(embeddings, labels):

    # label_dict = {'/people/person/place_lived': 0, '/people/person/nationality': 1, '/location/administrative_division/country':3, '/location/country/capital':4}
    # 0: purple, 1: blue, 3: green, 4: capital
    #label_dict = {'PER': 'b', 'LOC': 'r', 'ORG': 'g'}
    label_dict = {'person': 'b', 'organization': 'r', 'time': 'g', 'location': 'y', }
    #label_dict = {'Chemical': 'c', 'Disease':'y'}
    new_embeddings = []
    new_labels = []
    for embedding, label in zip(embeddings, labels):
        if label in label_dict:
            new_embeddings.append(embedding)
            new_labels.append(label_dict[label])

    return new_embeddings, new_labels

def compute_umap(embeddings):
    reducer = umap.UMAP(n_neighbors=10, metric='cosine')
    embedding = reducer.fit_transform(np.array(embeddings))

    return embedding

def plot(embedding, labels, xlabel, file_name):

    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, s=4)
    plt.xlabel(xlabel)
    plt.savefig(file_name, dpi=500)


#%%
# embedding_name = 'embedding_relation_endpoint.tsv'
# tags_name = 'embedding_relation_endpoint_tags.tsv'
# embeddings = read_embedding(embedding_name)
# labels = read_labels(tags_name)
# print(len(embeddings))
# embeddings, labels = convert_and_filter_labels(embeddings, labels)
# #labels = convert_label_color(labels)
# print(len(embeddings))
# embedding = compute_umap(embeddings)
# plot(embedding, labels, 'RoBERTa EndPoint','relation_endpoint.png')
# %%
# embedding_name = 'embedding_relation_maxpooling.tsv'
# tags_name = 'embedding_relation_maxpooling_tags.tsv'
# embeddings = read_embedding(embedding_name)
# labels = read_labels(tags_name)
# print(len(embeddings))
# embeddings, labels = convert_and_filter_labels(embeddings, labels)
# #labels = convert_label_color(labels)
# print(len(embeddings))
# embedding = compute_umap(embeddings)
# plot(embedding, labels, 'RoBERTa MaxPooling','relation_maxpooling.png')

#%%
# embedding_name = 'embedding_bc5cdr_normalized.tsv'
# tags_name = 'embedding_bc5cdr_tags_normalized.tsv'
# embeddings = read_embedding(embedding_name)
# labels = read_labels(tags_name)
# print(len(embeddings))
# embeddings, labels = convert_and_filter_labels(embeddings, labels)
# #labels = convert_label_color(labels)
# print(len(embeddings))
# embedding = compute_umap(embeddings)
# plot(embedding, labels, 'Luke relation representation','bc5cdr.png')

#%%
# embedding_name = 'embedding_relation_maxpooling.tsv'
# tags_name = 'embedding_relation_maxpooling_tags.tsv'
# embeddings = read_embedding(embedding_name)
# labels = read_labels(tags_name)
# print(len(embeddings))
# embeddings, labels = convert_and_filter_labels(embeddings, labels)
# #labels = convert_label_color(labels)
# print(len(embeddings))
# embedding = compute_umap(embeddings)
# plot(embedding, labels, 'Luke relation representation','relation_maxpooling.png')


#%%
embedding_name = 'embedding_openentity.pickle_vectors.tsv'
tags_name = 'embedding_openentity.pickle_meta.tsv'
embeddings = read_embedding(embedding_name)
labels = read_labels(tags_name)
print(len(embeddings))
embeddings, labels = convert_and_filter_labels(embeddings, labels)
#labels = convert_label_color(labels)
print(len(embeddings))
embedding = compute_umap(embeddings)
plot(embedding, labels, 'LUKE Entity Representation (OpenEntity)','LUKE_openentity.png')
# %%
