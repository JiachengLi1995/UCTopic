import torch
from clustering.utils import Confusion
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cosine
from sklearn import preprocessing

def get_kmeans(all_features, all_labels, num_classes):

    all_features = all_features.numpy()
    all_features = preprocessing.normalize(all_features)
    print('Clustering with kmeans...')
    # Perform kmean clustering
    confusion = Confusion(num_classes)
    clustering_model = KMeans(n_clusters=num_classes)
    clustering_model.fit(all_features)
    cluster_assignment = clustering_model.labels_

    if all_labels is None:
        return None, None, clustering_model.cluster_centers_

    true_labels = all_labels
    pred_labels = torch.tensor(cluster_assignment)    
    print("all_embeddings:{}, centers:{}, true_labels:{}, pred_labels:{}".format(all_features.shape, clustering_model.cluster_centers_.shape, len(true_labels), len(pred_labels)))

    confusion.add(pred_labels, true_labels)
    confusion.optimal_assignment(num_classes)
    
    confusion_factor = Confusion(num_classes)
    score_factor = np.matmul(all_features, clustering_model.cluster_centers_.transpose())
    pred_labels_factor = score_factor.argmax(axis=-1)
    pred_labels_factor = torch.tensor(pred_labels_factor)
    confusion_factor.add(pred_labels_factor, true_labels)
    confusion_factor.optimal_assignment(num_classes)

    confusion_cosine = Confusion(num_classes)
    score_cosine = cosine(all_features, clustering_model.cluster_centers_)
    pred_labels_cosine = score_cosine.argmax(axis=-1)
    pred_labels_cosine = torch.tensor(pred_labels_cosine)
    confusion_cosine.add(pred_labels_cosine, true_labels)
    confusion_cosine.optimal_assignment(num_classes)

    print("Clustering iterations:{}, L2 ACC:{:.3f}, Inner ACC:{:.3f}, Cosine ACC:{:.3f}".format(clustering_model.n_iter_, confusion.acc(), confusion_factor.acc(), confusion_cosine.acc()))
    
    return score_factor, score_cosine, clustering_model.cluster_centers_

def get_kmeans_centers(all_features, all_labels, num_classes):

    _, _, centers = get_kmeans(all_features, all_labels, num_classes)

    return centers

def get_kmeans_prediction_and_centers(all_features, all_labels, num_classes):

    _, score_cosine, centers = get_kmeans(all_features, all_labels, num_classes)
    pred_labels_cosine = score_cosine.argmax(axis=-1)
    return pred_labels_cosine, centers

def get_metric(features, centers, labels, num_classes):

    normalized_features = preprocessing.normalize(np.concatenate((centers, features), axis=0))
    centers, features = normalized_features[:num_classes], normalized_features[num_classes:]

    confusion_factor = Confusion(num_classes)
    score_factor = np.matmul(features, centers.transpose())
    pred_labels_factor = score_factor.argmax(axis=-1)
    pred_labels_factor = torch.tensor(pred_labels_factor)
    confusion_factor.add(pred_labels_factor, labels)
    confusion_factor.optimal_assignment(num_classes)

    confusion_cosine = Confusion(num_classes)
    score_cosine = cosine(features, centers)
    pred_labels_cosine = score_cosine.argmax(axis=-1)
    pred_labels_cosine = torch.tensor(pred_labels_cosine)
    confusion_cosine.add(pred_labels_cosine, labels)
    confusion_cosine.optimal_assignment(num_classes)

    print("Inner ACC:{:.3f}, Cosine ACC:{:.3f}".format(confusion_factor.acc(), confusion_cosine.acc()))

