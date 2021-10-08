import torch
import numpy as np
from utils import Confusion, get_features
from sklearn import cluster

def evaluate_embedding(data, tokenizer, model, args, step, logger):
    confusion, confusion_model = Confusion(args.num_classes), Confusion(args.num_classes)
    model.eval()
    
    all_embeddings, all_prob, all_labels = get_features(data, tokenizer, model, return_prob=True)

    all_pred = all_prob.max(1)[1]
    confusion_model.add(all_pred, all_labels)
    confusion_model.optimal_assignment(args.num_classes)
    acc_model = confusion_model.acc()

    kmeans = cluster.KMeans(n_clusters=args.num_classes, random_state=args.seed)
    embeddings = all_embeddings.cpu().numpy()
    kmeans.fit(embeddings)
    pred_labels = torch.tensor(kmeans.labels_.astype(np.int))
    # clustering accuracy 
    confusion.add(pred_labels, all_labels)
    confusion.optimal_assignment(args.num_classes)
    acc = confusion.acc()
    
    ressave = {"acc":acc, "acc_model":acc_model}
    for key, val in ressave.items():
        logger.add_scalar('Test/{}'.format(key), val, step)
    
    print('[Representation] Clustering scores:',confusion.clusterscores()) 
    print('[Representation] ACC: {:.3f}'.format(acc)) 
    print('[Model] Clustering scores:',confusion_model.clusterscores()) 
    print('[Model] ACC: {:.3f}'.format(acc_model))
