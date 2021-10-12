import torch
import numpy as np
import random
from tqdm import tqdm
from collections import defaultdict, Counter
from uctopic.models import UCTopicCluster, UCTopicConfig
from clustering.trainer import ClusterLearner
from clustering.kmeans import get_kmeans
from clustering.dataloader import get_train_loader
from clustering.consts import ARGS, TOKENIZER, DEVICE
from clustering.utils import dataset_reader, get_features, set_logger, update_logger, get_rankings
from clustering.evaluation import evaluate_embedding


def set_global_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def main():
    set_global_random_seed(ARGS.seed)

    # Conduct clustering with kmeans

    if 'conll2003' in ARGS.data_path:
        label_dict = {'PER':0, 'LOC':1, 'ORG':2}
    elif 'bc5cdr' in ARGS.data_path:
        label_dict = {'Chemical': 0, 'Disease': 1}
    else:
        raise NotImplementedError

    config = UCTopicConfig.from_pretrained("studio-ousia/luke-base")
    model = UCTopicCluster(config, ARGS)
    model.load_state_dict(torch.load('result/pytorch_model.bin'), strict= False)
    model.to(DEVICE)
    model.eval()
    
    clustering_data = dataset_reader(ARGS.data_path, label_dict)
    features, labels = get_features(clustering_data, TOKENIZER, model)
    score_factor, score_cosine, cluster_centers = get_kmeans(features, labels, ARGS.num_classes)

    rankings = get_rankings(score_cosine, positive_ratio=0.5)

    pseudo_label_dict = defaultdict(list)

    for i in range(len(rankings)):
        for j in range(len(rankings[i])):
            pseudo_label_dict[clustering_data[rankings[i][j]]['span_lemma']].append(j)
        

    ## majority vote
    for phrase, predictions in pseudo_label_dict.items():
        pseudo_label_dict[phrase] = Counter(predictions).most_common()[0][0]

    model.update_cluster_centers(cluster_centers)

    # dataset loader
    train_loader = get_train_loader(ARGS, pseudo_label_dict)

    # optimizer 
    optimizer = torch.optim.Adam(model.parameters(), lr=ARGS.lr)

    print(optimizer)

    # set up logger
    logger = set_logger(ARGS.save_path)
    global_step = 0
    # set up the trainer 
    evaluate_embedding(clustering_data, TOKENIZER, model, ARGS, global_step, logger)
    learner = ClusterLearner(model, optimizer)
    model.train()
    for epoch in range(ARGS.epoch):
        tqdm_dataloader = tqdm(train_loader, ncols=150)
        for features in tqdm_dataloader:

            for feature in features:
                for k, v in feature.items():
                    feature[k] = v.to(DEVICE)
            loss = learner.forward(features, use_perturbation=ARGS.use_perturbation)

            tqdm_dataloader.set_description(
                'Epoch{}, Global Step {}, CL-loss {:.5f}, clustering-loss {:.5f}, consist-loss {:.5f} '.format(
                    epoch, global_step,  loss['Instance-CL_loss'], loss['clustering_loss'], loss['local_consistency_loss']
                    ))

            update_logger(logger, loss, global_step)
            global_step+=1
            if global_step % ARGS.steps_per_eval == 0:
                evaluate_embedding(clustering_data, TOKENIZER, model, ARGS, global_step, logger)
        model.train()

    print('Final test:')
    evaluate_embedding(clustering_data, TOKENIZER, model, ARGS, global_step, logger)


if __name__ == '__main__':
    main()