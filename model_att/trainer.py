import utils
import torch
import random
import consts
from consts import DEVICE
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
from pathlib import Path
from tqdm import tqdm
import numpy as np
from model_att.model import LSTMFuzzyCRFModel


class AttmapDataset(Dataset):
    def __init__(self, instances, is_train=True):

        self.dataset = instances
        self.is_train = is_train

    def __len__(self):

        return len(self.dataset)

    def __getitem__(self, index):
        
        instance = self.dataset[index]
        input_id = instance['ids']
        _input_id = [consts.LM_TOKENIZER.bos_token_id] + input_id + [consts.LM_TOKENIZER.eos_token_id]
        if not self.is_train:
            return _input_id

        phrase_idxs = [[phrase[0][0]+1, phrase[0][1]+1] for phrase in instance['phrases']]  # +1 because if bos_token
        BIO = ["P"] * len(_input_id)
        positive_indices = []
        
        for start, end in phrase_idxs: #[start, end]
            #assert BIO[start] == 'P'
            if BIO[start] == 'P':
                BIO[start] = "B"
                positive_indices.append(start)
            for i in range(start+1, end + 1):
                #assert BIO[i] == 'P'
                if BIO[i] == 'P':
                    BIO[i] = 'I'
                    positive_indices.append(i)

        all_indices = list(range(len(BIO)))
        possible_negative = set(all_indices) - set(positive_indices)
        num_negative = min(len(possible_negative), int(len(positive_indices) * consts.NEGATIVE_RATIO))
        sampled_negative = random.sample(possible_negative, k=num_negative)
        for i in sampled_negative:
            assert BIO[i] == 'P'
            BIO[i] = 'O'

        label = [LSTMFuzzyCRFModel.mp[x] for x in BIO]

        return _input_id, label
    
    def collate_fn(self, data):

        if self.is_train:
            batch_input_id, batch_label = zip(*data)
        else:
            batch_input_id = data
            

        batch_size = len(batch_input_id)
        max_seq_len = consts.MAX_SENT_LEN+2 ## for bos and eos tokens

        padded_id = np.full((batch_size, max_seq_len), consts.LM_TOKENIZER.pad_token_id, dtype=np.int)
        padded_mask = np.full((batch_size, max_seq_len), 0, dtype=np.int)
        if self.is_train:
            padded_label = np.full((batch_size, max_seq_len), LSTMFuzzyCRFModel.mp['P'], dtype=np.int)

        if self.is_train:
            for i, (input_id, label) in enumerate(zip(batch_input_id, batch_label)):
                _len = len(input_id)
                padded_id[i][:_len] = input_id
                padded_mask[i][:_len] = 1
                padded_label[i][:_len] = label
        else:
            for i, input_id in enumerate(batch_input_id):
                _len = len(input_id)
                padded_id[i][:_len] = input_id
                padded_mask[i][:_len] = 1


        padded_id = torch.tensor(padded_id, device=consts.DEVICE)
        padded_mask = torch.tensor(padded_mask, device=consts.DEVICE)
        if self.is_train:
            padded_label = torch.tensor(padded_label, device=consts.DEVICE)

        with torch.no_grad():
            model_output = consts.LM_MODEL(padded_id, attention_mask=padded_mask,
                                            output_hidden_states=False,
                                            output_attentions=True,
                                            return_dict=True)

            batch_attentions = model_output.attentions  # layers, [batch_size, num_heads, seqlen, seqlen]
            batch_attentions = torch.stack(batch_attentions, dim=0)[:consts.CONFIG.num_lm_layers]  # [layers, batch_size, num_heads, seqlen, seqlen]
            batch_attentions = batch_attentions.transpose(0, 1).reshape(batch_size, -1, max_seq_len, max_seq_len)

        if self.is_train:
            return [batch_attentions, padded_label]
        else:
            return batch_attentions

class AttmapLoader:
    def __init__(self, random_seed=42):
        self.random_seed = random_seed

    def get_batch_size(self):
        return 512

    def get_loader(self, instances, is_train=True):
        batch_size = self.get_batch_size()
        dataset = AttmapDataset(instances, is_train=is_train)
        sampler = RandomSampler(dataset) if is_train else SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, collate_fn=dataset.collate_fn, num_workers=0)
        return dataloader

    def load_train_data(self, filepath, sample_ratio=-1):
        print('Loading training data...',)
        filepath = Path(filepath)
        sampled_docs = utils.OrJsonLine.load(filepath)
        instances = [sent for doc in sampled_docs for sent in doc['sents']]
        print(f'OK! {len(instances)} training instances')

        if sample_ratio > 0.0:
            assert sample_ratio < 1.0
            num_instances = int(sample_ratio * len(instances))
            instances = random.choices(instances, k=num_instances)
            print(f'[Trainer] Sampled {len(instances)} instances.')

        train_instances, valid_instances = train_test_split(instances, test_size=0.1, shuffle=True, random_state=self.random_seed)
        return self.get_loader(train_instances), self.get_loader(valid_instances)

    def load_test_data(self, filepath):

        print('Loading test data...',)
        filepath = Path(filepath)
        sampled_docs = utils.OrJsonLine.load(filepath)
        instances = [sent for doc in sampled_docs for sent in doc['sents']]
        docs = [(doc['_id_'], sent['ids'], sent['widxs']) for doc in sampled_docs for sent in doc['sents']]
        print(f'OK! {len(instances)} test instances')

        return docs, self.get_loader(instances, is_train=False)

class AttmapTrainer:
    def __init__(self, model: LSTMFuzzyCRFModel, sample_ratio=-1):
        self.sample_ratio = sample_ratio

        self.model = model.to(DEVICE)
        self.output_dir = model.model_dir
        self.train_loader = AttmapLoader()
        self.optimizer = Adam(self.model.parameters(), lr=1e-3)
        model_config_path = self.output_dir / 'model_config.json'
        utils.Json.dump(self.model.config(), model_config_path)

    def train(self, path_train_data, num_epochs=20):
        
        utils.Log.info(f'Start training: {path_train_data}')
        train_data, valid_data = self.train_loader.load_train_data(path_train_data, sample_ratio=self.sample_ratio)
        num_train = len(train_data)
        num_valid = len(valid_data)

        best_epoch = -1
        best_valid_f1 = -1.0
        for epoch in range(1, num_epochs + 1):
            utils.Log.info(f'Epoch [{epoch} / {num_epochs}]')

            # Train
            self.model.train()
            epoch_loss = 0
            for attmap_features, gtlabels in tqdm(train_data, total=num_train, ncols=100):
                self.model.zero_grad()
                batch_loss = self.model.get_loss(attmap_features, gtlabels)
                epoch_loss += batch_loss.item()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            train_loss = epoch_loss / num_train
            utils.Log.info(f'Train loss: {train_loss}')

            # Valid
            self.model.eval()
            gold_labels = []
            pred_labels = []
            with torch.no_grad():
                for attmap_features, gtlabels in tqdm(valid_data, total=num_valid, ncols=100):
                    pred_probs = self.model.get_probs(attmap_features)#.detach().cpu()
                    pred_probs = torch.tensor(pred_probs)
                    gtlabels = gtlabels.detach().cpu()
                    gtlabels = torch.tensor([LSTMFuzzyCRFModel._decode_bio(label.numpy().tolist()) for label in gtlabels], dtype=gtlabels.dtype)
                    gtlabels = gtlabels.flatten()
                    pred_probs = pred_probs.flatten()
                    not_mask = (gtlabels + 1).nonzero()
                    gold_labels.extend(gtlabels[not_mask].flatten().numpy().tolist())
                    pred_labels.extend([int(p > .5) for p in pred_probs[not_mask].flatten().numpy().tolist()])
            valid_f1 = f1_score(gold_labels, pred_labels, average="micro")
            utils.Log.info(f'valid f1: {valid_f1}')
            if valid_f1 < best_valid_f1:
                utils.Log.info(f'Stop training. Best epoch: {epoch - 1}')
                break
            best_epoch = epoch - 1
            best_valid_f1 = valid_f1

            ckpt_dict = {
                'epoch': epoch,
                'model': self.model,
                'valid_f1': valid_f1,
                'train_loss': train_loss,
            }
            ckpt_path = self.output_dir / f'epoch-{epoch}.ckpt'
            torch.save(ckpt_dict, ckpt_path)

        return best_epoch

