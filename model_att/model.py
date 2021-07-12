from model_att import feature
import utils
import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import consts
import random
from consts import DEVICE
from pathlib import Path
from model_base import BaseModel
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler
from pytorch_partial_crf import PartialCRF
import collections

class LSTMFuzzyCRFModel(BaseModel):
    mp = {'P': -1, 'B': 2, 'I': 1, 'O': 0}
    def __init__(self, num_features, model_dir):
        super().__init__(model_dir)
        self.kernel_size1 = (1, 2)
        self.height_after_pool1 = consts.MAX_SENT_LEN+2 - self.kernel_size1[0] + 1  # seq_len  +2 because of bos and eos token
        self.width_after_pool1 = consts.MAX_SENT_LEN+2 - self.kernel_size1[1] + 1 # seq_len - 1

        self.kernel_size2 = (1, self.width_after_pool1)
        self.height_after_pool2 = self.height_after_pool1 - self.kernel_size2[0] + 1 # seq_len
        self.width_after_pool2 = self.width_after_pool1 - self.kernel_size2[1] + 1  # 1

        self.in_channels = num_features
        self.out_channels = num_features

        self.cnn1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size1)
        self.cnn2 = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=self.kernel_size2)
        
        self.lstm = nn.LSTM(self.out_channels, self.out_channels, num_layers=1, batch_first=True, bidirectional=True)

        self.hidden2tag = nn.Linear(2 * self.out_channels, len(LSTMFuzzyCRFModel.mp) - 1)
        self.crf = PartialCRF(len(LSTMFuzzyCRFModel.mp) - 1) # BIO Phrases

    def config(self):
        return {
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'kernel_size1': self.kernel_size1,
            'kernel_size2': self.kernel_size2
        }

    @staticmethod
    def from_config(config_or_path_or_dir):
        assert False

    # @staticmethod
    # def _labels_to_biop(phrase_idxs, seq_len):
        
    #     BIO = ["P"] * seq_len
    #     positive_indices = []

    #     for start, end in phrase_idxs: #[start, end]
            
    #         assert BIO[start] == 'P'
    #         BIO[start] = "B"
    #         positive_indices.append(start)
    #         for i in range(start + 1, end + 1):
    #             assert BIO[i] == 'P'
    #             BIO[i] = 'I'
    #             positive_indices.append(i)

    #     all_indices = list(range(len(BIO)))
    #     possible_negative = set(all_indices) - set(positive_indices)
    #     num_negative = min(len(possible_negative), int(len(positive_indices) * consts.NEGATIVE_RATIO))
    #     sampled_negative = random.sample(possible_negative, k=num_negative)
    #     for i in sampled_negative:
    #         assert BIO[i] == 'P'
    #         BIO[i] = 'O'
    #     return [LSTMFuzzyCRFModel.mp[x] for x in BIO]

    @staticmethod
    def _decode_bio(labels):
        phrases = []
        in_phrase = False
        for i, label in enumerate(labels + [0]):
            if label == 0: # O
                if in_phrase:
                    in_phrase = False
                    phrases[-1][1] = i # end)
            elif label == 1: # I
                if not in_phrase: # error, fallback to B
                    in_phrase = True
                    phrases.append([i, -1])
            else: # B
                if in_phrase:
                    in_phrase = False
                    phrases[-1][1] = i
                in_phrase = True
                phrases.append([i, -1]) # [start
        # convert to spans, lazy zzZ
        predictions = [0] * len(labels)
        for start, end in phrases:
            if end == start + 1:
                continue
            for i in range(start, end):
                predictions[i] = 1
        return predictions

    @staticmethod
    def _decode_spans(labels):
        phrases = []
        in_phrase = False
        for i, label in enumerate(labels + [0]):
            if label == 0: # O
                if in_phrase:
                    in_phrase = False
                    phrases[-1][1] = i - 1 # end]
            elif label == 1: # I
                if not in_phrase: # error, fallback to B
                    in_phrase = True
                    phrases.append([i, -1])
            else: # B
                if in_phrase:
                    in_phrase = False
                    phrases[-1][1] = i - 1 # end]
                in_phrase = True
                phrases.append([i, -1]) #[start

        return phrases #[start, end]

    # @staticmethod
    # def pad_features(features, phrase_idxs=None):
        
    #     num_instances = len(features)
    #     num_features = features[0].shape[-1]
    #     max_seq_len = max(_feature.shape[0] for _feature in features)
    #     padded_features = numpy.zeros((num_instances, max_seq_len, max_seq_len, num_features), dtype=numpy.float16)
    #     padded_labels = numpy.zeros((num_instances, max_seq_len), dtype=numpy.int) - 1
    #     for i, _feature in enumerate(features):
    #         _len = _feature.shape[0]
    #         padded_features[i, :_len, :_len, :] = _feature
    #         if phrase_idxs is not None:
    #             padded_labels[i, :_len] = LSTMFuzzyCRFModel._labels_to_biop(phrase_idxs[i], _len)
    #     return torch.tensor(padded_features, dtype=torch.float32), torch.tensor(padded_labels, dtype=torch.long)

    def forward(self, features):
        """
        Args:
            features: [batch_size, num_features, max_seq_len, max_seq_len]
        """
        batch_size, num_features, seq_len, _ = features.size()
        x = features
        x = F.relu(self.cnn1(x))
     
        assert x.shape == (batch_size, self.out_channels, self.height_after_pool1, self.width_after_pool1)
        x = F.relu(self.cnn2(x)) #(batch, out_channels, seq_len, 1)
        assert x.shape == (batch_size, self.out_channels, self.height_after_pool2, self.width_after_pool2)
        x = x.squeeze().transpose(2, 1) #(batch, seq_len, out_channels)
        x = self.dropout(x)
        word_features, _ = self.lstm(x) #(batch, seq_len, out_channels * 2)
        assert word_features.size() == (batch_size, seq_len, 2 * self.out_channels)
        word_features = self.dropout(word_features)

        return self.hidden2tag(word_features)

    def get_loss(self, features, labels):
        emissions = self.forward(features)
        loss = self.crf(emissions, labels)
        return loss

    def get_probs(self, features):
        emissions = self.forward(features)
        tags = self.crf.viterbi_decode(emissions)
        return [LSTMFuzzyCRFModel._decode_bio(tag) for tag in tags]

    def get_spans(self, features):
        emissions = self.forward(features)
        tags = self.crf.viterbi_decode(emissions)
        return [LSTMFuzzyCRFModel._decode_spans(tag) for tag in tags]

    def _predict_padder(self, all_features, batch_size):
        word_features_fw, word_features_index_fw, word_features_bw, word_features_index_bw, _ = self.pad_features(all_features)
        self.eval()
        dataset = TensorDataset(word_features_fw, word_features_index_fw, word_features_bw, word_features_index_bw,
                                torch.tensor([len(feature["word_features_fw"]) for feature in all_features]))
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
        with torch.no_grad():
            for word_features_fw, word_features_index_fw, word_features_bw, word_features_index_bw, length in tqdm(dataloader, total=len(dataloader), ncols=100):
                word_features_fw = word_features_fw.to(DEVICE)
                word_features_index_fw = word_features_index_fw.to(DEVICE)
                word_features_bw = word_features_bw.to(DEVICE)
                word_features_index_bw = word_features_index_bw.to(DEVICE)
                probs = self.get_probs(
                    (word_features_fw, word_features_index_fw, word_features_bw, word_features_index_bw))
                # print(probs)
                # exit()
                for prob, le in zip(probs, length.numpy()):
                    if le == 0:
                        print("Some is 0")
                        yield numpy.zeros(1, dtype=numpy.int)  # dummy
                    else:
                        yield numpy.array(prob[: le - 1], dtype=numpy.float32)

    # @profile
    def predict(self, path_tokenized_id_corpus, dir_output, loader, batch_size=128):
        utils.Log.info(f'Generate prediction features...')
        docs, test_data = loader.load_test_data(path_tokenized_id_corpus)

        utils.Log.info(f'Predict: {dir_output}')
        self.eval()
        dir_output = Path(dir_output)
        path_output = dir_output / path_tokenized_id_corpus.name
        dir_output.mkdir(exist_ok=True)
        path_output.parent.mkdir(exist_ok=True)

        doc_spans = []
        for features in tqdm(test_data, ncols=100, desc='predict'):

            spans = self.get_spans(features)
            doc_spans += spans

        assert len(docs) == len(doc_spans)

        doc_predicted_instance_dict = collections.defaultdict(list)
        for doc, spans in zip(docs, doc_spans):

            _id_, ids, widxs = doc
            length = len(ids)
            if length == 0:
                spans = []
            else:
                spans = [[span[0]-1, span[1]-1] for span in spans] # -1 because of bos_token
            doc_predicted_instance_dict[_id_].append(
                {'spans': spans,
                'ids': ids}
            )

        predicted_docs = []
        for _id_, prediction in doc_predicted_instance_dict.items():
            predicted_docs.append({
                '_id_': _id_,
                'sents': prediction})

        utils.Pickle.dump(predicted_docs, path_output)
        return path_output
