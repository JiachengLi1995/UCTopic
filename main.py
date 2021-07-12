import torch
import evaluate
import utils
import consts
import model_att
import model_base
from tqdm import tqdm
from consts import ARGS
from pathlib import Path
from preprocess import Preprocessor
from preprocess import BaseAnnotator
from preprocess import WikiAnnotator
from preprocess import CoreAnnotator


class Experiment:
    rootdir = Path('../experiments')
    rootdir.mkdir(exist_ok=True)

    def __init__(self):
        self.config = consts.CONFIG
        # establish experiment folder
        self.exp_name = f'{consts.DIR_DATA.stem}-{consts.LM_NAME_SUFFIX}'
        if ARGS.exp_prefix:
            self.exp_name += f'.{ARGS.exp_prefix}'
        self.dir_exp = self.rootdir / self.exp_name
        self.dir_exp.mkdir(exist_ok=True)
        utils.Json.dump(self.config.todict(), self.dir_exp / 'config.json')

        # preprocessor
        self.train_preprocessor = Preprocessor(
            path_corpus=self.config.path_train,
            num_cores=consts.NUM_CORES,
            use_cache=True
        )

        # annotator (supervision)
        self.train_annotator: BaseAnnotator = {
            'wiki': WikiAnnotator(
                use_cache=True,
                preprocessor=self.train_preprocessor,
                path_standard_phrase=self.config.path_phrase
            ),
            'core': CoreAnnotator(
                use_cache=True,
                preprocessor=self.train_preprocessor
            )
        }[self.config.annotator]

        # model
        model_prefix = '.' + ARGS.model_prefix if ARGS.model_prefix else ''
        model_dir = self.dir_exp / f'model{model_prefix}'

        model = model_att.LSTMFuzzyCRFModel(
            model_dir=model_dir,
            num_features=self.config.num_lm_layers * 12)

        self.trainer = model_att.AttmapTrainer(model=model, sample_ratio=0.01)
        
    def train(self, num_epochs=20):
        self.train_preprocessor.tokenize_corpus()
        path_train_data = self.train_annotator.mark_corpus()
        self.trainer.train(path_train_data=path_train_data, num_epochs=num_epochs)

    def select_best_epoch(self):
        paths_ckpt = [p for p in self.trainer.output_dir.iterdir() if p.suffix == '.ckpt']
        best_epoch = None
        best_valid_f1 = 0.0
        for p in paths_ckpt:
            ckpt = torch.load(p, map_location='cpu')
            if ckpt['valid_f1'] > best_valid_f1:
                best_valid_f1 = ckpt['valid_f1']
                best_epoch = ckpt['epoch']
        utils.Log.info(f'Best epoch: {best_epoch}. F1: {best_valid_f1}')
        return best_epoch

    def predict(self, epoch):
        test_preprocessor = None
        test_preprocessor = Preprocessor(
            path_corpus=self.config.path_tagging_docs,
            num_cores=consts.NUM_CORES,
            use_cache=True)

        
        test_preprocessor.tokenize_corpus()

        ''' Model Predict '''
        dir_prefix = 'tagging.'
        dir_predict = self.trainer.output_dir / f'{dir_prefix}predict.epoch-{epoch}'
        path_ckpt = self.trainer.output_dir / f'epoch-{epoch}.ckpt'
        model: model_base.BaseModel = model_base.BaseModel.load_ckpt(path_ckpt).eval().to(consts.DEVICE)

        data_loader = self.trainer.train_loader
        path_predicted_docs = model.predict(
            path_tokenized_id_corpus=test_preprocessor.path_tokenized_id_corpus, 
            dir_output=dir_predict,
            loader=data_loader,
            batch_size=128)

        ''' Model Decode and Evaluate'''
        dir_decoded = self.trainer.output_dir / f'{dir_prefix}decoded.epoch-{epoch}'
        dir_decoded.mkdir(exist_ok=True)

        path_decoded_doc2sents = model_base.BaseModel.decode(
            path_predicted_docs=path_predicted_docs,
            output_dir=dir_decoded,
            threshold=0, #self.config.threshold,
            use_cache=True,
            use_tqdm=True
        )
        evaluator = evaluate.SentEvaluator()
        paths_gold = self.config.paths_tagging_human
        print(f'Evaluate {path_decoded_doc2sents}')
        print(evaluator.evaluate(path_decoded_doc2sents, paths_doc2golds=paths_gold))


if __name__ == '__main__':
    exp = Experiment()
    exp.train()
    best_epoch = exp.select_best_epoch()
    print('Best Epoch:', best_epoch)
    exp.predict(epoch=best_epoch)
