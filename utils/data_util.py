import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import BertTokenizer
import spacy
from fastcoref import spacy_component
import CONFIG


class Coref:
    def __init__(self):
        self.nlp = spacy.load(
            "en_core_web_sm", exclude=["parser", "lemmatizer", "ner", "textcat"]
        )
        self.nlp.add_pipe(
            "fastcoref",
            config={
                "model_architecture": "LingMessCoref",
                "model_path": "biu-nlp/lingmess-coref",
                "device": "cuda:0",
            },
        )

    def get_resolved_text(self, text):
        res = self.nlp(
            text, component_cfg={"fastcoref": {"resolve_text": True}}
        )._.resolved_text
        return res


class DocDataset(Dataset):
    def __init__(self, dataframe, params, args=None):
        self.data = dataframe
        self.text = dataframe.text
        self.targets = dataframe.label
        self.params = params
        if args is None:
            self.max_len = self.params.max_len
            # get tokenizer
            if self.params.bert_type == "bert-base-uncased":
                self.tokenizer = BertTokenizer.from_pretrained(CONFIG.BERT_BASE_PATH)
            if self.params.bert_type == "bert-large-uncased":
                self.tokenizer = BertTokenizer.from_pretrained(CONFIG.BERT_LARGE_PATH)
        else:
            self.args = args
            self.max_len = self.args.max_len
            # get tokenizer
            if self.args.bert_type == "bert-base-uncased":
                self.tokenizer = BertTokenizer.from_pretrained(CONFIG.BERT_BASE_PATH)
            if self.args.bert_type == "bert-large-uncased":
                self.tokenizer = BertTokenizer.from_pretrained(CONFIG.BERT_LARGE_PATH)

        # bert tokenizer parameters
        # MAX_SEQ_LEN = 128
        # PAD_INDEX = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        # UNK_INDEX = self.tokenizer.convert_tokens_to_ids(self.tokenizer.unk_token)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_token_type_ids=True,
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            # single int value label
            "targets": torch.tensor(self.targets[index], dtype=torch.float).unsqueeze(
                -1
            ),
        }


class loader_train:
    def __init__(self, params):
        self.params = params

    def get_loader(self):
        def seed_worker(worker_id):
            worker_seed = self.params.seed
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(self.params.seed)

        train_loader_params = {
            "batch_size": self.params.train_batch,
            "shuffle": True,
            "worker_init_fn": seed_worker,
            "generator": g,
        }
        test_loader_params = {
            "batch_size": self.params.test_batch,
            "shuffle": True,
            "worker_init_fn": seed_worker,
            "generator": g,
        }
        # dataset settings
        if self.params.valid_enable is False:
            train_size = self.params.frac
            df = pd.read_csv(os.path.join(CONFIG.DATA_PATH, self.params.dataset))
            train_dataset = df.sample(frac=train_size, random_state=200).reset_index(
                drop=True
            )
            test_dataset = df.drop(train_dataset.index).reset_index(drop=True)

            # get the train set and test set
            train_set = DocDataset(train_dataset, self.params)
            test_set = DocDataset(test_dataset, self.params)
            training_loader = DataLoader(train_set, **train_loader_params)
            testing_loader = DataLoader(test_set, **test_loader_params)
        else:
            train_dataset = pd.read_csv(
                os.path.join(CONFIG.DATA_PATH, self.params.dataset)
            )
            test_dataset = pd.read_csv(
                os.path.join(CONFIG.DATA_PATH, self.params.valid_dataset)
            )
            train_set = DocDataset(train_dataset, self.params)
            test_set = DocDataset(test_dataset, self.params)
            training_loader = DataLoader(train_set, **train_loader_params)
            testing_loader = DataLoader(test_set, **test_loader_params)
        return training_loader, testing_loader


class DDP_loader_train:
    def __init__(self, params):
        self.params = params

    def get_loader(self):
        def seed_worker(worker_id):
            worker_seed = self.params.seed
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(self.params.seed)

        train_loader_params = {
            "batch_size": self.params.train_batch // 2,
            "shuffle": False,
            "worker_init_fn": seed_worker,
            "generator": g,
        }
        test_loader_params = {
            "batch_size": self.params.test_batch // 2,
            "shuffle": False,
            "worker_init_fn": seed_worker,
            "generator": g,
        }
        # dataset settings
        if self.params.valid_enable is False:
            train_size = self.params.frac
            df = pd.read_csv(os.path.join(CONFIG.DATA_PATH, self.params.dataset))
            train_dataset = df.sample(frac=train_size, random_state=200).reset_index(
                drop=True
            )
            test_dataset = df.drop(train_dataset.index).reset_index(drop=True)

            # get the train set and test set
            train_set = DocDataset(train_dataset, self.params)
            test_set = DocDataset(test_dataset, self.params)

            # get the sampler
            train_sampler = DistributedSampler(train_set)
            test_sampler = DistributedSampler(test_set)

            # get the dataloader
            training_loader = DataLoader(
                train_set, **train_loader_params, sampler=train_sampler
            )
            testing_loader = DataLoader(
                test_set, **test_loader_params, sampler=test_sampler
            )
        else:
            train_dataset = pd.read_csv(
                os.path.join(CONFIG.DATA_PATH, self.params.dataset)
            )
            test_dataset = pd.read_csv(
                os.path.join(CONFIG.DATA_PATH, self.params.valid_dataset)
            )
            # get the dataset
            train_set = DocDataset(train_dataset, self.params)
            test_set = DocDataset(test_dataset, self.params)

            # get the sampler
            train_sampler = DistributedSampler(train_set)
            test_sampler = DistributedSampler(test_set)

            # get the loader
            training_loader = DataLoader(
                train_set, **train_loader_params, sampler=train_sampler
            )
            testing_loader = DataLoader(
                test_set, **test_loader_params, sampler=test_sampler
            )
        return training_loader, testing_loader, train_sampler, test_sampler


class loader_eval:
    def __init__(self, params, train_args):
        self.params = params
        self.train_args = train_args

    def get_loader(self):
        def seed_worker(worker_id):
            worker_seed = self.params.seed
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(self.params.seed)
        eval_loader_params = {
            "batch_size": self.train_args.test_batch,
            "shuffle": True,
            "worker_init_fn": seed_worker,
            "generator": g,
        }
        eval_dataset = pd.read_csv(
            os.path.join(CONFIG.DATA_PATH, self.params.eval_dataset)
        )
        eval_set = DocDataset(eval_dataset, self.params, args=self.train_args)
        eval_loader = DataLoader(eval_set, **eval_loader_params)
        return eval_loader
