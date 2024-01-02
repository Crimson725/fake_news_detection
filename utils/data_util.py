import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import BertTokenizer
import CONFIG

from utils.kg_util import KG_embedding


def get_doc_embedding(doc):
    # Get the token vectors for each word in the document
    token_vectors = []
    for token in doc:
        if token.has_vector:
            token_vectors.append(token.vector)

    # If the document contains no vectors, return zeros
    if not token_vectors:
        return np.zeros(300)

    # Average the token vectors to get the document vector
    return np.mean(token_vectors, axis=0)


class DocDataset(Dataset):
    def __init__(self, dataframe, params, args=None):
        self.data = dataframe
        self.text = dataframe.text
        self.targets = dataframe.label
        self.params = params

        # the triplet
        self.triplet = dataframe.triplet_kg
        if self.params.bert:
            # get tokenizer
            self.tokenizer = BertTokenizer.from_pretrained(CONFIG.BERT_BASE_PATH)
        elif self.params.glove:
            import spacy

            self.tokenizer = spacy.load("en_core_web_lg")
        elif self.params.fasttext:
            import spacy

            self.nlp = spacy.load("en_core_web_lg")
            embeddings_index = {}
            with open(
                "/data2/zhixinzeng/gra_research/fake_news_detection/wiki-news-300d-1M.vec",
                encoding="utf-8",
            ) as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    coefs = np.asarray(values[1:], dtype="float32")
                    embeddings_index[word] = coefs
        if args is None:
            self.max_len = self.params.max_len
            self.kg_generator = KG_embedding()
        else:
            # use saved args for initialization
            self.args = args
            self.max_len = self.args.max_len
            self.kg_generator = KG_embedding()

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())
        if self.params.bert:
            # bert tokenization
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
        elif self.params.glove:
            vec = self.tokenizer(text).vector
        elif self.params.fasttext:
            vec = get_doc_embedding(self.nlp(text))
        triplet_lists = eval(self.triplet[index])
        # using the aggregator in model for performance
        if self.params.hrt:
            # a list of tensors
            hrt_score_list = self.kg_generator.get_triplet_score(triplet_lists)
            hrt_score_list = torch.tensor(hrt_score_list, dtype=torch.float)
        if self.params.hrt_embedding:
            hrt_embedding_list = self.kg_generator.generate_hrt_embedding(triplet_lists)
            hrt_embedding_list = torch.cat(hrt_embedding_list, dim=0)

        if self.params.bert:
            return {
                "ids": torch.tensor(ids, dtype=torch.long),
                "mask": torch.tensor(mask, dtype=torch.long),
                "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
                # single int value label
                "targets": torch.tensor(
                    self.targets[index], dtype=torch.float
                ).unsqueeze(-1),
                # only when hrt is true
                "hrt_score_list": [] if not self.params.hrt else hrt_score_list,
                "hrt_embedding_list": []
                if not self.params.hrt_embedding
                else hrt_embedding_list,
            }
        elif self.params.glove or self.params.fasttext:
            return {
                "embedding": torch.tensor(vec, dtype=torch.float),
                "targets": torch.tensor(
                    self.targets[index], dtype=torch.float
                ).unsqueeze(-1),
                "hrt_score_list": [] if not self.params.hrt else hrt_score_list,
                "hrt_embedding_list": []
                if not self.params.hrt_embedding
                else hrt_embedding_list,
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
        train_dataset = pd.read_csv(os.path.join(CONFIG.DATA_PATH, self.params.dataset))
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
        train_dataset = pd.read_csv(os.path.join(CONFIG.DATA_PATH, self.params.dataset))
        test_dataset = pd.read_csv(
            os.path.join(CONFIG.DATA_PATH, self.params.valid_dataset)
        )
        # get the dataset
        train_set = DocDataset(train_dataset, self.params)
        test_set = DocDataset(test_dataset, self.params)

        # get the sampler
        train_sampler = DistributedSampler(train_set)
        test_sampler = DistributedSampler(test_set, shuffle=False)

        # get the loader
        training_loader = DataLoader(
            train_set,
            **train_loader_params,
            sampler=train_sampler,
            pin_memory=True,
            num_workers=40
        )
        testing_loader = DataLoader(
            test_set,
            **test_loader_params,
            sampler=test_sampler,
            pin_memory=True,
            num_workers=40
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
