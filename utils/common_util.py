import argparse
import random, os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import CONFIG

tokenizer = BertTokenizer.from_pretrained(CONFIG.BERT_PATH)
# bert tokenizer parameters
MAX_SEQ_LEN = 128
PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def save_checkpoint(path, model, valid_loss):
    if path == None:
        return
    state_dict = {"model_state_dict": model.state_dict(), "valid_loss": valid_loss}
    torch.save(state_dict, path)
    print("model saved to ==>{}".format(path))


def load_checkpoint(path, model):
    if path == None:
        return
    state_dict = torch.load(path, map_location=torch.device("cpu"))
    print("loading model from <=={}".format(path))
    model.load_state_dict(state_dict["model_state_dict"])
    return state_dict["valid_loss"]


def save_metrics(path, train_loss_list, valid_loss_list, global_steps_list):
    if path == None:
        return
    state_dict = {
        "train_loss_list": train_loss_list,
        "valid_loss_list": valid_loss_list,
        "global_steps_list": global_steps_list,
    }
    torch.save(state_dict, path)
    print("metrics saved to ==>{}".format(path))


def load_metrics(path):
    if path == None:
        return
    state_dict = torch.load(path, map_location=torch.device("cpu"))
    print("loading metrics from <=={}".format(path))
    return (
        state_dict["train_loss_list"],
        state_dict["valid_loss_list"],
        state_dict["global_steps_list"],
    )


class CustomDataset(Dataset):
    # TODO: ADD THE PATH FEATURE
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.targets = self.data.label
        self.max_len = max_len

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


class Dataloader_train:
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
            train_set = CustomDataset(train_dataset, tokenizer, self.params.max_len)
            test_set = CustomDataset(test_dataset, tokenizer, self.params.max_len)
            training_loader = DataLoader(train_set, **train_loader_params)
            testing_loader = DataLoader(test_set, **test_loader_params)
        else:
            train_dataset = pd.read_csv(
                os.path.join(CONFIG.DATA_PATH, self.params.dataset)
            )
            test_dataset = pd.read_csv(
                os.path.join(CONFIG.DATA_PATH, self.params.valid_dataset)
            )
            train_set = CustomDataset(train_dataset, tokenizer, self.params.max_len)
            test_set = CustomDataset(test_dataset, tokenizer, self.params.max_len)
            training_loader = DataLoader(train_set, **train_loader_params)
            testing_loader = DataLoader(test_set, **test_loader_params)
        return training_loader, testing_loader


class Dataloader_eval:
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
        eval_set = CustomDataset(eval_dataset, tokenizer, self.train_args.max_len)
        eval_loader = DataLoader(eval_set, **eval_loader_params)
        return eval_loader


def get_train_parser():
    argparser = argparse.ArgumentParser(
        description="Arg parser for fake news detection. Implemented model: BERT, TextCNN",
        epilog='For example: python main.py --seed 42 --cuda 5 --dataset "LUN/lun_train_comparenet.csv"  --valid_dataset "LUN/lun_test_comparenet.csv" --weight_decay 0.5 --epochs 5 --lstm True --multihead_attention True',
    )

    argparser.add_argument("--seed", type=int, default=42, help="seed")
    argparser.add_argument("--cuda", type=int, default=0, help="device id")

    argparser.add_argument(
        "--dataset", type=str, default="real_and_fake/train.csv", help="dataset"
    )
    argparser.add_argument(
        "--valid_enable",
        action="store_true",
        default=True,
        help="validation using another dataset",
    )
    valid_enable_group = argparser.add_mutually_exclusive_group(required=True)
    valid_enable_group.add_argument(
        "--frac",
        type=float,
        default=None,
        help="the fraction of the dataset to use for validation (only when valid_enable is False)",
    )
    valid_enable_group.add_argument(
        "--valid_dataset",
        type=str,
        default=None,
        help="the path to the validation dataset (only when valid_enable is True)",
    )
    argparser.add_argument(
        "--train_eval", type=bool, default=False, help="train and evaluate"
    )

    argparser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    argparser.add_argument(
        "--train_batch", type=int, default=8, help="training set batch size"
    )
    argparser.add_argument(
        "--test_batch", type=int, default=8, help="validation set batch size"
    )
    argparser.add_argument(
        "--eval_every", type=int, default=5, help="evaluate every n step"
    )
    argparser.add_argument(
        "--max_len", type=int, default=512, help="max length to padding"
    )
    argparser.add_argument("--epochs", type=int, default=1, help="epoch of training ")
    argparser.add_argument("--lstm", type=bool, default=False, help="use lstm")
    argparser.add_argument(
        "--multihead_attention",
        type=bool,
        default=False,
        help="use multihead attention",
    )
    argparser.add_argument(
        "--dropout", type=float, default=0.5, help="dropout rate of the model"
    )
    argparser.add_argument(
        "--early_stop_patience", type=int, default=5, help="early stop patience"
    )

    argparser.add_argument(
        "--weight_decay", type=float, default=0.0, help="weight decay of adam"
    )
    argparser.add_argument(
        "--model_name", type=str, default="customBERT", help="name of the model"
    )

    argparser.add_argument(
        "--log_args",
        type=bool,
        default=True,
        help="log the args of the training process",
    )
    argparser.add_argument(
        "--verbose",
        type=bool,
        default=True,
        help="log verbose (loss) info of the training process",
    )
    argparser.add_argument(
        "--print_logs",
        type=bool,
        default=False,
        help="print the verbose info of the training process",
    )
    args = argparser.parse_args()
    return args


def get_eval_parser():
    argparser = argparse.ArgumentParser(
        description="Arg parser for fake news detection. Implemented model: BERT, TextCNN",
        epilog="For example:",
    )
    argparser.add_argument("--seed", type=int, default=42, help="seed")
    argparser.add_argument("--cuda", type=int, default=0, help="device id")
    argparser.add_argument(
        "--model_path", type=str, default=None, help="path to the model"
    )
    argparser.add_argument(
        "--eval_dataset", type=str, default=None, help="path to the evaluation dataset"
    )

    args = argparser.parse_args()
    return args
