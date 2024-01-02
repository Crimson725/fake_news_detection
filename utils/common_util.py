import argparse
import random, os
import numpy as np
import torch
from torch.backends import cudnn
import torch.distributed as dist
import re
import datetime
from collections import OrderedDict

import CONFIG


def seed_init(seed):
    # random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True


def DDP_seed_init(seed, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


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
    new_state_dict = OrderedDict()
    for k, v in state_dict["model_state_dict"].items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
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


def get_path(name):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_path = os.path.join(CONFIG.DESTINATION_PATH, timestamp + "_" + name)
    tf_path = os.path.join(file_path, "tf_logs")
    if not os.path.exists(tf_path):
        os.makedirs(tf_path)
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    return file_path, tf_path


def get_DDP_path(name):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_path = os.path.join(CONFIG.DDP_DESTINATION_PATH, timestamp + "_" + name)
    tf_path = os.path.join(file_path, "tf_logs")
    if dist.get_rank() == 0:
        if not os.path.exists(tf_path):
            os.makedirs(tf_path)
        if not os.path.exists(file_path):
            os.makedirs(file_path)
    return file_path, tf_path


def get_eval_parser():
    argparser = argparse.ArgumentParser(
        description="Arg parser for fake news detection.",
    )
    argparser.add_argument("--seed", type=int, default=42, help="seed")
    argparser.add_argument("--cuda", type=int, default=0, help="cuda device")
    argparser.add_argument(
        "--model_path", type=str, default=None, help="path to the model"
    )
    argparser.add_argument(
        "--eval_dataset", type=str, default=None, help="path to the evaluation dataset"
    )

    exclusive_group = argparser.add_mutually_exclusive_group(required=True)
    exclusive_group.add_argument(
        "--hrt", action="store_true", help="use triplet scoring"
    )
    exclusive_group.add_argument("--raw", action="store_true", help="use raw text")

    argparser.add_argument(
        "--hrt_embedding", action="store_true", help="use hrt embedding"
    )
    exclusive_group_embed = argparser.add_mutually_exclusive_group(required=True)

    exclusive_group_embed.add_argument(
        "--bert", action="store_true", help="use bert embedding"
    )
    exclusive_group_embed.add_argument(
        "--glove", action="store_true", help="use glove embedding"
    )
    exclusive_group_embed.add_argument(
        "--fasttext", action="store_true", help="use fasttext embedding"
    )
    args = argparser.parse_args()

    return args


def get_train_parser_DDP():
    argparser = argparse.ArgumentParser(
        description="Arg parser for fake news detection using DDP training. Model Used: bert-base-uncased; Entity Embedding: CSKG dataset and TransE"
    )

    argparser.add_argument("--seed", type=int, default=42, help="seed")
    exclusive_group = argparser.add_mutually_exclusive_group(required=True)
    exclusive_group_embed = argparser.add_mutually_exclusive_group(required=True)
    exclusive_group.add_argument(
        "--hrt", action="store_true", help="use triplet scoring"
    )
    exclusive_group.add_argument("--raw", action="store_true", help="use raw text")
    exclusive_group_embed.add_argument(
        "--bert", action="store_true", help="use bert embedding"
    )
    exclusive_group_embed.add_argument(
        "--glove", action="store_true", help="use glove embedding"
    )
    exclusive_group_embed.add_argument(
        "--fasttext", action="store_true", help="use fasttext embedding"
    )

    argparser.add_argument(
        "--hrt_embedding", action="store_true", help="use hrt embedding"
    )
    argparser.add_argument(
        "--dataset", type=str, default="real_and_fake/train.csv", help="dataset"
    )

    argparser.add_argument(
        "--valid_dataset",
        type=str,
        default="real_and_fake/test.csv",
        help="the path to the validation dataset (only when valid_enable is True)",
    )
    argparser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    argparser.add_argument(
        "--train_batch", type=int, default=12, help="training set batch size"
    )
    argparser.add_argument(
        "--test_batch", type=int, default=12, help="validation set batch size"
    )
    argparser.add_argument(
        "--eval_every", type=int, default=50, help="evaluate every n step"
    )
    argparser.add_argument(
        "--max_len", type=int, default=512, help="max length to padding (For doc mode)"
    )
    argparser.add_argument("--epochs", type=int, default=3, help="epoch of training ")
    argparser.add_argument(
        "--dropout", type=float, default=0.5, help="dropout rate of the model"
    )

    argparser.add_argument(
        "--weight_decay", type=float, default=1e-5, help="weight decay of adam"
    )
    args = argparser.parse_args()

    match = re.search(r"^(.+?)/", args.dataset)
    args.model_name = match.group(1)
    return args


def get_inf_parser():
    argparser = argparse.ArgumentParser(
        description="Arg parser for fake news detection.",
    )
    argparser.add_argument("--cuda", type=int, default=0, help="cuda device")
    argparser.add_argument(
        "--model_path", type=str, default=None, help="path to the model"
    )
