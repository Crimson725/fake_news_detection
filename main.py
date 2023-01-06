from transformers import BertConfig

import CONFIG
from eval import Evaluator
from models.layers import customBERT
from train import Trainer
import os
from utils.common_util import seed_everything, Dataloader_train, Dataloader_eval
from utils.common_util import get_parser
import platform
import torch

# set the environment variable
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def train(params):
    trainer = Trainer(params)
    trainer.train_customBERT()


def eval(params):
    config = BertConfig(label2id=CONFIG.LABEL2ID, id2label=CONFIG.ID2LABEL)
    device = torch.device(torch.device("cuda:{}".format(params.cuda)))
    model = customBERT(config, params).to(device)
    loader = Dataloader_eval(params)
    eval_loader = loader.get_loader()
    evaluator = Evaluator(model, testing_loader=eval_loader, params=params)
    evaluator.validation()


if __name__ == "__main__":
    params = get_parser()
    if platform.system() == "Linux":
        seed_everything(params.seed)
    if params.mode == 'train':
        train(params)
    else:
        eval(params)
