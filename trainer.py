from transformers import BertConfig
import torch
import CONFIG
from models.layers import customBERT
from train import Trainer
import os
from utils.common_util import seed_init
from utils.common_util import get_train_parser
import platform


# for reproducibility
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def main(params):
    # init the model
    if params.bert_type == "bert-base-uncased":
        config = BertConfig(label2id=CONFIG.LABEL2ID, id2label=CONFIG.ID2LABEL)
    if params.bert_type == "bert-large-uncased":
        config = BertConfig.from_json_file(os.path.join(CONFIG.BERT_LARGE_PATH, "config.json"))
        config.label2id = CONFIG.LABEL2ID
        config.id2label = CONFIG.ID2LABEL
    device = torch.device(torch.device("cuda:{}".format(params.cuda)))
    model = customBERT(config, params=params).to(device)

    trainer = Trainer(params, model, device)
    trainer.train()


if __name__ == "__main__":
    params = get_train_parser()
    if platform.system() == "Linux":
        seed_init(params.seed)
    main(params)
