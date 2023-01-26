from transformers import BertConfig

import CONFIG
from models.layers import customBERT
from train import Trainer
import os
from utils.common_util import ddp_seed_init
from utils.common_util import get_train_parser
import torch.distributed as dist
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

# for reproducibility
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


dist.init_process_group(backend="nccl")
rank = dist.get_rank()
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
# set device (using local rank)
torch.cuda.set_device(LOCAL_RANK)


def main(params):
    # init the model
    if params.bert_type == "bert-base-uncased":
        config = BertConfig(label2id=CONFIG.LABEL2ID, id2label=CONFIG.ID2LABEL)
    if params.bert_type == "bert-large-uncased":
        config = BertConfig.from_json_file(os.path.join(CONFIG.BERT_LARGE_PATH, "config.json"))
        config.label2id = CONFIG.LABEL2ID
        config.id2label = CONFIG.ID2LABEL
    torch.cuda.set_device(LOCAL_RANK)
    model = customBERT(config, params=params).to(LOCAL_RANK)
    # DDP
    model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, find_unused_parameters=True)
    trainer = Trainer(params, model)
    trainer.ddp_train()


if __name__ == "__main__":
    params = get_train_parser()
    ddp_seed_init(params.seed+LOCAL_RANK)
    main(params)
