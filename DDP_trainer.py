from transformers import BertConfig
import pykeen
import CONFIG
from models.layers import customBERT
from train import Trainer
import os
from utils.common_util import DDP_seed_init, get_DDP_path, get_train_parser_DDP
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
device = torch.device("cuda", LOCAL_RANK)


def main(params):
    # init the model
    config = BertConfig(label2id=CONFIG.LABEL2ID, id2label=CONFIG.ID2LABEL)
    torch.cuda.set_device(LOCAL_RANK)
    model = customBERT(config, params=params).to(LOCAL_RANK)

    modelname = params.model_name

    file_path, tf_path = get_DDP_path(modelname)
    # save config
    model.config.to_json_file(file_path + "/" + "config.json")

    # DDP model
    model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)
    trainer = Trainer(
        params, model, file_path=file_path, tf_path=tf_path, device=device
    )
    trainer.DDP_train()


if __name__ == "__main__":
    params = get_train_parser_DDP()
    DDP_seed_init(params.seed + LOCAL_RANK)
    main(params)
