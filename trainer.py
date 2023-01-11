from train import Trainer
import os
from utils.common_util import seed_everything
from utils.common_util import get_train_parser
import platform

# set the environment variable
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def train(params):
    trainer = Trainer(params)
    trainer.train_customBERT()


if __name__ == "__main__":
    params = get_train_parser()
    if platform.system() == "Linux":
        seed_everything(params.seed)
    train(params)
