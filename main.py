import argparse
from train import Trainer
import os
from utils.Util import seed_everything
from utils.Util import get_parser
import platform

# set the environment variable
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def main(params):
    trainer = Trainer(params)
    trainer.train_customBERT()


if __name__ == "__main__":
    params = get_parser()
    if platform.system() == "Linux":
        seed_everything(params.seed)
    main(params)
