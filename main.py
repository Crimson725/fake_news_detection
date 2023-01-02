import argparse
from train import Trainer


def parse_arguments():
    parser = argparse.ArgumentParser(description="Arg parser for fake news detection")
    parser.add_argument("--seed", type=int, default=42, help="seed")
    parser.add_argument("--cuda", type=int, default=0, help="device id")
    parser.add_argument(
        "--frac", type=float, default=0.75, help="ration of train set & test set (only for valid_enable=False)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="real_and_fake/train.csv",
        help="dataset name",
    )
    parser.add_argument(
        "--valid_enable",
        type=bool,
        default=True,
        help="enable cross domain validation",
    )
    parser.add_argument(
        "--valid_dataset",
        type=str,
        default="real_and_fake/test.csv",
        help="cross domain validate set",
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument(
        "--train_batch", type=int, default=8, help="training set batch size"
    )
    parser.add_argument(
        "--test_batch", type=int, default=8, help="validation set batch size"
    )
    parser.add_argument(
        "--max_len", type=int, default=512, help="max length to padding"
    )
    parser.add_argument("--epochs", type=int, default=1, help="epoch of training ")
    parser.add_argument(
        "--dropout", type=float, default=0.5, help="dropout rate of the model"
    )
    parser.add_argument(
        "--model_name", type=str, default="customBERT", help="name of the model"
    )

    parser.add_argument("--log_args", type=bool, default=True, help="log the args of the training process")
    parser.add_argument(
        "--verbose",
        type=bool,
        default=True,
        help="log verbose (loss) info of the training process",
    )
    parser.add_argument(
        "--print_logs",
        type=bool,
        default=False,
        help="print the verbose info of the training process",
    )
    args = parser.parse_args()
    return args


def main(params):
    trainer = Trainer(params)
    trainer.train_customBERT()


if __name__ == "__main__":
    params = parse_arguments()
    main(params)
