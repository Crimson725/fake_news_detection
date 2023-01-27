import os
import pickle
import platform

import numpy as np
from sklearn import metrics
from transformers import BertConfig

import CONFIG
import torch
from sklearn.metrics import classification_report

from models.layers import customBERT
from utils.common_util import (
    load_checkpoint,
    loader_eval,
    get_eval_parser,
    seed_init,
)

# set the environment variable
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


class Evaluator:
    def __init__(
            self, model, testing_loader=None, device=None, params=None, model_path=None
    ):
        self.model = model
        self.testing_loader = testing_loader
        self.device = device
        if params is None:
            load_checkpoint(model_path, self.model)
        else:
            load_checkpoint(params.model_path, self.model, params.mode)

    # for the original BERT
    # def evaluate(self, model, test_loader):
    #     y_pred = []
    #     y_true = []
    #     model.eval()
    #     with torch.no_grad():
    #         for _, data in enumerate(test_loader):
    #             ids = data["ids"].to(device, dtype=torch.long)
    #             mask = data["mask"].to(device, dtype=torch.long)
    #             token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
    #             targets = data["targets"].to(device, dtype=torch.float)
    #             outputs = model(ids, mask, token_type_ids)
    #             y_pred.extend(torch.argmax(outputs, 1).tolist())
    #             y_true.extend(targets.tolist())
    #     print("Classification Report: ")
    #     print(classification_report(y_true, y_pred, target_names=CONFIG.ID2LABEL.values()))
    #     cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    #     ax = plt.subplot()
    #     fig = sns.heatmap(cm, annot=True, ax=ax, fmt="d", cmap="Blues")
    #     fig.get_figure().savefig(CONFIG.PLOT_PATH + "/" + "confusion_matrix.png")
    #
    #     ax.set_title("Confusion Matrix")
    #     ax.set_xlabel("Predicted Labels")
    #     ax.set_ylabel("True Labels")
    #
    #     ax.xaxis.set_ticklabels(CONFIG.ID2LABEL.values())
    #     ax.yaxis.set_ticklabels(CONFIG.ID2LABEL.values())

    # for new
    def validation(self, logger=None):
        self.model.eval()
        fin_targets = []
        fin_outputs = []
        with torch.no_grad():
            for _, data in enumerate(self.testing_loader):
                ids = data["ids"].to(self.device, dtype=torch.long)
                mask = data["mask"].to(self.device, dtype=torch.long)
                token_type_ids = data["token_type_ids"].to(
                    self.device, dtype=torch.long
                )
                targets = data["targets"].to(self.device, dtype=torch.float)
                outputs = self.model(ids, mask, token_type_ids)
                fin_targets.extend(targets.cpu().detach().numpy().tolist())
                fin_outputs.extend(outputs.cpu().detach().numpy().tolist())
        outputs = np.array(fin_outputs) >= 0.5
        accuracy = metrics.accuracy_score(fin_targets, outputs)
        f1_score_micro = metrics.f1_score(fin_targets, outputs, average="micro")
        f1_score_macro = metrics.f1_score(fin_targets, outputs, average="macro")
        recall = metrics.recall_score(fin_targets, outputs)
        report = classification_report(
            fin_targets, outputs, target_names=CONFIG.ID2LABEL.values()
        )
        print(f"Accuracy Score = {accuracy}")
        print(f"F1 Score (Micro) = {f1_score_micro}")
        print(f"F1 Score (Macro) = {f1_score_macro}")
        print(f"Recall Score = {recall}")
        print(f"Report: {report}")
        if logger is not None:
            logger.log(
                f"Accuracy Score = {accuracy}, F1 Score (Micro) = {f1_score_micro}, F1 Score (Macro) = {f1_score_macro}, Recall Score = {recall}, Report: {report}"
            )


def eval(params):
    train_args_path = os.path.join(os.path.dirname(params.model_path), "train_args.pkl")
    with open(train_args_path, "rb") as f:
        train_args = pickle.load(f)
    config = BertConfig(label2id=CONFIG.LABEL2ID, id2label=CONFIG.ID2LABEL)
    device = torch.device(torch.device("cuda:{}".format(params.cuda)))
    model = customBERT(config, train_args).to(device)
    loader = loader_eval(params, train_args)
    eval_loader = loader.get_loader()
    evaluator = Evaluator(
        model, testing_loader=eval_loader, device=device, params=params
    )
    evaluator.validation()


if __name__ == "__main__":
    params = get_eval_parser()
    if platform.system() == "Linux":
        seed_init(params.seed)
    eval(params)
