import platform

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from transformers import BertConfig

import CONFIG
import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from models.layers import customBERT
from utils.common_util import load_checkpoint, Dataloader_train, seed_everything, get_parser


# TODO: REWRITE THE VALIDATION FUNCTION
class Evaluator:
    def __init__(
        self, model, testing_loader=None, device=None, params=None, model_path=None
    ):
        loader = Dataloader_train(params)
        self.model = model
        if params is None:
            self.device = device
            self.testing_loader = testing_loader
            load_checkpoint(model_path, self.model)
        else:
            self.device = torch.device(torch.device("cuda:{}".format(params.cuda)))
            _, self.testing_loader = loader.get_loader()
            load_checkpoint(params.model_path, self.model)

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


