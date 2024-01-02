import os
import pickle
import platform
import datetime

import numpy as np
from sklearn import metrics
from transformers import BertConfig

import CONFIG
import torch
from sklearn.metrics import classification_report

from models.layers import TF_BERT, TF_glove_fasttext
from utils.common_util import (
    load_checkpoint,
    get_eval_parser,
    seed_init,
)
from utils.data_util import loader_eval
from utils.logger import Logger

# set the environment variable for reproducibility
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


class Evaluator:
    def __init__(
        self, model, testing_loader=None, device=None, params=None, model_path=None
    ):
        self.model = model
        self.testing_loader = testing_loader
        self.device = device
        load_checkpoint(model_path, self.model)

    def kg_check_alter(
        self,
        data,
        vec,
        hrt_enable=False,
        hrt_embedding_enable=False,
    ):
        if hrt_enable and hrt_embedding_enable:
            hrt_score_list = data["hrt_score_list"].to(self.device, dtype=torch.float)
            hrt_embedding_list = data["hrt_embedding_list"].to(
                self.device, dtype=torch.float
            )
            output = self.model(
                vec,
                hrt_score_list=hrt_score_list,
                hrt_embedding_list=hrt_embedding_list,
            )
        elif hrt_enable:
            hrt_score_list = data["hrt_score_list"].to(self.device, dtype=torch.float)
            output = self.model(vec, hrt_score_list=hrt_score_list)
        elif hrt_embedding_enable:
            hrt_embedding_list = data["hrt_embedding_list"].to(
                self.device, dtype=torch.float
            )
            output = self.model(vec, hrt_embedding_list=hrt_embedding_list)
        else:
            output = self.model(vec)
        return output

    def kg_check(
        self,
        data,
        ids,
        mask,
        token_type_ids,
        hrt_enable=False,
        hrt_embedding_enable=False,
    ):
        if hrt_enable and hrt_embedding_enable:
            hrt_score_list = data["hrt_score_list"].to(self.device, dtype=torch.float)
            hrt_embedding_list = data["hrt_embedding_list"].to(
                self.device, dtype=torch.float
            )
            output = self.model(
                ids,
                mask,
                token_type_ids,
                hrt_score_list=hrt_score_list,
                hrt_embedding_list=hrt_embedding_list,
            )
        elif hrt_enable:
            hrt_score_list = data["hrt_score_list"].to(self.device, dtype=torch.float)
            output = self.model(
                ids, mask, token_type_ids, hrt_score_list=hrt_score_list
            )
        elif hrt_embedding_enable:
            hrt_embedding_list = data["hrt_embedding_list"].to(
                self.device, dtype=torch.float
            )
            output = self.model(
                ids, mask, token_type_ids, hrt_embedding_list=hrt_embedding_list
            )
        else:
            output = self.model(ids, mask, token_type_ids)
        return output

    # for TF-BERT
    def validation(
        self,
        logger=None,
        hrt_enable=False,
        hrt_embedding_enable=False,
    ):
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
                outputs = self.kg_check(
                    data,
                    ids,
                    mask,
                    token_type_ids,
                    hrt_enable=hrt_enable,
                    hrt_embedding_enable=hrt_embedding_enable,
                )
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

        logger.log(f"Accuracy Score = {accuracy}")
        logger.log(f"F1 Score (Micro) = {f1_score_micro}")
        logger.log(f"F1 Score (Macro) = {f1_score_macro}")
        logger.log(f"Recall Score = {recall}")
        formatted_report = "\n".join(
            [
                f"    {line}" if i > 0 else line
                for i, line in enumerate(report.split("\n"))
            ]
        )
        logger.log(f"Report:\n{formatted_report}")

    def validation_alter(
        self,
        logger=None,
        hrt_enable=False,
        hrt_embedding_enable=False,
    ):
        self.model.eval()
        fin_targets = []
        fin_outputs = []
        with torch.no_grad():
            for _, data in enumerate(self.testing_loader):
                targets = data["targets"].to(self.device, dtype=torch.float)
                vec = data["embedding"].to(self.device, dtype=torch.float)
                outputs = self.kg_check_alter(
                    data,
                    vec,
                    hrt_enable=hrt_enable,
                    hrt_embedding_enable=hrt_embedding_enable,
                )
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

        logger.log(f"Accuracy Score = {accuracy}")
        logger.log(f"F1 Score (Micro) = {f1_score_micro}")
        logger.log(f"F1 Score (Macro) = {f1_score_macro}")
        logger.log(f"Recall Score = {recall}")
        formatted_report = "\n".join(
            [
                f"    {line}" if i > 0 else line
                for i, line in enumerate(report.split("\n"))
            ]
        )
        logger.log(f"Report:\n{formatted_report}")


def eval(params, logger):
    train_args_path = os.path.join(os.path.dirname(params.model_path), "train_args.pkl")
    with open(train_args_path, "rb") as f:
        train_args = pickle.load(f)
    if params.bert:
        config = BertConfig(label2id=CONFIG.LABEL2ID, id2label=CONFIG.ID2LABEL)
        device = torch.device("cuda:{}".format(params.cuda))
        # use train_args as params to initialize the model
        model = TF_BERT(config, train_args).to(device)
    elif params.glove or params.fasttext:
        device = torch.device("cuda:{}".format(params.cuda))
        model = TF_glove_fasttext(train_args).to(device)
    loader = loader_eval(params, train_args)
    eval_loader = loader.get_loader()
    evaluator = Evaluator(
        model, testing_loader=eval_loader, device=device, params=params
    )
    if params.bert:
        evaluator.validation(
            logger,
            hrt_enable=train_args.hrt,
            hrt_embedding_enable=train_args.hrt_embedding,
        )
    else:
        evaluator.validation_alter(
            logger,
            hrt_enable=train_args.hrt,
            hrt_embedding_enable=train_args.hrt_embedding,
        )


if __name__ == "__main__":
    params = get_eval_parser()

    # get timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # get dic path
    dic_path = os.path.dirname(params.model_path)
    eval_path = os.path.join(dic_path, timestamp, "eval")
    os.makedirs(eval_path)
    # get logger
    logger = Logger("Eval", eval_path)

    if platform.system() == "Linux":
        seed_init(params.seed)
    eval(params, logger)
