import pickle
import torch
import torch.nn as nn
from torch import optim
from transformers import BertConfig
from models.layers import customBERT
from utils.common_util import (
    save_checkpoint,
    save_metrics,
    Docloader_train,
)
from utils.logger import Logger
import time
import datetime
import CONFIG
import os
from eval import Evaluator
from tensorboardX import SummaryWriter


class Trainer:
    def __init__(self, params):

        self.params = params

        self.device = torch.device(torch.device("cuda:{}".format(self.params.cuda)))
        # define the customBERT
        # load the default config
        if self.params.bert_type== "bert-base-uncased":
            self.config = BertConfig(label2id=CONFIG.LABEL2ID, id2label=CONFIG.ID2LABEL)
        if self.params.bert_type=="bert-large-uncased":
            self.config = BertConfig.from_json_file(os.path.join(CONFIG.BERT_LARGE_PATH, "config.json"))
            self.config.label2id = CONFIG.LABEL2ID
            self.config.id2label = CONFIG.ID2LABEL
        self.model = customBERT(self.config, params=self.params).to(self.device)
        self.loss_fn = nn.BCELoss()
        self.epochs = self.params.epochs
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.params.lr,
            weight_decay=self.params.weight_decay,
        )
        self.model_name = self.params.model_name+self.params.bert_type

        self.train_set = None
        self.test_set = None

        self.early_stop_patience = self.params.early_stop_patience

    def get_path(self, name):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_path = os.path.join(CONFIG.DESTINATION_PATH, timestamp + "_" + name)
        tf_path = os.path.join(file_path, "tf_logs")
        os.mkdir(file_path)
        os.mkdir(tf_path)
        return file_path, tf_path

    def train_doc(self, best_valid_loss=float("Inf")):

        loader = Docloader_train(self.params)
        # get the dataloader
        training_loader, testing_loader = loader.get_loader()

        # get the model
        model = self.model
        model_name = self.model_name
        # make the path to save the log and models
        file_path, tf_path = self.get_path(model_name)
        model_path = file_path + "/" + "model.pt"
        best_metrics_path = file_path + "/" + "best_metrics.pt"
        metrics_path = file_path + "/" + "metrics.pt"
        train_args_path = file_path + "/" + "train_args.pkl"
        # save the args
        with open(train_args_path, "wb") as f:
            pickle.dump(self.params, f)
        # train/valid loss
        running_loss = 0
        valid_running_loss = 0
        global_step = 0
        train_loss_list = []
        valid_loss_list = []
        global_step_list = []
        # early stop counter
        early_stop_counter = 0

        # eval setting
        eval_every = len(training_loader) // self.params.eval_every
        # get the logger
        logger = Logger("Training", file_path)
        tflogger = SummaryWriter(tf_path)

        # training
        model.train()
        start_time = time.time()
        logger.log("Start training...")
        logger.log("Training size: {}".format(len(training_loader.dataset)))
        logger.log("Test size: {}".format(len(testing_loader.dataset)))

        logger.log("--------------------Args--------------------")
        # print and log args
        print("--------------------Args--------------------")
        if self.params.log_args:
            for k, v in vars(self.params).items():
                logger.log(f"{k} = {v}")
                print(f"{k} = {v}")
        #  training args
        epochs = self.epochs
        loss_fn = self.loss_fn
        optimizer = self.optimizer
        verbose = self.params.verbose
        print_logs = self.params.print_logs
        logger.log("--------------------Loss--------------------")
        for epoch in range(epochs):
            for _, data in enumerate(training_loader):
                ids = data["ids"].to(self.device, dtype=torch.long)
                mask = data["mask"].to(self.device, dtype=torch.long)
                token_type_ids = data["token_type_ids"].to(
                    self.device, dtype=torch.long
                )
                targets = data["targets"].to(self.device, dtype=torch.float)

                optimizer.zero_grad()
                output = model(ids, mask, token_type_ids)

                loss = loss_fn(output, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                global_step += 1
                if global_step % eval_every == 0:
                    model.eval()
                    with torch.no_grad():
                        for _, data in enumerate(testing_loader):
                            ids = data["ids"].to(self.device, dtype=torch.long)
                            mask = data["mask"].to(self.device, dtype=torch.long)
                            token_type_ids = data["token_type_ids"].to(
                                self.device, dtype=torch.long
                            )
                            targets = data["targets"].to(self.device, dtype=torch.float)

                            output = model(ids, mask, token_type_ids)
                            loss = loss_fn(output, targets)
                            valid_running_loss += loss.item()

                        average_train_loss = running_loss / eval_every
                        average_valid_loss = valid_running_loss / len(testing_loader)
                        train_loss_list.append(average_train_loss)
                        valid_loss_list.append(average_valid_loss)
                        global_step_list.append(global_step)
                        if tflogger is not None:
                            tflogger.add_scalar(
                                "Training loss", average_train_loss, global_step
                            )
                            tflogger.add_scalar(
                                "Validation loss", average_valid_loss, global_step
                            )
                            for name, param in model.named_parameters():
                                tflogger.add_histogram(
                                    name, param.clone().cpu().data.numpy(), global_step
                                )

                        running_loss = 0
                        valid_running_loss = 0
                        model.train()
                        msg = "Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}".format(
                            epoch + 1,
                            epochs,
                            global_step,
                            epochs * len(training_loader),
                            average_train_loss,
                            average_valid_loss,
                        )
                        if verbose:
                            if print_logs:
                                # print the training msg
                                print(msg)
                            logger.log(msg)
                        if best_valid_loss > average_valid_loss:
                            best_valid_loss = average_valid_loss
                            early_stop_counter = 0
                            save_checkpoint(model_path, model, best_valid_loss)
                            save_metrics(
                                best_metrics_path,
                                train_loss_list,
                                valid_loss_list,
                                global_step_list,
                            )
                            model.config.to_json_file(file_path + "/" + "config.json")
                        else:
                            early_stop_counter += 1
                        if early_stop_counter >= self.early_stop_patience:
                            print("Early stopping")
                            logger.log(
                                f"Early stopping in step:{global_step}/{epochs * len(training_loader)}"
                            )
                            break

        if self.params.valid_enable:
            eval_model = self.model
            evaluator = Evaluator(
                eval_model,
                testing_loader,
                device=self.device,
                params=None,
                model_path=model_path,
            )
            logger.log("--------------------Evaluation--------------------")
            evaluator.validation(logger)

        save_metrics(
            metrics_path,
            train_loss_list,
            valid_loss_list,
            global_step_list,
        )
        end_time = time.time()
        elapsed_time = end_time - start_time

        logger.log(
            f"Model: {str(model.__class__.__name__)}, Best valid loss: {best_valid_loss}, Elapsed time: {elapsed_time}"
        )
        print("Finished Training!")

