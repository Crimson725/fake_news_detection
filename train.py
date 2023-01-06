import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from transformers import BertConfig
from models.layers import customBERT
from utils.Util import save_checkpoint, save_metrics, seed_everything, load_checkpoint
from utils.Util import CustomDataset, tokenizer, customDataloader
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
        self.config = BertConfig(label2id=CONFIG.LABEL2ID, id2label=CONFIG.ID2LABEL)
        self.model = customBERT(self.config, params=self.params).to(self.device)
        self.loss_fn = nn.BCELoss()
        self.epochs = self.params.epochs
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.params.lr,weight_decay=self.params.weight_decay)
        self.model_name = self.params.model_name

        self.train_set = None
        self.test_set = None

    def get_loader(self):
        def seed_worker(worker_id):
            worker_seed = self.params.seed
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(self.params.seed)

        train_loader_params = {
            "batch_size": self.params.train_batch,
            "shuffle": True,
            "worker_init_fn": seed_worker,
            "generator": g,
        }
        test_loader_params = {
            "batch_size": self.params.test_batch,
            "shuffle": True,
            "worker_init_fn": seed_worker,
            "generator": g,
        }
        # dataset settings
        if self.params.valid_enable is False:
            train_size = self.params.frac
            df = pd.read_csv(os.path.join(CONFIG.DATA_PATH, self.params.dataset))
            train_dataset = df.sample(frac=train_size, random_state=200).reset_index(
                drop=True
            )
            test_dataset = df.drop(train_dataset.index).reset_index(drop=True)

            # get the train set and test set
            train_set = CustomDataset(train_dataset, tokenizer, self.params.max_len)
            test_set = CustomDataset(test_dataset, tokenizer, self.params.max_len)
            training_loader = DataLoader(train_set, **train_loader_params)
            testing_loader = DataLoader(test_set, **test_loader_params)
        else:
            train_dataset = pd.read_csv(
                os.path.join(CONFIG.DATA_PATH, self.params.dataset)
            )
            test_dataset = pd.read_csv(
                os.path.join(CONFIG.DATA_PATH, self.params.valid_dataset)
            )
            train_set = CustomDataset(train_dataset, tokenizer, self.params.max_len)
            test_set = CustomDataset(test_dataset, tokenizer, self.params.max_len)
            training_loader = DataLoader(train_set, **train_loader_params)
            testing_loader = DataLoader(test_set, **test_loader_params)
        return training_loader, testing_loader

    def get_path(self, name):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_path = os.path.join(CONFIG.DESTINATION_PATH, timestamp + "_" + name)
        tf_path = os.path.join(file_path, "tf_logs")
        os.mkdir(file_path)
        os.mkdir(tf_path)
        return file_path, tf_path

    def train_customBERT(
        self,
        best_valid_loss=float("Inf"),
    ):
        loader = customDataloader(self.params)
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
        # train/valid loss
        running_loss = 0
        valid_running_loss = 0
        global_step = 0
        train_loss_list = []
        valid_loss_list = []
        global_step_list = []

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
                if self.params.l2 is not None:
                    l2_loss = torch.sum(model.l3.weight**3) * self.params.l2
                    loss = loss_fn(output, targets) + l2_loss
                    loss.backward()
                else:
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
                            save_checkpoint(model_path, model, best_valid_loss)
                            save_metrics(
                                best_metrics_path,
                                train_loss_list,
                                valid_loss_list,
                                global_step_list,
                            )
                            model.config.to_json_file(file_path + "/" + "config.json")
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
