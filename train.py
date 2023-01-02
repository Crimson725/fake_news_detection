import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from transformers import BertConfig
from models.layers import customBERT
from utils.Util import save_checkpoint, save_metrics, seed_everything, seed_worker
from utils.Util import CustomDataset, tokenizer
from utils.logger import Logger
import time
import datetime
import CONFIG
import os
from eval import validation
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
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.params.lr)
        self.model_name = self.params.model_name

        self.train_set = None
        self.test_set = None

    def get_loader(self):
        # dataset settings
        g = torch.Generator().manual_seed(42)
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
            train_params = {
                "batch_size": self.params.train_batch,
                "shuffle": True,
                "num_workers": 0,
                "worker_init_fn": seed_worker,
                "generator": g,
            }

            test_params = {
                "batch_size": self.params.test_batch,
                "shuffle": True,
                "num_workers": 0,
                "worker_init_fn": seed_worker,
                "generator": g,
            }
            training_loader = DataLoader(train_set, **train_params)
            testing_loader = DataLoader(test_set, **test_params)
            self.train_set, self.test_set = self.params.dataset
        else:
            train_dataset = pd.read_csv(
                os.path.join(CONFIG.DATA_PATH, self.params.dataset)
            )
            test_dataset = pd.read_csv(
                os.path.join(CONFIG.DATA_PATH, self.params.valid_dataset)
            )
            train_set = CustomDataset(train_dataset, tokenizer, self.params.max_len)
            test_set = CustomDataset(test_dataset, tokenizer, self.params.max_len)
            train_params = {
                "batch_size": self.params.train_batch,
                "shuffle": True,
                "num_workers": 0,
                "worker_init_fn": seed_worker,
                "generator": g,
            }

            test_params = {
                "batch_size": self.params.test_batch,
                "shuffle": True,
                "num_workers": 0,
                "worker_init_fn": seed_worker,
                "generator": g,
            }

            training_loader = DataLoader(train_set, **train_params)
            testing_loader = DataLoader(test_set, **test_params)
            self.train_set = self.params.dataset
            self.test_set = self.params.valid_dataset
        return training_loader, testing_loader

    def get_path(self, name):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_path = os.path.join(CONFIG.DESTINATION_PATH, timestamp + "_" + name)
        tf_path = os.path.join(file_path, "tf_logs")
        os.mkdir(file_path)
        os.mkdir(tf_path)
        # get the tflogger
        return file_path, tf_path

    def train_customBERT(
            self,
            best_valid_loss=float("Inf"),
            validate=True,
    ):
        seed_everything(42)

        # get the datasloader
        training_loader, testing_loader = self.get_loader()

        # get the model
        model = self.model
        model_name = self.model_name
        # make the path to save the log and models
        file_path, tf_path = self.get_path(model_name)
        # train/valid loss
        running_loss = 0
        valid_running_loss = 0
        global_step = 0
        train_loss_list = []
        valid_loss_list = []
        global_step_list = []

        # eval setting
        eval_every = len(training_loader) // 2
        # get the logger
        logger = Logger("Training", file_path)
        tflogger = SummaryWriter(tf_path)

        # training
        model.train()
        start_time = time.time()
        logger.log("Start training...")
        # if self.params.valid_enable:
        #     logger.log("Cross Validation enabled")
        #     logger.log(f"Train set: {self.train_set}, Valid set: {self.test_set}")
        # else:
        #     logger.log("Cross Validation disabled")
        #     logger.log(f"Train set: {self.train_set}, Valid set: {self.test_set}")

        # print and log args
        if self.params.log_args:
            for k, v in vars(self.params).items():
                logger.verbose(f"{k} = {v}")
                print(f'{k} = {v}')
        #  training args
        epochs = self.epochs
        loss_fn = self.loss_fn
        optimizer = self.optimizer
        verbose = self.params.verbose
        print_logs = self.params.print_logs
        for epoch in range(epochs):
            for _, data in enumerate(training_loader):
                ids = data["ids"].to(self.device, dtype=torch.long)
                mask = data["mask"].to(self.device, dtype=torch.long)
                token_type_ids = data["token_type_ids"].to(
                    self.device, dtype=torch.long
                )
                targets = data["targets"].to(self.device, dtype=torch.float)

                output = model(ids, mask, token_type_ids)
                loss = loss_fn(output, targets)
                optimizer.zero_grad()
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
                            save_checkpoint(
                                file_path + "/" + "model.pt", model, best_valid_loss
                            )
                            save_metrics(
                                file_path + "/" + "metrics.pt",
                                train_loss_list,
                                valid_loss_list,
                                global_step_list,
                            )
                            model.config.to_json_file(file_path + "/" + "config.json")
        if validate:
            validation(logger, testing_loader, model, self.device)

        save_metrics(
            file_path + "/" + "metrics.pt",
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

# define the model
# config = BertConfig(hidden_size=768, label2id=CONFIG.LABEL2ID, id2label=CONFIG.ID2LABEL)
# model = BERT(config).to(device)
# define the optimizer
# optimizer = optim.Adam(model.parameters(), lr=1e-6)

# define the datasets for original bert
# dataset = Dataset(CONFIG.DATA_PATH)
# train_loader = dataset.train_iter
# valid_loader = dataset.valid_iter
# test_loader = dataset.train_iter


# train original BERT
# seed_everything(42)
# train(model, optimizer=optimizer, train_loader=train_loader, valid_loader=valid_loader,
#       num_epochs=EPOCHS, eval_every=len(train_loader) // 2, file_path=file_path)


# train custom BERT


# train_customBERT(
#     model,
#     loss_fn=torch.nn.BCELoss(),
#     optimizer=optimizer,
#     train_loader=training_loader,
#     valid_loader=testing_loader,
#     num_epochs=EPOCHS,
#     eval_every=len(training_loader) // 2,
#     file_path=file_path,
#     validate=True,
#     tflogger=tflogger,
# )
