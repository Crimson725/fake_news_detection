import os
import pickle
import torch
import torch.nn as nn
from torch import optim
import torch.distributed as dist

from utils.common_util import (
    save_checkpoint,
    save_metrics,
)
from utils.data_util import loader_train, DDP_loader_train
from utils.logger import Logger
import time

from eval import Evaluator

from tensorboardX import SummaryWriter


class Trainer:
    def __init__(self, params, model, file_path=None, tf_path=None, device=None):

        self.params = params
        self.device = device
        self.model = model

        self.loss_fn = nn.BCELoss()
        self.epochs = self.params.epochs
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.params.lr,
            weight_decay=self.params.weight_decay,
        )
        self.model_name = self.params.model_name

        self.train_set = None
        self.test_set = None

        self.early_stop_patience = self.params.early_stop_patience

        self.file_path = file_path
        self.tf_path = tf_path

        self.best_metrics_path = os.path.join(self.file_path, "best_metrics.pt")
        self.metrics_path = os.path.join(self.file_path, "metrics.pt")
        self.train_args_path = os.path.join(self.file_path, "train_args.pkl")
        self.model_path = os.path.join(self.file_path, "model.pt")

    def train(self, best_valid_loss=float("Inf")):

        loader = loader_train(self.params)
        # get the dataloader
        training_loader, testing_loader = loader.get_loader()

        # get the model
        model = self.model

        # save the args
        with open(self.train_args_path, "wb") as f:
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
        logger = Logger("Training", self.file_path)
        tflogger = SummaryWriter(self.tf_path)

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
                                "Training_loss", average_train_loss, global_step
                            )
                            tflogger.add_scalar(
                                "Validation_loss", average_valid_loss, global_step
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
                            save_checkpoint(self.model_path, model, best_valid_loss)
                            save_metrics(
                                self.best_metrics_path,
                                train_loss_list,
                                valid_loss_list,
                                global_step_list,
                            )
                            model.config.to_json_file(
                                self.file_path + "/" + "config.json"
                            )

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
                model_path=self.model_path,
            )
            logger.log("--------------------Evaluation--------------------")
            evaluator.validation(logger)

        save_metrics(
            self.metrics_path,
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

    def DDP_train(self, best_valid_loss=float("Inf")):

        loader = DDP_loader_train(self.params)
        # get the dataloader
        (
            training_loader,
            testing_loader,
            training_sampler,
            testing_sampler,
        ) = loader.get_loader()

        # get the model
        model = self.model

        if dist.get_rank() == 0:
            # save the args
            with open(self.train_args_path, "wb") as f:
                pickle.dump(self.params, f)
            # get the logger
            logger = Logger("Training", self.file_path)
            tflogger = SummaryWriter(self.tf_path)
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

        # training
        model.train()
        start_time = time.time()
        if dist.get_rank() == 0:
            logger.log("Start training...")
            logger.log("Training size: {}".format(len(training_loader.dataset)))
            logger.log("Test size: {}".format(len(testing_loader.dataset)))

            logger.log("--------------------Args--------------------")

        if self.params.log_args:
            if dist.get_rank() == 0:
                # print and log args
                print("--------------------Args--------------------")
                for k, v in vars(self.params).items():
                    logger.log(f"{k} = {v}")
                    print(f"{k} = {v}")
        #  training args
        epochs = self.epochs
        loss_fn = self.loss_fn
        optimizer = self.optimizer
        verbose = self.params.verbose
        print_logs = self.params.print_logs
        if dist.get_rank() == 0:
            logger.log("--------------------Loss--------------------")
        for epoch in range(epochs):
            # set epoch for shuffle
            training_sampler.set_epoch(epoch)

            for _, data in enumerate(training_loader):
                ids = data["ids"].to(self.device, dtype=torch.long)
                mask = data["mask"].to(self.device, dtype=torch.long)
                token_type_ids = data["token_type_ids"].to(
                    self.device, dtype=torch.long
                )
                targets = data["targets"].to(self.device, dtype=torch.float)

                optimizer.zero_grad()

                # entity enable
                if self.params.entity:
                    entity_embedding = data["entity_embedding"].to(
                        self.device, dtype=torch.float
                    )
                    output = model(ids, mask, token_type_ids, entity_embedding)
                else:
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
                            # entity enable
                            if self.params.entity:
                                entity_embedding = data["entity_embedding"].to(
                                    self.device, dtype=torch.float
                                )
                                output = model(
                                    ids, mask, token_type_ids, entity_embedding
                                )
                            else:
                                output = model(ids, mask, token_type_ids)

                            loss = loss_fn(output, targets)
                            valid_running_loss += loss.item()

                        average_train_loss = running_loss / eval_every
                        average_valid_loss = valid_running_loss / len(testing_loader)
                        train_loss_list.append(average_train_loss)
                        valid_loss_list.append(average_valid_loss)
                        global_step_list.append(global_step)
                        if dist.get_rank() == 0:
                            if tflogger is not None:
                                tflogger.add_scalar(
                                    "Training_loss", average_train_loss, global_step
                                )
                                tflogger.add_scalar(
                                    "Validation_loss", average_valid_loss, global_step
                                )
                                for name, param in model.named_parameters():
                                    tflogger.add_histogram(
                                        name,
                                        param.clone().cpu().data.numpy(),
                                        global_step,
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
                            if dist.get_rank() == 0:
                                if print_logs:
                                    # print the training msg
                                    print(msg)
                                logger.log(msg)
                        if best_valid_loss > average_valid_loss:
                            best_valid_loss = average_valid_loss
                            early_stop_counter = 0
                            if dist.get_rank() == 0:
                                save_checkpoint(self.model_path, model, best_valid_loss)
                                save_metrics(
                                    self.best_metrics_path,
                                    train_loss_list,
                                    valid_loss_list,
                                    global_step_list,
                                )
                        else:
                            early_stop_counter += 1
                        if early_stop_counter >= self.early_stop_patience:
                            print("Early stopping")
                            if dist.get_rank() == 0:
                                logger.log(
                                    f"Early stopping in step:{global_step}/{epochs * len(training_loader)}"
                                )
                            break
        if dist.get_rank() == 0:
            # validation
            if self.params.valid_enable:
                eval_model = self.model
                evaluator = Evaluator(
                    eval_model,
                    testing_loader,
                    device=self.device,
                    params=None,
                    model_path=self.model_path,
                )
                if dist.get_rank() == 0:
                    logger.log("--------------------Evaluation--------------------")

                evaluator.validation(logger, entity_enable=self.params.entity)

            save_metrics(
                self.metrics_path,
                train_loss_list,
                valid_loss_list,
                global_step_list,
            )

        end_time = time.time()
        elapsed_time = end_time - start_time

        if dist.get_rank() == 0:
            logger.log(
                f"Model: {str(model.__class__.__name__)}, Best valid loss: {best_valid_loss}, Elapsed time: {elapsed_time}"
            )
        print("Finished Training!")
