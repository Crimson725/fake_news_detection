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
            eps=1e-4,
        )
        self.model_name = self.params.model_name

        self.train_set = None
        self.test_set = None

        # self.early_stop_patience = self.params.patience

        self.file_path = file_path
        self.tf_path = tf_path

        self.best_metrics_path = os.path.join(self.file_path, "best_metrics.pt")
        self.metrics_path = os.path.join(self.file_path, "metrics.pt")
        self.train_args_path = os.path.join(self.file_path, "train_args.pkl")
        self.model_path = os.path.join(self.file_path, "model.pt")

    def kg_check(self, model, data, ids, mask, token_type_ids):
        if self.params.hrt_embedding and self.params.hrt:
            hrt_embedding_list = data["hrt_embedding_list"]
            hrt_score_list = data["hrt_score_list"]
            output = model(
                ids,
                mask,
                token_type_ids,
                hrt_score_list=hrt_score_list,
                hrt_embedding_list=hrt_embedding_list,
            )
        elif self.params.hrt:
            hrt_score_list = data["hrt_score_list"]
            output = model(ids, mask, token_type_ids, hrt_score_list=hrt_score_list)
        elif self.params.hrt_embedding:
            hrt_embedding_list = data["hrt_embedding_list"]
            output = model(
                ids, mask, token_type_ids, hrt_embedding_list=hrt_embedding_list
            )

        else:
            output = model(ids, mask, token_type_ids)
        return output

    def kg_check_alter(
        self,
        model,
        data,
        vec,
    ):
        if self.params.hrt_embedding and self.params.hrt:
            hrt_score_list = data["hrt_score_list"].to(self.device, dtype=torch.float)
            hrt_embedding_list = data["hrt_embedding_list"].to(
                self.device, dtype=torch.float
            )
            output = model(
                vec,
                hrt_score_list=hrt_score_list,
                hrt_embedding_list=hrt_embedding_list,
            )
        elif self.params.hrt:
            hrt_score_list = data["hrt_score_list"].to(self.device, dtype=torch.float)
            output = model(vec, hrt_score_list=hrt_score_list)
        elif self.params.hrt_embedding:
            hrt_embedding_list = data["hrt_embedding_list"].to(
                self.device, dtype=torch.float
            )
            output = model(vec, hrt_embedding_list=hrt_embedding_list)
        else:
            output = model(vec)
        return output

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
        # early_stop_counter = 0

        # eval setting
        eval_every = self.params.eval_every

        # training
        model.train()
        start_time = time.time()
        if dist.get_rank() == 0:
            logger.log("Start training...")
            logger.log("Training size: {}".format(len(training_loader.dataset)))
            logger.log("Test size: {}".format(len(testing_loader.dataset)))

            logger.log("--------------------Args--------------------")

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
        if dist.get_rank() == 0:
            logger.log("--------------------Loss--------------------")
        for epoch in range(1, epochs + 1):
            # set epoch for shuffle
            training_sampler.set_epoch(epoch)
            testing_sampler.set_epoch(epoch)

            for _, data in enumerate(training_loader):
                targets = data["targets"].to(self.device, dtype=torch.float)
                if self.params.bert:
                    ids = data["ids"].to(self.device, dtype=torch.long)
                    mask = data["mask"].to(self.device, dtype=torch.long)
                    token_type_ids = data["token_type_ids"].to(
                        self.device, dtype=torch.long
                    )
                else:
                    vec = data["embedding"].to(self.device, dtype=torch.float)

                optimizer.zero_grad()
                if self.params.bert:
                    output = self.kg_check(model, data, ids, mask, token_type_ids)
                else:
                    output = self.kg_check_alter(model, data, vec)
                loss = loss_fn(output, targets)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                global_step += 1
                # if global_step % 100 == 0:
                #     print("global_step: ", global_step)
                #     print("total: ", epochs * len(training_loader))
                if global_step % eval_every == 0:
                    model.eval()
                    with torch.no_grad():
                        for _, data in enumerate(testing_loader):
                            targets = data["targets"].to(self.device, dtype=torch.float)
                            if self.params.bert:
                                ids = data["ids"].to(self.device, dtype=torch.long)
                                mask = data["mask"].to(self.device, dtype=torch.long)
                                token_type_ids = data["token_type_ids"].to(
                                    self.device, dtype=torch.long
                                )
                            else:
                                vec = data["embedding"].to(
                                    self.device, dtype=torch.float
                                )
                            if self.params.bert:
                                output = self.kg_check(
                                    model, data, ids, mask, token_type_ids
                                )
                            else:
                                output = self.kg_check_alter(model, data, vec)
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
                            epoch,
                            epochs,
                            global_step,
                            epochs * len(training_loader),
                            average_train_loss,
                            average_valid_loss,
                        )
                        if dist.get_rank() == 0:
                            print(msg)
                            logger.log(msg)

                        if best_valid_loss > average_valid_loss:
                            best_valid_loss = average_valid_loss
                            # early_stop_counter = 0
                            if dist.get_rank() == 0:
                                save_checkpoint(self.model_path, model, best_valid_loss)
                                save_metrics(
                                    self.best_metrics_path,
                                    train_loss_list,
                                    valid_loss_list,
                                    global_step_list,
                                )

        if dist.get_rank() == 0:
            # eval the model on the test set
            eval_model = self.model
            evaluator = Evaluator(
                eval_model,
                testing_loader,
                device=self.device,
                params=None,
                model_path=self.model_path,
            )
            logger.log("--------------------Evaluation--------------------")
            if self.params.bert:
                evaluator.validation(
                    logger,
                    hrt_enable=self.params.hrt,
                    hrt_embedding_enable=self.params.hrt_embedding,
                )
            else:
                evaluator.validation_alter(
                    logger,
                    hrt_enable=self.params.hrt,
                    hrt_embedding_enable=self.params.hrt_embedding,
                )
            save_metrics(
                self.metrics_path,
                train_loss_list,
                valid_loss_list,
                global_step_list,
            )
        dist.barrier()

        end_time = time.time()
        elapsed_time = end_time - start_time

        if dist.get_rank() == 0:
            logger.log(
                f"Model: {str(model.__class__.__name__)}, Best valid loss: {best_valid_loss}, Elapsed time: {elapsed_time}"
            )
        print("Finished Training!")
