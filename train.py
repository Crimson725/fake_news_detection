import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader
from transformers import BertConfig
from models.layers import customBERT
from utils.helper_function import save_checkpoint, save_metrics, seed_everything, seed_worker
from utils.helper_function import CustomDataset, tokenizer
from utils.logger import Logger
import time
import datetime
import CONFIG
import os
from eval import validation
from tensorboardX import SummaryWriter
import warnings

# Suppress the warning about unused weights
warnings.filterwarnings("ignore", category=UserWarning, module='transformers')


EPOCHS = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model,
          optimizer,
          train_loader,
          valid_loader,
          num_epochs,
          eval_every,
          file_path,
          best_valid_loss=float('Inf')):
    running_loss = 0
    valid_running_loss = 0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_step_list = []

    logger = Logger('Training', file_path)

    model.train()
    start_time = time.time()
    logger.log("Start training...")
    for epoch in range(num_epochs):
        for (labels, title, text, titletext), _ in train_loader:
            labels = labels.type(torch.LongTensor).to(device)
            # use the title and text for training
            # can be changed for different datasets
            titletext = titletext.type(torch.LongTensor).to(device)
            output = model(titletext, labels)
            loss, _ = output

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            global_step += 1
            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():
                    for (labels, title, text, titletext), _ in valid_loader:
                        labels = labels.type(torch.LongTensor).to(device)
                        titletext = titletext.type(torch.LongTensor).to(device)
                        output = model(titletext, labels)
                        loss, _ = output
                        valid_running_loss += loss.item()

                    average_train_loss = running_loss / eval_every
                    average_valid_loss = valid_running_loss / len(valid_loader)
                    train_loss_list.append(average_train_loss)
                    valid_loss_list.append(average_valid_loss)
                    global_step_list.append(global_step)

                    running_loss = 0
                    valid_running_loss = 0
                    model.train()
                    print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'.format(epoch + 1,
                                                                                                       num_epochs,
                                                                                                       global_step,
                                                                                                       num_epochs * len(
                                                                                                           train_loader),
                                                                                                       average_train_loss,
                                                                                                       average_valid_loss))
                    if best_valid_loss > average_valid_loss:
                        best_valid_loss = average_valid_loss
                        save_checkpoint(file_path + '/' + 'model.pt', model, best_valid_loss)
                        save_metrics(file_path + '/' + 'metrics.pt', train_loss_list, valid_loss_list, global_step_list)
                        model.config.to_json_file(file_path + '/' + 'config.json')
    save_metrics(file_path + '/' + 'metrics.pt', train_loss_list, valid_loss_list, global_step_list)
    end_time = time.time()
    elapsed_time = end_time - start_time

    logger.log(
        f'Model: {str(model.__class__.__name__)}, Best valid loss: {best_valid_loss}, Elapsed time: {elapsed_time}')
    print('Finished Training!')


def train_customBERT(model,
                     loss_fn,
                     optimizer,
                     train_loader,
                     valid_loader,
                     num_epochs,
                     eval_every,
                     file_path,
                     best_valid_loss=float('Inf'),
                     validate=True,
                     tflogger=None):
    running_loss = 0
    valid_running_loss = 0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_step_list = []
    logger = Logger('Training', file_path)

    model.train()

    start_time = time.time()
    logger.log("Start training...")

    for epoch in range(num_epochs):
        for idx, data in enumerate(train_loader):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)
            # print('ids: {}, mask: {}, token_type_ids: {}'.format(ids.shape, mask.shape, token_type_ids.shape))
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
                    for idx, data in enumerate(train_loader):
                        ids = data['ids'].to(device, dtype=torch.long)
                        mask = data['mask'].to(device, dtype=torch.long)
                        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                        targets = data['targets'].to(device, dtype=torch.float)

                        output = model(ids, mask, token_type_ids)
                        loss = loss_fn(output, targets)
                        valid_running_loss += loss.item()

                    average_train_loss = running_loss / eval_every
                    average_valid_loss = valid_running_loss / len(valid_loader)
                    train_loss_list.append(average_train_loss)
                    valid_loss_list.append(average_valid_loss)
                    global_step_list.append(global_step)
                    if tflogger is not None:
                        tflogger.add_scalar('Training loss', average_train_loss, global_step)
                        tflogger.add_scalar('Validation loss', average_valid_loss, global_step)
                        for name, param in model.named_parameters():
                            tflogger.add_histogram(name, param.clone().cpu().data.numpy(), global_step)

                    running_loss = 0
                    valid_running_loss = 0
                    model.train()
                    print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'.format(epoch + 1,
                                                                                                       num_epochs,
                                                                                                       global_step,
                                                                                                       num_epochs * len(
                                                                                                           train_loader),
                                                                                                       average_train_loss,
                                                                                                       average_valid_loss))
                    if best_valid_loss > average_valid_loss:
                        best_valid_loss = average_valid_loss
                        save_checkpoint(file_path + '/' + 'model.pt', model, best_valid_loss)
                        save_metrics(file_path + '/' + 'metrics.pt', train_loss_list, valid_loss_list, global_step_list)
                        model.config.to_json_file(file_path + '/' + 'config.json')
    if validate:
        validation(logger, valid_loader, model)

    save_metrics(file_path + '/' + 'metrics.pt', train_loss_list, valid_loss_list, global_step_list)
    end_time = time.time()
    elapsed_time = end_time - start_time

    logger.log(
        f'Model: {str(model.__class__.__name__)}, Best valid loss: {best_valid_loss}, Elapsed time: {elapsed_time}')
    print('Finished Training!')


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

# define the customBERT
config = BertConfig(label2id=CONFIG.LABEL2ID, id2label=CONFIG.ID2LABEL)
model = customBERT(config).to(device)
optimizer = optim.Adam(model.parameters(), lr=CONFIG.LEARNING_RATE)

# define the dataset and dataloader
train_size = 0.75
g = torch.Generator().manual_seed(42)
df = pd.read_csv(CONFIG.DATA_PATH + '/train.csv')
train_dataset = df.sample(frac=train_size, random_state=200).reset_index(drop=True)
test_dataset = df.drop(train_dataset.index).reset_index(drop=True)

train_set = CustomDataset(train_dataset, tokenizer, CONFIG.MAX_LEN)
test_set = CustomDataset(test_dataset, tokenizer, CONFIG.MAX_LEN)
train_params = {'batch_size': CONFIG.TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0,
                'worker_init_fn': seed_worker,
                'generator': g
                }

test_params = {'batch_size': CONFIG.VALID_BATCH_SIZE,
               'shuffle': True,
               'num_workers': 0,
               'worker_init_fn': seed_worker,
               'generator': g
               }

training_loader = DataLoader(train_set, **train_params)
testing_loader = DataLoader(test_set, **test_params)

# make the path to save the log and models
timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
modelname = str(model.__class__.__name__)
file_path = os.path.join(CONFIG.DESTINATION_PATH, timestamp + '_' + modelname)
tf_path = os.path.join(file_path, 'tf_logs')
os.mkdir(file_path)
os.mkdir(tf_path)
# get the tflogger
tflogger = SummaryWriter(tf_path)

# train custom BERT
seed_everything(42)

train_customBERT(model, loss_fn=torch.nn.BCEWithLogitsLoss(), optimizer=optimizer, train_loader=training_loader,
                 valid_loader=testing_loader, num_epochs=EPOCHS, eval_every=len(training_loader) // 2,
                 file_path=file_path, validate=True, tflogger=tflogger)
