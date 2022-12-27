import torch.nn as nn
import torch
from torch import optim

from models.layers import BERT
from utils.helper_function import save_checkpoint, save_metrics, seed_everything
from utils.helper_function import Dataset
import config

EPOCHS = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model,
          optimizer,
          criterion,
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

    model.train()
    for epoch in range(num_epochs):
        for (labels, title, text, titletext), _ in train_loader:
            labels = labels.type(torch.LongTensor).to(device)
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
    print('Finished Training!')


model = BERT().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-6)
dataset = Dataset(config.DATA_PATH)
train_loader = dataset.train_iter
valid_loader = dataset.valid_iter
test_loader = dataset.train_iter

seed_everything(42)
train(model, optimizer=optimizer, criterion=nn.BCELoss(), train_loader=train_loader, valid_loader=valid_loader,
      num_epochs=EPOCHS, eval_every=len(train_loader) // 2, file_path=config.DESTINATION_PATH)
