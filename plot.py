import matplotlib.pyplot as plt
import torch

from train import testing_loader
from utils.helper_function import load_metrics, device
import CONFIG
from models.layers import customBERT
from transformers import BertConfig
from torchviz import make_dot

import os
from tensorboardX import SummaryWriter


# on my windows
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'


def plot_metrics():
    train_loss_list, valid_loss_list, global_steps_list = load_metrics(CONFIG.DESTINATION_PATH + "/" + 'metrics.pt')
    plt.plot(global_steps_list, train_loss_list, label='Train')
    plt.plot(global_steps_list, valid_loss_list, label='Validation')
    plt.xlabel('Global Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(CONFIG.PLOT_PATH + '/' + 'loss.png')


def plot_model_structure(model, inputs, file_name):
    dot = make_dot(model(*inputs), params=dict(model.named_parameters()))
    dot.render(CONFIG.PLOT_PATH + '/' + file_name, view=True)

def tensorlogs(model, inputs):
    """
    using the tensorboard
    :return:
    """
    writer=SummaryWriter(log_dir=CONFIG.TENSORLOG_PATH)
    writer.add_graph(model, inputs)

#TODO: REWRITE THE CODES FOR PLOTTING THE MODEL STRUCTURE USING EXAMPLE INPUT


config = BertConfig(label2id=CONFIG.LABEL2ID, id2label=CONFIG.ID2LABEL)
model = customBERT(config).to(device)
params=dict(model.named_parameters())

# get one input
batch = next(iter(testing_loader))
ids = batch['ids'].to(device, dtype=torch.long)
mask = batch['mask'].to(device, dtype=torch.long)
token_type_ids = batch['token_type_ids'].to(device, dtype=torch.long)
inputs = (ids, mask, token_type_ids)

# plot_model_structure(model, inputs, 'model.png')
tensorlogs(model, inputs)