import random, os
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from torchtext.data import Field, TabularDataset, BucketIterator, Iterator
import config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained(config.BERT_PATH, mirror='tuna')
# bert tokenizer parameters
MAX_SEQ_LEN = 128
PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)


def seed_everything(seed: int):
    """
    seed everything
    :param seed: int seed
    :return:
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(path, model, valid_loss):
    if path == None:
        return
    state_dict = {'model_state_dict': model.state_dict(), 'valid_loss': valid_loss}
    torch.save(state_dict, path)
    print('model saved to ==>{}'.format(path))


def load_checkpoint(path, model):
    if path == None:
        return
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    print('loading model from <=={}'.format(path))
    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_loss']


def save_metrics(path, train_loss_list, valid_loss_list, global_steps_list):
    if path == None:
        return
    state_dict = {'train_loss_list': train_loss_list, 'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}
    torch.save(state_dict, path)
    print('metrics saved to ==>{}'.format(path))


def load_metrics(path):
    if path == None:
        return
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    print('loading metrics from <=={}'.format(path))
    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']


labels = {'fake': 0,
          'real': 1}


class Dataset:
    def __init__(self, path):
        self.label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
        self.text_field = Field(use_vocab=False, tokenize=tokenizer.encode, lower=False, batch_first=True,
                                include_lengths=False,
                                fix_length=MAX_SEQ_LEN, pad_token=PAD_INDEX, unk_token=UNK_INDEX)
        self.fields = [('label', self.label_field), ('title', self.text_field), ('text', self.text_field),
                       ('titletext', self.text_field)]
        self.train, self.valid, self.test = TabularDataset.splits(path=path, train='train.csv', validation='valid.csv',
                                                                  test='test.csv', format='CSV', fields=self.fields,
                                                                  skip_header=True)
        self.train_iter = BucketIterator(self.train, batch_size=16, sort_key=lambda x: len(x.text), device=device,
                                         train=True, sort=True, sort_within_batch=True)
        self.valid_iter = BucketIterator(self.valid, batch_size=16, sort_key=lambda x: len(x.text), device=device,
                                         train=True, sort=True, sort_within_batch=True)
        self.test_iter = Iterator(self.test, batch_size=16, device=device, train=False, shuffle=False, sort=False)

    def __len__(self):
        return len(self.train) + len(self.valid) + len(self.test)
