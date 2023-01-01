import torch
from torch import nn
from transformers import BertModel, BertForSequenceClassification
import CONFIG


class BERT(nn.Module):
    def __init__(self, config):
        super(BERT, self).__init__()
        # load the bert with provided config
        self.config = config
        pretrained_model = CONFIG.BERT_PATH
        self.encoder = BertForSequenceClassification.from_pretrained(
            pretrained_model, config=self.config
        )

    def forward(self, text, label):
        loss, text_fea = self.encoder(text, labels=label)[:2]
        # print(loss,text_fea)
        return loss, text_fea


class customBERT(nn.Module):
    def __init__(self, config, params):
        super(customBERT, self).__init__()
        # load the bert with provided config
        self.config = config
        pretrained_model = CONFIG.BERT_PATH
        self.l1 = BertModel.from_pretrained(pretrained_model, config=self.config)
        self.l2 = torch.nn.Dropout(params.dropout)
        # classification layer
        self.l3 = torch.nn.Linear(768, 1)

    def forward(self, ids, mask, token_type_ids):
        _, output_1 = self.l1(
            ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False
        )
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        output = torch.sigmoid(output)
        return output
