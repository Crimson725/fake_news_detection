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
        self.params = params
        dropout = self.params.dropout

        # load the bert with provided config
        self.config = config
        pretrained_model = CONFIG.BERT_PATH
        self.l1 = BertModel.from_pretrained(pretrained_model, config=self.config)
        # initialize lstm and multihead attention
        self.lstm = None
        self.multihead_attention = None

        # initialize lstm layer
        if self.params.lstm:
            # hidden_size corresponds to bert
            self.lstm = nn.LSTM(input_size=768,hidden_size=768, num_layers=1, bidirectional=True)
        # initialize multihead attention layer
        if self.params.multihead_attention:
            self.multihead_attention = nn.MultiheadAttention(embed_dim=768, num_heads=12)
        # add dropout
        self.dropout = torch.nn.Dropout(dropout)
        # add classification layer
        if self.params.multihead_attention:
            self.classifier = torch.nn.Linear(768*3, 1)

    def forward(self, ids, mask, token_type_ids):
        _, bert_output = self.l1(
            ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False
        )
        if hasattr(self, "lstm") and hasattr(self, "multihead_attention"):
            lstm_output, _ = self.lstm(bert_output)
            multihead_output, _ = self.multihead_attention(bert_output,bert_output,bert_output)
            cat_lstm_multihead = torch.cat((lstm_output, multihead_output), dim=-1)
            dropout_output = self.dropout(cat_lstm_multihead)
            classifier_output = self.classifier(dropout_output)
            final_output = torch.sigmoid(classifier_output)
            return final_output
        elif hasattr(self, "lstm"):
            lstm_output, _ = self.lstm(bert_output)
            dropout_output = self.dropout(lstm_output)
            classifier_output = self.classifier(dropout_output)
            final_output = torch.sigmoid(classifier_output)
            return final_output
        elif hasattr(self, "multihead_attention"):
            multihead_output, _ = self.multihead_attention(bert_output,bert_output,bert_output)
            dropout_output = self.dropout(multihead_output)
            classifier_output = self.classifier(dropout_output)
            final_output = torch.sigmoid(classifier_output)
            return final_output
        else:
            dropout_output = self.dropout(bert_output)
            classifier_output = self.classifier(dropout_output)
            final_output = torch.sigmoid(classifier_output)
            return final_output
