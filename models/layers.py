import torch
from torch import nn
from transformers import BertModel, BertForSequenceClassification
import CONFIG
import nltk
from nltk import tokenize


class BERT(nn.Module):
    def __init__(self, config):
        super(BERT, self).__init__()
        # load the bert with provided config
        self.config = config
        pretrained_model = CONFIG.BERT_BASE_PATH
        self.encoder = BertForSequenceClassification.from_pretrained(
            pretrained_model, config=self.config
        )

    def forward(self, text, label):
        loss, text_fea = self.encoder(text, labels=label)[:2]
        # print(loss,text_fea)
        return loss, text_fea


class customBERT(nn.Module):
    def __init__(self, config, params):
        global size, num_heads
        global pretrained_model

        super(customBERT, self).__init__()
        self.params = params
        dropout = self.params.dropout

        # load the bert with provided config
        self.config = config

        pretrained_model = CONFIG.BERT_BASE_PATH
        size = 768
        num_heads = 12

        self.l1 = BertModel.from_pretrained(pretrained_model, config=self.config)

        # initialize lstm layer
        if self.params.lstm:
            # hidden_size corresponds to bert
            self.lstm = nn.LSTM(
                input_size=size,
                hidden_size=size,
                num_layers=1,
                bidirectional=True,
            )
        # initialize multihead attention layer
        if self.params.multihead_attention:
            self.multihead_attention = nn.MultiheadAttention(
                embed_dim=size, num_heads=num_heads
            )

        # if self.params.entity:
        #     self.self_attention = SelfAttention()

        # add dropout
        self.dropout = torch.nn.Dropout(dropout)
        # add classification layer
        if self.params.multihead_attention:
            if self.params.entity:
                self.classifier = torch.nn.Linear(size * 3 + 50, 1)
            else:
                self.classifier = torch.nn.Linear(size * 3, 1)
        else:
            self.classifier = torch.nn.Linear(size, 1)

    # TODO: FIX THE FORWARD FUNCTION FOR ABLATION EXPERIMENT
    def forward(self, ids, mask, token_type_ids, entity_embedding):

        _, bert_output = self.l1(
            ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False
        )
        if hasattr(self, "lstm") and hasattr(self, "multihead_attention"):
            lstm_output, _ = self.lstm(bert_output)
            multihead_output, _ = self.multihead_attention(
                bert_output, bert_output, bert_output
            )
            cat_lstm_multihead = torch.cat((lstm_output, multihead_output), dim=-1)

            # get the self attention for the attention list
            if self.params.entity:
                # attention_embedding = self.self_attention(embeddings)
                cat_lstm_multihead = torch.cat(
                    (cat_lstm_multihead, entity_embedding), dim=-1
                )
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
            multihead_output, _ = self.multihead_attention(
                bert_output, bert_output, bert_output
            )
            dropout_output = self.dropout(multihead_output)
            classifier_output = self.classifier(dropout_output)
            final_output = torch.sigmoid(classifier_output)
            return final_output
        else:
            dropout_output = self.dropout(bert_output)
            classifier_output = self.classifier(dropout_output)
            final_output = torch.sigmoid(classifier_output)
            return final_output


class SelfAttention(nn.Module):
    # take a list of tensors as input
    def __init__(self, input_size=50, attention_size=128):
        super(SelfAttention, self).__init__()
        self.input_size = input_size
        self.attention_size = attention_size
        self.fc1 = nn.Linear(input_size, attention_size)
        self.fc2 = nn.Linear(attention_size, 1)

    def forward(self, tensor_list):
        # calculate attention scores
        attention_scores = torch.zeros(len(tensor_list), 1)
        for i, tensor in enumerate(tensor_list):
            tensor = tensor.view(-1, self.input_size)
            attention_scores[i] = self.fc2(torch.tanh(self.fc1(tensor)))

        # normalize the scores
        attention_weights = torch.softmax(attention_scores, dim=0)

        # apply the scores to the input tensors
        weighted_tensors = [
            attention_weights[i] * tensor for i, tensor in enumerate(tensor_list)
        ]
        weighted_tensors = torch.stack(weighted_tensors)

        # return the sum of weighted tensors
        return torch.sum(weighted_tensors, dim=0).view(1, -1).squeeze(0)
