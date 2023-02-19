import torch
from torch import nn
import torch.nn.init as init
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
        # the size for bert base is 768
        size = 768
        num_heads = 12

        # the size of the entity embedding
        entity_size = 50

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
                embed_dim=size * 2, num_heads=num_heads
            )

        # add dropout
        self.dropout = torch.nn.Dropout(dropout)
        # add classification layer
        if self.params.multihead_attention:
            if self.params.entity:
                # size*3 = multihead attention
                # +50 = entity embedding shape
                self.classifier = torch.nn.Linear(size * 2 + entity_size, 1)
            else:
                self.classifier = torch.nn.Linear(size * 2, 1)
        else:
            if self.params.entity:
                self.classifier = torch.nn.Linear(size + entity_size, 1)
            else:
                self.classifier = torch.nn.Linear(size, 1)

        # initialize the weights using Xavier Normal
        self.init_weights()

    def init_weights(self):
        if self.params.lstm:
            for name, param in self.lstm.named_parameters():
                if "weight" in name:
                    init.xavier_normal_(param)
        if self.params.multihead_attention:
            for name, param in self.multihead_attention.named_parameters():
                if "weight" in name:
                    init.xavier_normal_(param)

    # TODO: FIX THE FORWARD FUNCTION FOR ABLATION EXPERIMENT
    def forward(self, ids, mask, token_type_ids, entity_embedding=None):

        _, bert_output = self.l1(
            ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False
        )
        if hasattr(self, "lstm") and hasattr(self, "multihead_attention"):
            lstm_output, _ = self.lstm(bert_output)
            multihead_output, _ = self.multihead_attention(
                lstm_output, lstm_output, lstm_output
            )
            # final_output = torch.cat((lstm_output, multihead_output), dim=-1)
            final_output = multihead_output

            if self.params.entity:
                # attention_embedding = self.self_attention(embeddings)
                final_output = torch.cat((final_output, entity_embedding), dim=-1)
            dropout_output = self.dropout(final_output)
            classifier_output = self.classifier(dropout_output)
            final_output = torch.sigmoid(classifier_output)
            return final_output
        elif hasattr(self, "lstm") and not hasattr(self, "multihead_attention"):
            lstm_output, _ = self.lstm(bert_output)
            dropout_output = self.dropout(lstm_output)
            classifier_output = self.classifier(dropout_output)
            final_output = torch.sigmoid(classifier_output)
            return final_output
        elif hasattr(self, "multihead_attention") and not hasattr(self, "lstm"):
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

        # initialize the weights using Xavier Normal
        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)

    def forward(self, tensor_list):
        if len(tensor_list) == 0 or tensor_list is None:
            # avoid empty tensor error
            return torch.nn.init.xavier_uniform_(torch.zeros(1, 50)).squeeze(0)
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
