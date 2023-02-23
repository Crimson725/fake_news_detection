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
    def __init__(self, config, entity_size, params):

        super(customBERT, self).__init__()
        self.params = params
        dropout = self.params.dropout

        # load the bert with provided config
        self.config = config

        pretrained_model = CONFIG.BERT_BASE_PATH
        # the size for bert base is 768
        self.size = 768
        # multihead attention
        self.num_heads = 12

        # the size of the entity embedding
        self.entity_size = entity_size

        self.l1 = BertModel.from_pretrained(pretrained_model, config=self.config)

        # initialize lstm layer
        if self.params.lstm:
            # hidden_size corresponds to bert
            # output shape = size*2 (due to bidirectional)
            self.lstm = nn.LSTM(
                input_size=self.size,
                hidden_size=self.size,
                num_layers=1,
                bidirectional=True,
            )
        # initialize multihead attention layer
        if self.params.multihead_attention:
            if self.params.lstm:
                self.multihead_attention = nn.MultiheadAttention(
                    embed_dim=self.size * 2, num_heads=self.num_heads
                )
            else:
                self.multihead_attention = nn.MultiheadAttention(
                    embed_dim=self.size, num_heads=self.num_heads
                )

        # add dropout
        self.dropout = torch.nn.Dropout(dropout)

        if self.params.lstm and self.params.multihead_attention:
            if self.params.entity:
                # 768*2*2+50
                self.classifier = torch.nn.Linear(self.size * 4 + self.entity_size, 1)
            else:
                self.classifier = torch.nn.Linear(self.size * 4, 1)
        elif self.params.lstm and not self.params.multihead_attention:
            if self.params.entity:
                # 768*2+50
                self.classifier = torch.nn.Linear(self.size * 2 + self.entity_size, 1)
            else:
                self.classifier = torch.nn.Linear(self.size * 2, 1)
        elif self.params.multihead_attention and not self.params.lstm:
            if self.params.entity:
                # 768+50
                self.classifier = torch.nn.Linear(self.size + self.entity_size, 1)
            else:
                self.classifier = torch.nn.Linear(self.size, 1)
        else:
            if self.params.entity:
                # 768+50
                self.classifier = torch.nn.Linear(self.size + self.entity_size, 1)
            else:
                self.classifier = torch.nn.Linear(self.size, 1)

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
            cat_output = torch.cat((lstm_output, multihead_output), dim=-1)
            # final_output = multihead_output
            # concat the entity embedding to the output
            if self.params.entity:
                # attention_embedding = self.self_attention(embeddings)
                cat_output = torch.cat((cat_output, entity_embedding), dim=-1)
            dropout_output = self.dropout(cat_output)
            classifier_output = self.classifier(dropout_output)
            final_output = torch.sigmoid(classifier_output)
            return final_output
        elif hasattr(self, "lstm") and not hasattr(self, "multihead_attention"):
            lstm_output, _ = self.lstm(bert_output)
            # 768*2+50
            if self.params.entity:
                lstm_output = torch.cat((lstm_output, entity_embedding), dim=-1)
            dropout_output = self.dropout(lstm_output)
            # shape 768*2 (lstm)
            classifier_output = self.classifier(dropout_output)
            final_output = torch.sigmoid(classifier_output)
            return final_output
        elif hasattr(self, "multihead_attention") and not hasattr(self, "lstm"):
            multihead_output, _ = self.multihead_attention(
                bert_output, bert_output, bert_output
            )
            # shape 768
            final_output = multihead_output
            # shape 768+50
            if self.params.entity:
                final_output = torch.cat((final_output, entity_embedding), dim=-1)
            dropout_output = self.dropout(final_output)
            classifier_output = self.classifier(dropout_output)
            final_output = torch.sigmoid(classifier_output)
            return final_output
        else:
            # shape 768+50
            if self.params.entity:
                bert_output = torch.cat((bert_output, entity_embedding), dim=-1)
            dropout_output = self.dropout(bert_output)
            classifier_output = self.classifier(dropout_output)
            final_output = torch.sigmoid(classifier_output)
            return final_output
