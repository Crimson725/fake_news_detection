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
        # initialize the aggregator (for sent level classification)
        self.aggregator = None

        # initialize lstm layer
        if self.params.lstm:
            # hidden_size corresponds to bert
            self.lstm = nn.LSTM(
                input_size=768,
                hidden_size=768,
                num_layers=self.params.num_layers,
                bidirectional=True,
            )
        # initialize multihead attention layer
        if self.params.multihead_attention:
            self.multihead_attention = nn.MultiheadAttention(
                embed_dim=768, num_heads=self.params.num_heads
            )
        # add dropout
        self.dropout = torch.nn.Dropout(dropout)
        # add classification layer
        if self.params.multihead_attention:
            self.classifier = torch.nn.Linear(768 * 3, 1)
        else:
            self.classifier = torch.nn.Linear(768, 1)
        # add sent level aggregator
        if self.params.mode == "sent":
            self.aggregator = SentenceAggregator(
                input_size=768, hidden_size=768, num_classes=1
            )

    def forward(self, ids, mask, token_type_ids):
        _, bert_output = self.l1(
            ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False
        )
        if self.params.mode == "doc":
            if hasattr(self, "lstm") and hasattr(self, "multihead_attention"):
                lstm_output, _ = self.lstm(bert_output)
                multihead_output, _ = self.multihead_attention(
                    bert_output, bert_output, bert_output
                )
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
        if self.params.mode == "sent":
            sentence_embeddings = []
            for i in range(ids.size(1)):
                sentence_embeddings.append(bert_output[:, i, :])
            sentence_embeddings = torch.cat(sentence_embeddings, 0)
            if hasattr(self, "lstm"):
                sentence_embeddings, _ = self.lstm(sentence_embeddings)
            if hasattr(self, "multihead_attention"):
                sentence_embeddings, _ = self.multihead_attention(
                    sentence_embeddings, sentence_embeddings, sentence_embeddings
                )
            final_output = self.aggregator(sentence_embeddings)
            return final_output


class SentenceAggregator(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=1, num_heads=1):
        super(SentenceAggregator, self).__init__()
        self.fc1 = nn.Linear(input_size * num_heads * num_layers, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, inputs):
        x = self.fc1(inputs)
        x = self.relu(x)
        x = self.fc2(x)
        return x
