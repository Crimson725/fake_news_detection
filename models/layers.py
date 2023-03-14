import torch
from torch import nn
import torch.nn.init as init
from transformers import BertModel, BertForSequenceClassification
import CONFIG

# disable the warning
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()


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
        if self.params.entity:
            self.entity_size = self.params.entity_size

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
                # 768*2*2+entity_size
                self.classifier = torch.nn.Linear(self.size * 4 + self.entity_size, 1)
            else:
                self.classifier = torch.nn.Linear(self.size * 4, 1)
        elif self.params.lstm and not self.params.multihead_attention:
            if self.params.entity:
                # 768*2+entity_size
                self.classifier = torch.nn.Linear(self.size * 2 + self.entity_size, 1)
            else:
                self.classifier = torch.nn.Linear(self.size * 2, 1)
        elif self.params.multihead_attention and not self.params.lstm:
            if self.params.entity:
                # 768+entity_size
                self.classifier = torch.nn.Linear(self.size + self.entity_size, 1)
            else:
                self.classifier = torch.nn.Linear(self.size, 1)
        else:
            if self.params.entity:
                # 768+entity_size
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
            # 768*2+entity_size
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
            # shape 768+entity_size
            if self.params.entity:
                final_output = torch.cat((final_output, entity_embedding), dim=-1)
            dropout_output = self.dropout(final_output)
            classifier_output = self.classifier(dropout_output)
            final_output = torch.sigmoid(classifier_output)
            return final_output
        else:
            # shape 768+entity_size
            if self.params.entity:
                bert_output = torch.cat((bert_output, entity_embedding), dim=-1)
            dropout_output = self.dropout(bert_output)
            classifier_output = self.classifier(dropout_output)
            final_output = torch.sigmoid(classifier_output)
            return final_output


class SelfAttention(nn.Module):
    # take a list of tensors as input
    # using this to aggregate the context information of the entities in the news content
    # the input size depends on the embedding size
    # the output shape is (1, input_size)
    def __init__(self, input_size, attention_size=128):
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
