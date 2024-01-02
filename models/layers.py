import torch
from torch import nn
import torch.nn.init as init
from transformers import BertModel
import CONFIG


# disable the warning
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()


class TF_BERT(nn.Module):
    def __init__(self, config, params):
        super(TF_BERT, self).__init__()
        self.params = params
        dropout = self.params.dropout

        # load the bert with provided config
        self.config = config

        pretrained_model = CONFIG.BERT_BASE_PATH
        # the size for bert base is 768
        self.size = 768
        # multihead attention
        self.num_heads = 12

        # max_length for hrt list
        self.max_length = CONFIG.LENGTH

        # for hrt_embedding
        self.embed_size = 50
        self.hrt_attn_heads = 5

        self.l1 = BertModel.from_pretrained(pretrained_model, config=self.config)

        # hrt embedding output size
        self.hrt_output_size = 128

        # add dropout
        self.dropout = torch.nn.Dropout(dropout)

        if self.params.hrt:
            # add gating for hrt score
            self.gating = Gating(params=self.params)
            self.aggregator = FCNet(params=self.params, input_size=CONFIG.LENGTH)

        # initialize the classifier
        if self.params.hrt_embedding:
            self.hrt_attention = hrt_SelfAttention(
                embed_size=self.embed_size,
                heads=self.hrt_attn_heads,
                output_size=self.hrt_output_size,
            )
            self.classifier = torch.nn.Linear(self.size + +self.hrt_output_size, 1)
        else:
            self.classifier = torch.nn.Linear(self.size, 1)

        # initialize the weights for modules of the model
        self.init_weights()

    def init_weights(self):
        if self.params.hrt:
            for name, param in self.gating.named_parameters():
                if "weight" in name:
                    init.xavier_normal_(param)
            for name, param in self.aggregator.named_parameters():
                if "weight" in name:
                    init.xavier_normal_(param)
        if self.params.hrt_embedding:
            for name, param in self.hrt_attention.named_parameters():
                if "weight" in name:
                    init.xavier_normal_(param)

    def forward(
        self,
        ids,
        mask,
        token_type_ids,
        hrt_score_list=None,
        hrt_embedding_list=None,
    ):
        _, bert_output = self.l1(
            ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False
        )
        if self.params.hrt:
            hrt_score = self.aggregator(hrt_score_list)

        if self.params.hrt_embedding:
            hrt_embedding = self.hrt_attention(hrt_embedding_list)
            bert_output = torch.cat((bert_output, hrt_embedding), dim=-1)
        dropout_output = self.dropout(bert_output)
        classifier_output = self.classifier(dropout_output)
        final_output = torch.sigmoid(classifier_output)
        if self.params.hrt:
            final_output = self.gating(final_output, hrt_score)
        return final_output


class TF_glove_fasttext(nn.Module):
    def __init__(self, params):
        super(TF_glove_fasttext, self).__init__()
        self.params = params
        dropout = self.params.dropout

        # the size for bert base is 768
        self.size = 300
        # multihead attention
        self.num_heads = 12

        # max_length for hrt list
        self.max_length = CONFIG.LENGTH

        # for hrt_embedding
        self.embed_size = 50
        self.hrt_attn_heads = 5

        # hrt embedding output size
        self.hrt_output_size = 128

        # add dropout
        self.dropout = torch.nn.Dropout(dropout)

        self.mha = nn.MultiheadAttention(self.size, self.num_heads)

        if self.params.hrt:
            # add gating for hrt score
            self.gating = Gating(params=self.params)
            self.aggregator = FCNet(params=self.params, input_size=CONFIG.LENGTH)

        # initialize the classifier
        if self.params.hrt_embedding:
            self.hrt_attention = hrt_SelfAttention(
                embed_size=self.embed_size,
                heads=self.hrt_attn_heads,
                output_size=self.hrt_output_size,
            )
            self.classifier = torch.nn.Linear(self.size + +self.hrt_output_size, 1)
        else:
            self.classifier = torch.nn.Linear(self.size, 1)

        # initialize the weights for modules of the model
        self.init_weights()

    def init_weights(self):
        if self.params.hrt:
            for name, param in self.gating.named_parameters():
                if "weight" in name:
                    init.xavier_normal_(param)
            for name, param in self.aggregator.named_parameters():
                if "weight" in name:
                    init.xavier_normal_(param)
        if self.params.hrt_embedding:
            for name, param in self.hrt_attention.named_parameters():
                if "weight" in name:
                    init.xavier_normal_(param)
        for name, param in self.mha.named_parameters():
            if "weight" in name:
                init.xavier_normal_(param)

    def forward(
        self,
        vec,
        hrt_score_list=None,
        hrt_embedding_list=None,
    ):
        embedding = vec
        embedding, _ = self.mha(embedding, embedding, embedding)

        if self.params.hrt:
            hrt_score = self.aggregator(hrt_score_list)

        if self.params.hrt_embedding:
            hrt_embedding = self.hrt_attention(hrt_embedding_list)
            embedding = torch.cat((embedding, hrt_embedding), dim=-1)
        dropout_output = self.dropout(embedding)
        classifier_output = self.classifier(dropout_output)
        final_output = torch.sigmoid(classifier_output)
        if self.params.hrt:
            final_output = self.gating(final_output, hrt_score)
        return final_output


class FCNet(nn.Module):
    def __init__(self, params, input_size):
        super(FCNet, self).__init__()
        self.params = params
        self.dropout = self.params.dropout
        self.input_size = input_size
        self.fc1 = nn.Linear(self.input_size, 128)
        self.dropout1 = nn.Dropout(p=self.dropout)
        self.fc2 = nn.Linear(128, 32)
        self.dropout2 = nn.Dropout(p=self.dropout)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))

        x = self.dropout1(x)
        x = nn.functional.relu(self.fc2(x))

        x = self.dropout2(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x


class Gating(nn.Module):
    def __init__(self, params, input_size=2, hidden_size=256):
        super(Gating, self).__init__()
        self.params = params
        self.dropout_prob = self.params.dropout
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.LeakyReLU(0.3)
        # self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(self.dropout_prob)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.LeakyReLU(0.3)
        # self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(self.dropout_prob)

        self.fc4 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, score1, score2):
        # Concatenate the scores
        x = torch.cat((score1, score2), dim=1)

        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        # Apply the output layer
        x = self.fc4(x)
        gating_weights = self.sigmoid(x)

        # Compute the gated score
        gated_score = gating_weights * score1 + (1 - gating_weights) * score2

        return gated_score


class hrt_SelfAttention(nn.Module):
    def __init__(self, embed_size, heads, output_size):
        super(hrt_SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.max_length = CONFIG.LENGTH
        self.output_size = output_size
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(
            self.max_length * 3 * embed_size, self.output_size
        )  # max_length*3
        self.relu = nn.ReLU()
        self.mha = nn.MultiheadAttention(embed_size, heads)

    def forward(self, x):
        # print(x.shape)
        attn_output, _ = self.mha(x, x, x)
        res1 = self.flatten(attn_output)
        res2 = self.fc(res1)
        res3 = self.relu(res2)
        return res3
