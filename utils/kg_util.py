from typing import List
import pykeen
from pykeen.datasets import CSKG
import torch
import torch.nn as nn
import torch.nn.init as init

import re
import CONFIG


class KG_embedding:
    # used to generate entity embedding for a document
    def __init__(self):

        # load the embedding model (pkl file)
        self.model = torch.load(CONFIG.KG_PATH)
        # get embeddings from the model
        self.eneity_representation = (
            self.model.entity_representations[0](indices=None).detach().cpu().numpy()
        )
        self.relation_representation = (
            self.model.relation_representations[0](indices=None).detach().cpu().numpy()
        )
        # triple factory for indexing
        self.tf = CSKG().training

        # shape of the embedding
        self.entity_embedding_shape = self.eneity_representation.shape[1]
        self.relation_embedding_shape = self.relation_representation.shape[1]

        # the aggregator will take a list of tensors and return a single tensor
        self.aggregator = SelfAttention(input_size=self.entity_embedding_shape)

    def generate_entity_embedding(self, entity_list: List[str]) -> List[torch.Tensor]:
        # list of all the entity embeddings for the doc
        # return a list of tensors
        embeddings = []
        for i in entity_list:
            try:
                # find the entity_id for indexing, only for CSKG graph
                # for example: piano->/c/en/piano
                i = "/c/en/" + re.sub("[^A-Za-z0-9]+", "", i).lower()
                entity_id = self.tf.entity_to_id[i]
                # add to the embeddings list
                # torch tensor
                embeddings.append(
                    torch.from_numpy(self.eneity_representation[entity_id])
                )

            except:
                # incase the entity is not in the KG
                embeddings.append(
                    torch.nn.init.xavier_uniform_(
                        torch.zeros(1, self.entity_embedding_shape)
                    ).squeeze(0)
                )
        embedding = self.aggregator(embeddings)
        return embedding

    def generate_relation_embedding(
        self, relation_list: List[str]
    ) -> List[torch.Tensor]:
        # list of all the entity embeddings for the doc
        # return a list of tensors
        embeddings = []
        for i in relation_list:
            try:
                # find the relation_id for indexing
                relation_id = self.tf.relation_to_id[i]
                # get the embedding
                embeddings.append(
                    torch.from_numpy(self.relation_representation[relation_id])
                )
            except:
                # incase the relation is not in the KG
                embeddings.append(
                    torch.nn.init.xavier_uniform_(
                        torch.zeros(1, self.relation_embedding_shape)
                    ).squeeze(0)
                )
        embedding = self.aggregator(embeddings)
        return embedding


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
