from typing import List
import pykeen
from pykeen.datasets import CSKG
import torch
import torch.nn as nn
import torch.nn.init as init
import CONFIG
import re
import os

# LOCAL_RANK = int(os.environ["LOCAL_RANK"])
# # set device (using local rank)
# torch.cuda.set_device(LOCAL_RANK)
# device = torch.device("cuda", LOCAL_RANK)


class KG_embedding:
    # used to generate entity embedding for a document
    def __init__(self, aggregator=None):
        # load the embedding model (pkl file)
        self.model = torch.load(CONFIG.KG_PATH, map_location=torch.device("cpu"))
        # device = torch.device("cuda", LOCAL_RANK)
        # self.model.to(device)
        # get embeddings from the model
        self.entity_representation = (
            self.model.entity_representations[0](indices=None).detach().cpu().numpy()
        )
        self.relation_representation = (
            self.model.relation_representations[0](indices=None).detach().cpu().numpy()
        )

        # triple factory for indexing
        self.tf = CSKG().training

        # shape of the embedding
        self.entity_embedding_shape = self.entity_representation.shape[1]
        self.relation_embedding_shape = self.relation_representation.shape[1]

        # the aggregator will take a list of tensors and return a single tensor
        # self.aggregator = SelfAttention(input_size=self.entity_embedding_shape)
        self.aggregator = aggregator

        # padding for hrt
        self.max_length = CONFIG.LENGTH

    def generate_entity_embedding(self, entity_list: List[str]):
        embeddings = []
        for i in entity_list:
            try:
                # add to the embeddings list
                # conceptnet format
                # for experiment
                # i = "/c/en/" + re.sub("[^A-Za-z0-9]+", "", i).lower()
                entity_id = self.tf.entity_to_id[i]
                embeddings.append(
                    torch.from_numpy(self.eneity_representation[entity_id])
                )

            except:
                # incase the entity is not in the KG
                embeddings.append(
                    init.xavier_uniform_(
                        torch.zeros(1, self.entity_embedding_shape)
                    ).squeeze(0)
                )
        embedding = self.aggregator(embeddings)
        return embedding.data

    def generate_relation_embedding(self, relation_list: List[str]):
        embeddings = []
        for i in relation_list:
            try:
                # conceptnet format
                # for experiment
                # i = "/r/" + re.sub("[^A-Za-z0-9]+", "", i).lower()
                relation_id = self.tf.relation_to_id[i]
                # get the embedding
                embeddings.append(
                    torch.from_numpy(self.relation_representation[relation_id])
                )
            except:
                # incase the relation is not in the KG
                embeddings.append(
                    init.xavier_uniform_(
                        torch.zeros(1, self.relation_embedding_shape)
                    ).squeeze(0)
                )
        embedding = self.aggregator(embeddings)
        return embedding

    def generate_hrt_embedding(self, triplet: List[List[str]]) -> List[torch.Tensor]:
        hrt_embedding_list = []
        for i in range(len(triplet)):
            head_id = torch.as_tensor(self.tf.entity_to_id[triplet[i][0]])
            relation_id = torch.as_tensor(self.tf.relation_to_id[triplet[i][1]])
            tail_id = torch.as_tensor(self.tf.entity_to_id[triplet[i][2]])
            head_embedding = torch.from_numpy(
                self.entity_representation[head_id]
            ).unsqueeze(0)
            relation_embedding = torch.from_numpy(
                self.relation_representation[relation_id]
            ).unsqueeze(0)
            tail_embedding = torch.from_numpy(
                self.entity_representation[tail_id]
            ).unsqueeze(0)
            hrt_embedding = torch.cat(
                [head_embedding, relation_embedding, tail_embedding], dim=0
            )  # shape 3,50
            hrt_embedding_list.append(hrt_embedding)
        # pad the length
        if len(hrt_embedding_list) < self.max_length:
            pad_tensor = torch.cat(
                [
                    torch.zeros(1, self.entity_embedding_shape),
                    torch.zeros(1, self.relation_embedding_shape),
                    torch.zeros(1, self.entity_embedding_shape),
                ],
                dim=0,
            )
            hrt_embedding_list = hrt_embedding_list + [
                pad_tensor for i in range(self.max_length - len(hrt_embedding_list))
            ]
        elif len(hrt_embedding_list) > self.max_length:
            hrt_embedding_list = hrt_embedding_list[: self.max_length]
        return hrt_embedding_list  # list of tensors in shape[3,50]

    def get_triplet_score(self, triplet: List[List[str]]) -> List[float]:
        score_list = []
        for i in range(len(triplet)):
            head_id = torch.as_tensor(self.tf.entity_to_id[triplet[i][0]])
            relation_id = torch.as_tensor(self.tf.relation_to_id[triplet[i][1]])
            tail_id = torch.as_tensor(self.tf.entity_to_id[triplet[i][2]])
            hrt_id = torch.tensor([head_id, relation_id, tail_id]).unsqueeze(0)
            score = self.model.score_hrt(hrt_id)  # shape [1,1]
            score_list.append(score)
        # pad the length of the list
        if len(score_list) < self.max_length:
            score_list = score_list + [
                torch.zeros(1, 1) for i in range(self.max_length - len(score_list))
            ]
        elif len(score_list) > self.max_length:
            score_list = score_list[: self.max_length]
        # score_list = self.batch_normalize_tensors(score_list)
        return score_list
