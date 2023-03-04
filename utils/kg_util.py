from typing import List
import pykeen
from pykeen.datasets import CSKG
import torch
import torch.nn.init as init
import re
import CONFIG


class KG_embedding:
    # used to generate entity embedding for a document
    def __init__(self, aggregator):

        # load the embedding model (pkl file)
        self.model = torch.load(CONFIG.KG_PATH, map_location=torch.device("cpu"))
        # get embeddings from the model
        self.eneity_representation = (
            self.model.entity_representations[0](indices=None).detach().cpu().numpy()
        )
        self.relation_representation = (
            self.model.relation_representations[0](indices=None).detach().cpu().numpy()
        )
        # triple factory for indexing
        self.tf = CSKG().training

        # label for indexing
        self.entity_labels = list(self.tf.entity_id_to_label.values())
        self.relation_labels = list(self.tf.relation_id_to_label.values())

        # shape of the embedding
        self.entity_embedding_shape = self.eneity_representation.shape[1]
        self.relation_embedding_shape = self.relation_representation.shape[1]

        # the aggregator will take a list of tensors and return a single tensor
        # self.aggregator = SelfAttention(input_size=self.entity_embedding_shape)
        self.aggregator = aggregator

    def generate_entity_embedding(self, entity_list: List[str]) -> List[torch.Tensor]:
        # list of all the entity embeddings for the doc
        # return a list of tensors
        embeddings = []
        for i in entity_list:
            try:
                # add to the embeddings list
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
        return embedding

    def generate_relation_embedding(
        self, relation_list: List[str]
    ) -> List[torch.Tensor]:
        # list of all the entity embeddings for the doc
        # return a list of tensors
        embeddings = []
        for i in relation_list:
            try:
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
