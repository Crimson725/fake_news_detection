import spacy
from spacy import Language
from typing import List
from spacy.tokens import Doc, Span
from transformers import pipeline

import utils.spacy_component
import pykeen
from pykeen.datasets import CSKG
import torch
from models.layers import SelfAttention
import numpy as np

import CONFIG

# -1 for cpu
DEVICE = -1


class KG_embedding:
    # used to generate entity embedding for a document
    def __init__(self, params):
        self.params = params

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

        self.entity_embedding_shape = self.eneity_representation.shape[1]
        self.relation_embedding_shape = self.relation_representation.shape[1]

        # load rebel model

        self.nlp = spacy.load("en_core_web_sm")
        # if params.coref:
        #     self.nlp.add_pipe("fastcoref",
        #                       config={
        #                           "model_architecture": "LingMessCoref",
        #                           "model_path": "biu-nlp/lingmess-coref",
        #                           "device": f'cuda:{params.device_id}',
        #                       }, )
        self.nlp.add_pipe(
            "rebel",
            after="senter",
            config={
                "device": 0,
                "model_name": "Babelscape/rebel-large",
            },
        )
        # combine a list of tensors into one tensor
        self.aggregator = SelfAttention()

    def get_entity_embedding(self, entity_list: list) -> list:
        # list of all the entity embeddings for the doc
        # return a list of tensors
        embeddings = []
        for i in entity_list:
            try:
                # find the entity_id for indexing
                entity_id = self.tf.entity_to_id[i]
                # add to the embeddings list
                # torch tensor
                embeddings.append(
                    torch.from_numpy(self.eneity_representation[entity_id])
                )

            except:
                embeddings.append(
                    torch.nn.init.xavier_uniform_(torch.zeros(1, self.entity_embedding_shape)).squeeze(0)
                )
        return embeddings

    def get_relation_embedding(self, relation_list: list) -> list:
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
                embeddings.append(
                    torch.nn.init.xavier_uniform_(torch.zeros(1, self.relation_embedding_shape)).squeeze(0)
                )
        return embeddings

    def get_embeddings(self, doc) -> {}:
        # take the doc as input and return a dictionary of embeddings
        embedding_dict = {}

        relation_list = []
        head_span_list = []
        tail_span_list = []

        doc = self.nlp(doc)

        for _, rel_dict in doc._.rel.items():
            for key, value in rel_dict.items():
                if key == "relation":
                    relation_list.append(value)
                elif key == "head_span":
                    head_span_list.append(value)
                elif key == "tail_span":
                    tail_span_list.append(value)
        # deduplicate the list
        relation_list = list(set(relation_list))
        head_span_list = list(set(head_span_list))
        tail_span_list = list(set(tail_span_list))

        # use the aggregator to combine the list and get the final embedding tensor
        embedding_dict["head_span"] = self.aggregator(self.get_entity_embedding(head_span_list))
        embedding_dict["tail_span"] = self.aggregator(self.get_entity_embedding(tail_span_list))
        embedding_dict["relation"] = self.aggregator(self.get_relation_embedding(relation_list))

        return embedding_dict
