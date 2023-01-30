import requests
import re
import hashlib

import spacy
from spacy import Language
from typing import List
from spacy.tokens import Doc, Span
from transformers import pipeline
import fastcoref
from fastcoref import spacy_component

import pykeen
import torch


# -1 for cpu
DEVICE = -1
class KG_embedding:
    def __init__(self,path):
        self.model=torch.load(path)
        self.eneity_representation=self.model.entity_representations[0](indices=None).detach().cpu().numpy()
        self.relation_representation=self.model.relation_representation[0](indices=None).detach().cpu().numpy()

    def get_entity_embedding(self,entity:str):
        pass

    def get_relation_embedding(self,relation:str):
        pass
def call_wiki_api(item):
    proxies = {
        "http": "http://127.0.0.1:7890",
        "https": "http://127.0.0.1:7890",
    }

    try:
        url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={item}&language=en&format=json"
        data = requests.get(url, proxies=proxies).json()
        # Return the first id (Could upgrade this in the future)
        return data["search"][0]["id"]
    except:
        return "id-less"


def extract_triplets(text):
    """
    Function to parse the generated text and extract the triplets
    """
    triplets = []
    relation, subject, relation, object_ = "", "", "", ""
    text = text.strip()
    current = "x"
    for token in text.replace("", "").replace("", "").replace("", "").split():
        if token == "":
            current = "t"
            if relation != "":
                triplets.append(
                    {
                        "head": subject.strip(),
                        "type": relation.strip(),
                        "tail": object_.strip(),
                    }
                )
                relation = ""
            subject = ""
        elif token == "":
            current = "s"
            if relation != "":
                triplets.append(
                    {
                        "head": subject.strip(),
                        "type": relation.strip(),
                        "tail": object_.strip(),
                    }
                )
            object_ = ""
        elif token == "":
            current = "o"
            relation = ""
        else:
            if current == "t":
                subject += " " + token
            elif current == "s":
                object_ += " " + token
            elif current == "o":
                relation += " " + token
    if subject != "" and relation != "" and object_ != "":
        triplets.append(
            {"head": subject.strip(), "type": relation.strip(), "tail": object_.strip()}
        )

    return triplets


@Language.factory(
    "rebel",
    requires=["doc.sents"],
    assigns=["doc._.rel"],
    default_config={
        "model_name": "Babelscape/rebel-large",
        "device": DEVICE,
    },
)
class RebelComponent:
    def __init__(
        self,
        nlp,
        name,
        model_name: str,
        device: int,
    ):
        assert model_name is not None, ""
        self.triplet_extractor = pipeline(
            "text2text-generation",
            model=model_name,
            tokenizer=model_name,
            device=device,
        )
        self.entity_mapping = {}
        # Register custom extension on the Doc
        if not Doc.has_extension("rel"):
            Doc.set_extension("rel", default={})

    # def get_wiki_id(self, item: str):
    #     mapping = self.entity_mapping.get(item)
    #     if mapping:
    #         return mapping
    #     else:
    #         res = call_wiki_api(item)
    #         self.entity_mapping[item] = res
    #         return res

    def _generate_triplets(self, sent: Span) -> List[dict]:
        output_ids = self.triplet_extractor(
            sent.text, return_tensors=True, return_text=False
        )[0]["generated_token_ids"]["output_ids"]
        extracted_text = self.triplet_extractor.tokenizer.batch_decode(output_ids[0])
        extracted_triplets = extract_triplets(extracted_text[0])
        return extracted_triplets

    def set_annotations(self, doc: Doc, triplets: List[dict]):
        for triplet in triplets:

            # Remove self-loops (relationships that start and end at the entity)
            if triplet["head"] == triplet["tail"]:
                continue

            # Use regex to search for entities
            head_span = re.search(triplet["head"], doc.text)
            tail_span = re.search(triplet["tail"], doc.text)

            # Skip the relation if both head and tail entities are not present in the text
            # Sometimes the Rebel model hallucinates some entities
            if not head_span or not tail_span:
                continue

            index = hashlib.sha1(
                "".join([triplet["head"], triplet["tail"], triplet["type"]]).encode(
                    "utf-8"
                )
            ).hexdigest()
            if index not in doc._.rel:
                # Get wiki ids and store results
                doc._.rel[index] = {
                    "relation": triplet["type"],
                    "head_span": {
                        "text": triplet["head"],
                    },
                    "tail_span": {
                        "text": triplet["tail"],
                    },
                }

    def __call__(self, doc: Doc) -> Doc:
        for sent in doc.sents:
            sentence_triplets = self._generate_triplets(sent)
            self.set_annotations(doc, sentence_triplets)
        return doc


# coref


if __name__ == "__main__":
    # get coref pipeline
    nlp = spacy.load(
        "en_core_web_sm", exclude=["parser", "lemmatizer", "ner", "textcat"]
    )
    nlp.add_pipe(
        "fastcoref",
        config={
            "model_architecture": "LingMessCoref",
            "model_path": "biu-nlp/lingmess-coref",
            "device": DEVICE,
        },
    )
    # get rel_ext pipeline
    # rel_ext = spacy.load(
    #     "en_core_web_sm", disable=["ner", "lemmatizer", "attribute_rules", "tagger"]
    # )
    # rel_ext.add_pipe(
    #     "rebel",
    #     config={
    #         "device": DEVICE,  # Number of the GPU, -1 if want to use CPU
    #         "model_name": "Babelscape/rebel-large",
    #     },
    # )  # Model used, will default to 'Babelscape/rebel-large'

    text = "Alice goes down the rabbit hole. Where she would discover a new reality beyond her expectations."

    doc = nlp(text, component_cfg={"fastcoref": {"resolve_text": True}})
    # get coref results
    coref_text = doc._.coref_resolved
    print(coref_text)

    # get the rel_ext results
    # result = rel_ext(coref_text)
    # print(result._.rel)
