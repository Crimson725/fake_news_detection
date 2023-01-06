import hashlib
from typing import List
import re

import spacy
from spacy import Language
from spacy.tokens import Doc, Span
from transformers import pipeline

from utils.kg_util import call_wiki_api
from utils.kg_util import extract_triplets

DEVICE = 0

coref = spacy.load('en_core_web_sm', disable=['ner', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer'])

coref.add_pipe(
    "xx_coref", config={"chunk_size": 2500, "chunk_overlap": 2, "device": DEVICE})
rel_ext = spacy.load('en_core_web_sm', disable=['ner', 'lemmatizer', 'attribute_rules', 'tagger'])
rel_ext.add_pipe("rebel", config={
    'device': DEVICE,  # Number of the GPU, -1 if want to use CPU
    'model_name': 'Babelscape/rebel-large'}  # Model used, will default to 'Babelscape/rebel-large' if not given
                 )


@Language.factory(
    "rebel",
    requires=["doc.sents"],
    assigns=["doc._.rel"],
    default_config={
        "model_name": "Babelscape/rebel-large",
        "device": 0,
    },
)
class RebelComponent:
    def __init__(self, nlp, name, model_name: str, device: int):
        assert model_name is not None, "No model name specified "
        self.triplet_extractor = pipeline("text2text-generation", model=model_name, tokenizer=model_name, device=device)
        self.entity_mapping = {}
        if not Doc.has_extension("rel"):
            Doc.set_extension("rel", default={})

    def get_wiki_id(self, item: str):
        mapping = self.entity_mapping.get(item)
        if mapping:
            return mapping
        else:
            res = call_wiki_api(item)
            self.entity_mapping[item] = res
            return res

    def _generate_triplets(self, sent: Span) -> List[dict]:
        output_ids = \
            self.triplet_extractor(sent.text, return_tensors=True, return_text=False)[0]["generated_token_ids"][
                "output_ids"]
        extracted_text = self.triplet_extractor.tokenizer.batch_decode(output_ids[0])
        extracted_triplets = extract_triplets(extracted_text[0])
        return extracted_triplets

    def set_annotations(self, doc: Doc, triplets: List[dict]):
        for triplet in triplets:
            if triplet['head'] == triplet['tail']:
                continue
            head_span = re.search(triplet['head'], doc.text)
            tail_span = re.search(triplet['tail'], doc.text)
            if not head_span or not tail_span:
                continue
            index = hashlib.sha1(
                ''.join([triplet['head'], triplet['tail'], triplet['type']]).encode('utf-8')).hexdigest()
            if index not in doc._.rel:
                doc._.rel[index] = {"relation": triplet["type"],
                                    "head_span": {'text': triplet['head'], 'id': self.get_wiki_id(triplet['head'])},
                                    "tail_span": {'text': triplet['tail'], 'id': self.get_wiki_id(triplet['tail'])}}

    def __call__(self, doc: Doc) -> Doc:
        for sent in doc.sents:
            sentence_triplets = self._generate_triplets(sent)
            self.set_annotations(doc, sentence_triplets)
        return doc


if __name__ == "__main__":
    input