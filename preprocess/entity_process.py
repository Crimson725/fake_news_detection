import pandas as pd
import spacy
import spacy_component
from pykeen.datasets import CSKG
import argparse
from rapidfuzz import process


# load the rebel model for relation extraction
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe(
    "rebel",
    after="senter",
    config={
        "device": 3,
        "model_name": "Babelscape/rebel-large",
    },
)

# get the triples factory for indexing
tf = CSKG().training

# entity and relation labels
entity_labels = list(tf.entity_id_to_label.values())
relation_labels = list(tf.relation_id_to_label.values())


def fuzz_index(entity, labels):
    scores = process.cdist([entity], labels, workers=-1)
    max_index = scores.argmax()
    best_match = labels[max_index]
    # return the string
    return best_match


def get_parser():
    argparser = argparse.ArgumentParser(description="processing tool to get entity")
    argparser.add_argument("--entity", action="store_true", help="extract entity")
    argparser.add_argument("--hrt", action="store_true", help="extract hrt")
    argparser.add_argument(
        "--indexing",
        action="store_true",
        help="do fuzzy search to find the entity in the knowledge graph",
    )
    argparser.add_argument("--files", nargs="+", help="files to be processed")

    args = argparser.parse_args()
    return args


def entity_list(doc):
    # take the doc as input and return a dictionary of embeddings

    relation_list = []
    head_span_list = []
    tail_span_list = []

    doc = nlp(doc)

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

    return relation_list, head_span_list, tail_span_list


def triplet_extract(doc):
    doc = nlp(doc)

    combined_list = []
    for _, rel_dict in doc._.rel.items():
        relation = None
        head_span = None
        tail_span = None
        for key, value in rel_dict.items():
            if key == "relation":
                relation = value
            elif key == "head_span":
                head_span = str(value)
            elif key == "tail_span":
                tail_span = str(value)
        if relation and head_span and tail_span:
            combined_list.append([head_span, relation, tail_span])
    return combined_list


def entity_index(column):
    new_column = []
    for row in column:
        row_list = row.strip("[]").split(",")
        new_list = [fuzz_index(item, entity_labels) for item in row_list]
        new_column.append(new_list)
    return new_column


def relation_index(column):
    new_column = []
    for row in column:
        row_list = row.strip("[]").split(",")
        new_list = [fuzz_index(item, relation_labels) for item in row_list]
        new_column.append(new_list)
    return new_column


def trim(x):
    if len(x) >= 1024:
        return x[:1024]
    else:
        return x


def get_entity_list(df):
    result = df["text"].apply(trim).apply(entity_list)
    df["relation"], df["head_entity"], df["tail_entity"] = zip(*result)
    return df


def get_triplet(df):
    result = df["text"].apply(trim).apply(triplet_extract)
    df["triplet"] = result
    return df


def kg_entity_index(df):
    df["head_kg"] = df["head_entity"].apply(
        lambda x: [fuzz_index(i, entity_labels) for i in x.strip("[]").split(",")]
    )
    df["tail_kg"] = df["tail_entity"].apply(
        lambda x: [fuzz_index(i, entity_labels) for i in x.strip("[]").split(",")]
    )
    df["relation_kg"] = df["relation"].apply(
        lambda x: [fuzz_index(i, relation_labels) for i in x.strip("[]").split(",")]
    )


def process_triplets(df):
    def process_row(row):
        processed_triplets = []
        for triplet in row:
            head, relation, tail = triplet
            head_index = fuzz_index(head, entity_labels)
            relation_index = fuzz_index(relation, relation_labels)
            tail_index = fuzz_index(tail, entity_labels)
            processed_triplets.append([head_index, relation_index, tail_index])
        return processed_triplets

    df["triplet_kg"] = df["triplet"].apply(process_row)
    return df


def kg_index(df):
    df["head_kg"] = df["head_entity"].apply(
        lambda x: [fuzz_index(i, entity_labels) for i in x]
    )
    df["tail_kg"] = df["tail_entity"].apply(
        lambda x: [fuzz_index(i, entity_labels) for i in x]
    )
    df["relation_kg"] = df["relation"].apply(
        lambda x: [fuzz_index(i, relation_labels) for i in x]
    )


def main(params):
    if params.entity:
        if params.indexing:
            for file in params.files:
                df = pd.read_csv(file)
                df = get_entity_list(df)
                kg_index(df)
                df.to_csv(file.replace(".csv", "_entity_index.csv"), index=False)
        else:
            for file in params.files:
                df = pd.read_csv(file)
                df = get_entity_list(df)
                df.to_csv(file.replace(".csv", "_entity.csv"), index=False)
    elif params.hrt:
        if params.indexing:
            for file in params.files:
                df = pd.read_csv(file)
                df = get_triplet(df)
                df = process_triplets(df)
                df.to_csv(file.replace(".csv", "_hrt_index.csv"), index=False)
        else:
            for file in params.files:
                df = pd.read_csv(file)
                df = get_triplet(df)
                df.to_csv(file.replace(".csv", "_hrt.csv"), index=False)


if __name__ == "__main__":
    params = get_parser()
    main(params)
