import os

path = os.getcwd()

# path
DATA_PATH = os.path.join(path, "datasets/")
DESTINATION_PATH = os.path.join(path, "model_files/trained_models")
DDP_DESTINATION_PATH = os.path.join(path, "model_files/DDP_trained_models")
BERT_BASE_PATH = os.path.join(path, "model_files/bert-base-uncased")
PLOT_PATH = os.path.join(path, "figs")
# kg embedding path
KG_PATH = os.path.join(path, "model_files/CSKG_embeddings/TransE/trained_model.pkl")
# KG_PATH = os.path.join(path, "model_files/CSKG_embeddings/AutoSF/trained_model.pkl")
# KG_PATH = os.path.join(path, "model_files/CSKG_embeddings/RotatE/trained_model.pkl")

# label dict
LABEL2ID = {"fake": 0, "real": 1}
ID2LABEL = {k: v for v, k in LABEL2ID.items()}

# LOCAL_RANK = int(os.environ["LOCAL_RANK"])
