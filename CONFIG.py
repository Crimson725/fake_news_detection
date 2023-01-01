import os
import pandas as pd

path = os.getcwd()

# path
DATA_PATH = os.path.join(path, "datasets/")
DESTINATION_PATH = os.path.join(path, "model_files/trained_models")
BERT_PATH = os.path.join(path, "model_files/bert-base-uncased")
PLOT_PATH = os.path.join(path, "figs")

# label dict
LABEL2ID = {"fake": 0, "real": 1}
ID2LABEL = {k: v for v, k in LABEL2ID.items()}

DEVICE = "cuda:7"
