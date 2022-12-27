import os
path=os.getcwd()

DATA_PATH=os.path.join(path,'datasets/real_and_fake')
DESTINATION_PATH=os.path.join(path,'model_files/trained_models')
BERT_PATH=os.path.join(path,'model_files/bert-base-uncased')
PLOT_PATH=os.path.join(path,'figs')
LABEL2ID={
    'fake':0,
    'real':1
}
ID2LABEL={k:v for v,k in LABEL2ID.items()}
