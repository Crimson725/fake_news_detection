# README
This repository contains the code for my undergraduate thesis.
# SETUP
## Clone the repository
```bash
git clone https://github.com/Crimson725/fake_news_detection.git
```
## Install dependencies
```bash
pip install -r requirements.txt
```
## BERT and Word Embeddings
**Make sure you have git lfs installed.**

After cloning the repository, go to `model_files/` and run the following command to get `bert-base-uncased`:
```bash
git clone https://huggingface.co/bert-base-uncased
```
For FastText embedding (which is used for experiment only), run the following command to download `wiki-news-300d-1M.vec`:
```bash
wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip
unzip wiki-news-300d-1M.vec.zip
```
## KNOWLEDGE GRAPH EMBEDDING
The knowledge graph embeddings are trained using [PyKeen](https://github.com/pykeen/pykeen). You may want to check their docs about the usage.
[Here](https://drive.google.com/file/d/1-5GuLDxXMVX8ZO4JXpQWZ_Tf_GwV_4AC/view?usp=sharing) is the TransE embedding used in the project.
After downloading the embedding, put it in `model_files/TransE/`.
# DATASETS
Datasets need to be in csv format. The columns should be `id`, `title`, `text`, `label`. They need to be put in `datasets/dataset_name/`.
Original LUN and SLN datasets that are used can be found [here.](https://drive.google.com/drive/folders/1XNVKQ_W6JQRUDam-nFkhtXAWmpNPYFVf?usp=sharing)
# PREPROCESSING
Put the datasets you want to process in `preprocess/`, run the following command:
```bash
python entity_process.py --hrt --indexing --files example1.csv example2.csv
```
# TRAIN and EVALUATION
The model will use DDP to train on multiple GPUs.

Run the following command to train the model in the default setting:
```bash
CUDA_VISIBLE_DEVICES='0,1' torchrun --nproc_per_node=2 DDP_trainer.py --dataset "example/example1.csv" --valid_dataset "example/example2.csv" --hrt --hrt_embedding --bert --epochs 10
```
After training, run the following command to evaluate the model:
```bash
python eval.py --eval_dataset "example/example3.csv" --model_path "model_files/DDP_trained_models/example/model.pt" --hrt --hrt_embedding --bert
```
For settings about training and evaluating the model, you may want to check `utils/common_util.py`.
# INFERENCE
Codes for Inference need some modification, for now `inference.py` is incomplete. You can run the following the command to get the inference result for the text provided in the `inference.py`:
```bash
python inference.py --model_path "example/example.pt"
```
# CREDITS
`spacy_components.py` is from [here.](https://github.com/Babelscape/rebel)