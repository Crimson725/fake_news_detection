# README
This repository contains the code for fake news detection based on knowledge-guided semantic analysis
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
Run the following the command to get the inference result for the text provided in the `inference.py`:
```bash
python inference.py --model_path "example/example.pt"
```
# CREDITS
`spacy_components.py` is from [here.](https://github.com/Babelscape/rebel)