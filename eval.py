import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

import CONFIG
import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")


# TODO: REWRITE THE VALIDATION FUNCTION

# for the original BERT
def evaluate(model, test_loader):
    y_pred = []
    y_true = []
    model.eval()
    with torch.no_grad():
        for _, data in enumerate(test_loader):
            ids = data["ids"].to(device, dtype=torch.long)
            mask = data["mask"].to(device, dtype=torch.long)
            token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
            targets = data["targets"].to(device, dtype=torch.float)
            outputs = model(ids, mask, token_type_ids)
            y_pred.extend(torch.argmax(outputs, 1).tolist())
            y_true.extend(targets.tolist())
    print("Classification Report: ")
    print(classification_report(y_true, y_pred, target_names=CONFIG.ID2LABEL.values()))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    ax = plt.subplot()
    fig = sns.heatmap(cm, annot=True, ax=ax, fmt="d", cmap="Blues")
    fig.get_figure().savefig(CONFIG.PLOT_PATH + "/" + "confusion_matrix.png")

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")

    ax.xaxis.set_ticklabels(CONFIG.ID2LABEL.values())
    ax.yaxis.set_ticklabels(CONFIG.ID2LABEL.values())


# for new
def validation(logger, testing_loader, model,device):
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for _, data in enumerate(testing_loader):
            ids = data["ids"].to(device, dtype=torch.long)
            mask = data["mask"].to(device, dtype=torch.long)
            token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
            targets = data["targets"].to(device, dtype=torch.float)
            outputs = model(ids, mask, token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(outputs.cpu().detach().numpy().tolist())
    outputs = np.array(fin_outputs) >= 0.5
    accuracy = metrics.accuracy_score(fin_targets, outputs)
    f1_score_micro = metrics.f1_score(fin_targets, outputs, average="micro")
    f1_score_macro = metrics.f1_score(fin_targets, outputs, average="macro")
    recall = metrics.recall_score(fin_targets, outputs)
    print(f"Accuracy Score = {accuracy}")
    print(f"F1 Score (Micro) = {f1_score_micro}")
    print(f"F1 Score (Macro) = {f1_score_macro}")
    print(f"Recall Score = {recall}")
    logger.log(
        f"Accuracy Score = {accuracy}, F1 Score (Micro) = {f1_score_micro}, F1 Score (Macro) = {f1_score_macro}, Recall Score = {recall}"
    )


# define the customBERT
# model = BertModel.from_pretrained(
#     'model_files/trained_models/2022-12-29_12-29-41_customBERT/model.pt',
#     config='model_files/trained_models/2022-12-29_12-29-41_customBERT/config.json')
# model.to(device)
# # define the dataset and dataloader
# train_size = 0.75
# df = pd.read_csv(CONFIG.DATA_PATH + '/train.csv')
# train_dataset = df.sample(frac=train_size, random_state=200).reset_index(drop=True)
# test_dataset = df.drop(train_dataset.index).reset_index(drop=True)
#
# train_set = CustomDataset(train_dataset, tokenizer, CONFIG.MAX_LEN)
# test_set = CustomDataset(test_dataset, tokenizer, CONFIG.MAX_LEN)
# train_params = {'batch_size': CONFIG.TRAIN_BATCH_SIZE,
#                 'shuffle': True,
#                 'num_workers': 0
#                 }
#
# test_params = {'batch_size': CONFIG.VALID_BATCH_SIZE,
#                'shuffle': True,
#                'num_workers': 0
#                }
#
# training_loader = DataLoader(train_set, **train_params)
# testing_loader = DataLoader(test_set, **test_params)
# evaluate(model, testing_loader)
