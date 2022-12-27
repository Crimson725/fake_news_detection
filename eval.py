import matplotlib.pyplot as plt
import config
from models.layers import BERT
from utils.helper_function import Dataset, load_checkpoint
import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate(model, test_loader):
    y_pred = []
    y_true = []
    model.eval()
    with torch.no_grad():
        for (labels, title, text, titletext), _ in test_loader:
            labels = labels.type(torch.LongTensor).to(device)
            titletext = titletext.type(torch.LongTensor).to(device)
            output = model(titletext, labels)
            _, output = output
            y_pred.extend(torch.argmax(output, 1).tolist())
            y_true.extend(labels.tolist())
    print('Classification Report: ')
    print(classification_report(y_true, y_pred, target_names=config.ID2LABEL.values()))
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    ax = plt.subplot()
    fig=sns.heatmap(cm, annot=True, ax=ax, fmt='d', cmap='Blues')
    fig.get_figure().savefig(config.PLOT_PATH+'/'+'confusion_matrix.png')

    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')

    ax.xaxis.set_ticklabels(config.ID2LABEL.values())
    ax.yaxis.set_ticklabels(config.ID2LABEL.values())


model = BERT().to(device)
load_checkpoint(config.DESTINATION_PATH+'/'+'model.pt',model)
test_iter = Dataset(config.DATA_PATH).test_iter
evaluate(model, test_iter)
