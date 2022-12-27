from torch import nn
from transformers import BertForSequenceClassification,BertConfig
import config




class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        pretrained_model = config.BERT_PATH
        self.config = BertConfig(label2id=config.LABEL2ID,id2label=config.ID2LABEL)
        self.encoder = BertForSequenceClassification.from_pretrained(pretrained_model,config=self.config)

    def forward(self, text, label):
        loss, text_fea = self.encoder(text, labels=label)[:2]
        return loss, text_fea
