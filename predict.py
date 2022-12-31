from transformers import TextClassificationPipeline,BertForSequenceClassification
import CONFIG
from utils.helper_function import tokenizer

# previous version
# model=BertForSequenceClassification.from_pretrained(myconfig.DESTINATION_PATH+'/'+'model.pt',config=config.DESTINATION_PATH+'/'+'config.json')
# tokenizer=tokenizer
#
# classifier=TextClassificationPipeline(model=model,tokenizer=tokenizer)
# res=classifier("who is this")

# new version
model=BertForSequenceClassification.from_pretrained('model_files/trained_models/2022-12-29_15-44-25_customBERT/model.pt',config='model_files/trained_models/2022-12-29_12-29-41_customBERT/config.json')
tokenizer=tokenizer
classifier=TextClassificationPipeline(model=model,tokenizer=tokenizer)
res=classifier("trump is the president of china")

print(res)
