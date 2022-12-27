from transformers import TextClassificationPipeline,BertForSequenceClassification
import config
from utils.helper_function import tokenizer
model=BertForSequenceClassification.from_pretrained(config.DESTINATION_PATH+'/'+'model.pt',config=config.DESTINATION_PATH+'/'+'config.json')
tokenizer=tokenizer

classifier=TextClassificationPipeline(model=model,tokenizer=tokenizer)
res=classifier("who is this")


print(res)
