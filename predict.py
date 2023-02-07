from transformers import TextClassificationPipeline, BertForSequenceClassification

# from utils.common_util import tokenizer
# TODO: REWRITE THE INFERENCE FUNCTION FOR MODEL

# new version
model = BertForSequenceClassification.from_pretrained(
    "model_files/trained_models/2022-12-29_15-44-25_customBERT/model.pt",
    config="model_files/trained_models/2022-12-29_12-29-41_customBERT/config.json",
)
tokenizer = tokenizer
classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer)
res = classifier("trump is the president of china")

print(res)
