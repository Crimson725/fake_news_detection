from transformers import BertTokenizer
from transformers import BertConfig
from transformers import logging as hf_logging

from models.layers import TF_BERT
from utils.common_util import load_checkpoint, get_inf_parser
import os
import pickle
import CONFIG
from utils.kg_util import KG_embedding
import torch

# ignore warning
hf_logging.set_verbosity_error()

# get tokenizer
tokenizer = BertTokenizer.from_pretrained(CONFIG.BERT_BASE_PATH)

# TODO ADD REBEL MODEL FOR RAW DATA


# kg generator
kg_generator = KG_embedding()


# get test data
test_text = "When so many actors seem content to churn out performances for a quick paycheck, a performer who adheres to his principles really stands out. Thats why Jeff Bridges made waves this week when he announced that from now on, he will only perform nude scenes. In an interview in this months GQ, the Big Lebowski star made it clear that he was more than ready to move on to a new phase in his career, leaving his clothed roles in the past. Ive been there and Ive done that, said Bridges, rattling off a laundry list of the films hes appeared in covered up. Now, I can finally afford to only take on roles that excite me. Right now, those are roles with nude scenes. Why waste my time with anything else? Powerful. Though he made it clear that he doesnt regret his previous non-nude roles, Jeff admitted that hed always struggled with pressure from directors and studios to stay clothed on camera. No more towels; no more bathrobes; no more carefully placed plants, he added. Even if my character isnt written as nude, any director I work with will have to figure out how to make him that way. Itll be a challenge for both of us, and one I cant wait to tackle. For their part, Jeffs fans have been nothing but supportive. Wow! Whether or not you agree with him, youve got to have respect for a Hollywood star with that much conviction. You keep doing you, Jeff! "

test_triplet = [
    ["fn:performers", "/r/dbpedia/occupation", "/c/en/performances"],
    ["/c/en/performances", "/r/CreatedBy", "fn:performers"],
    ["/c/en/fridges", "/r/NotCapableOf", "/c/en/descendest"],
    ["/c/en/descendest", "/r/LocatedNear", "/c/en/fridges"],
    ["/c/en/borawski", "/r/InstanceOf", "/c/en/films"],
    ["/c/en/been_there/v", "/r/LocatedNear", "/c/en/ridges"],
    ["/c/en/descendest", "/r/Causes", "/c/en/roles"],
    ["at:powerfull", "/r/NotDesires", "at:powerfull"],
    ["/c/en/nonnude", "/r/MotivatedByGoal", "/c/en/aude"],
    ["/c/en/towels", "/r/DerivedFrom", "/c/en/bathrobes"],
    ["/c/en/bathrobes", "/r/DerivedFrom", "/c/en/towels"],
    ["/c/en/aude", "/r/DerivedFrom", "fn:direction"],
    ["fn:direction", "/r/DerivedFrom", "/c/en/aude"],
]

hrt_score_list = kg_generator.get_triplet_score(test_triplet)
hrt_score_list = torch.tensor(hrt_score_list, dtype=torch.float)

hrt_embedding_list = kg_generator.generate_hrt_embedding(test_triplet)
hrt_embedding_list = torch.cat(hrt_embedding_list, dim=0)

inputs = tokenizer.encode_plus(
    test_text,
    None,
    truncation=True,
    add_special_tokens=True,
    max_length=512,
    padding="max_length",
    return_token_type_ids=True,
)
test_data = {
    "ids": torch.tensor(inputs["input_ids"], dtype=torch.long).unsqueeze(0),
    "mask": torch.tensor(inputs["attention_mask"], dtype=torch.long).unsqueeze(0),
    "token_type_ids": torch.tensor(
        inputs["token_type_ids"], dtype=torch.long
    ).unsqueeze(0),
    "hrt_score_list": hrt_score_list.unsqueeze(0),
    "hrt_embedding_list": hrt_embedding_list,
}


def inference(params, data):
    train_args_path = os.path.join(os.path.dirname(params.model_path), "train_args.pkl")
    with open(train_args_path, "rb") as f:
        train_args = pickle.load(f)
    config = BertConfig(label2id=CONFIG.LABEL2ID, id2label=CONFIG.ID2LABEL)
    device = torch.device("cuda:{}".format(params.cuda))
    # use train_args as params to initialize the model
    model = TF_BERT(config, train_args).to(device)
    load_checkpoint(params.model_path)
    return model(
        ids=data["ids"].to(device, dtype=torch.long),
        mask=data["mask"].to(device, dtype=torch.long),
        token_type_ids=data["token_type_ids"].to(device, dtype=torch.long),
        hrt_score_list=data["hrt_score_list"].to(device, dtype=torch.float),
        hrt_embedding_list=data["hrt_embedding_list"].to(device, dtype=torch.float),
    )


if __name__ == "__main__":
    params = get_inf_parser()
    # get dic path
    res = inference(params, test_data)
    print(res)
