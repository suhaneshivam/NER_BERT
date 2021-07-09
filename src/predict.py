
import numpy as np

import torch
import joblib

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import config
import dataset
import engine
from model import EntityModel


if __name__ == '__main__':

    metadata = joblib.load("meta.bin")
    enc_pos = metadata["enc_pos"]
    enc_tag = metadata["enc_tag"]


    num_pos = len(list(enc_pos.classes_))
    num_tag = len(list(enc_tag.classes_))

    sentence = """
    shivam is going to the market to buy vegetabel
    """
    tokenized_sentence = config.TOKENIZER.encode(sentence)

    sentence = sentence.split()
    print(sentence)

    test_dataset = dataset.EntityDataset(texts = [sentence], pos = [[0] * len(sentence)], tags = [[0] * len(sentence)] )

    model = EntityModel(num_pos = num_pos, num_tag = num_tag)
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location= torch.device("cpu")))


    with torch.no_grad():
        data = test_dataset[0]
        for k, v in data.items():
            data[k] = v.unsqueeze(0)
        
        tag, pos, _ = model(**data)

        print(enc_tag.inverse_transform(tag.argmax(2).cpu().numpy().reshape(-1)[:len(tokenized_sentence)]))
        print(enc_pos.inverse_transform(pos.argmax(2).cpu().numpy().reshape(-1)[:len(tokenized_sentence)]))
