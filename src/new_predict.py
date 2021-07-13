import config
import joblib
import numpy as np
import dataset
import torch
from ordered_set import OrderedSet
from model import EntityModel


d = {"Geographical Entity"       : "B-geo",
     "Organization"              : "B-org",
     "Person"                    : "B-per",
     "Geopolitical Entity"       : "B-gpe",
     "Time indicator"            : "B-tim",
     "Artifact"                  : "B-art",
     "Event"                     : "B-eve",
     "Natural Phenomenon"        : "B-nat",
     "Inter Geographical Entity" : "I-geo",
     "Inter Organization"        : "I-org",
     "Inter Person"              : "I-per",
     "Inter Geopolitical"        : "I-gpe",
     "Inter Time indicator"      : "I-tim",
     "Inter Artifact"            : "I-art",
     "Inter Event"               : "I-eve",
     "None"                      : "O"
    }

class NerTagger:
    def __init__(self, texts, classes):
        self.sentences = texts
        self.classes = classes
        metadata = joblib.load("meta.bin")
        self.enc_pos = metadata["enc_pos"]
        self.enc_tag = metadata["enc_tag"]
        self.num_pos = len(list(self.enc_pos.classes_))
        self.num_tag = len(list(self.enc_tag.classes_))
        self.model = EntityModel(self.num_pos, self.num_tag)
        self.model.load_state_dict(torch.load(config.MODEL_PATH, map_location=torch.device("cpu")))

    def tag(self, upto = 1):

        sentence_tag_list = []
        return_tags = []
        for sentence in self.sentences:
            tokenized_sentence = config.TOKENIZER.encode(sentence)
            len_tokens = len(tokenized_sentence)
            sentence = sentence.split()
            test_dataset = dataset.EntityDataset(texts = [sentence],
                                                pos = [[0] * len(sentence)],
                                                tags  = [[0] * len(sentence)] )
            with torch.no_grad():
                data = test_dataset[0]
                for k, v in data.items():
                    data[k] = v.unsqueeze(0)
                tag, pos, _ = self.model(**data)

            sentence_tag_list.append(tag[0][1 : len_tokens - 1])
        # print(len(sentence_tag_list[0]))

        for i, sentence_tags in enumerate(sentence_tag_list):
            word_tag_mapping = {}
            tokenized_sentence = config.TOKENIZER.encode(self.sentences[i])
            len_tokens = len(tokenized_sentence)

            for id, word_tags in zip(tokenized_sentence[1 : len_tokens - 1], sentence_tags):
                word = config.TOKENIZER.decode([id])
                tags_index = word_tags.argsort(descending = True).numpy()
                tags = self.enc_tag.inverse_transform(tags_index)
                # print(word, ":", tags[:3])
                # for tag in self.classes:
                #     if d[tag] in tags[0]:
                #         word_tag_mapping[word] = tag
                if all(x in self.classes for x in tags[:upto]):
                    word_tag_mapping[word] = tags[:upto]
                else:
                    list_tag = list(d[tag] for tag in self.classes)
                    both = OrderedSet(tags[:upto]).intersection(list_tag)
                    indices_classes = [list_tag.index(x) for x in both]
                    lst = list(self.classes[x] for x in indices_classes)
                    if not (len(lst) == 0 or (len(lst) == 1 and lst[0] == "None")):
                        word_tag_mapping[word] = lst



            return_tags.append(word_tag_mapping)
        return return_tags


if __name__ == '__main__':
    texts = ["Google is based in New York", "British troops are marching", "Bill Gates is the owner of Microsoft", "I will move to New Delhi on 2008-06-29", "Australian Open is held every 4 years"]
    classes = ["Geographical Entity", "Time indicator", "Person", "Geopolitical Entity", "Inter Person", "Organization", "Inter Geographical Entity", "None"]
    tagger = NerTagger(texts, classes)
    tags = tagger.tag()
    print(tags)


