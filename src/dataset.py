import config
import torch

class EntityDataset:
    """
    texts : list of lists of words. eg. [["Hi", ",", "my", "name", "is", "Shivam"],["......"], ]
    tags/pos : list of lists of tags and pos. eg. [[1, 2, 2, 6, 8], ]
    """
    def __init__(self, texts, pos, tags):
        self.texts = texts
        self.pos = pos
        self.tags = tags

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        pos = self.pos[item]
        tag = self.tags[item]

        ids = []
        target_pos = []
        target_tag = []

        for i, s in enumerate(text):
            """
            >>> text = ["Shivam", "is", "going", "to", "market"]
            >>> for t in text:
            ...     inputs = token.encode(t, add_special_tokens = False)
            ...     print(inputs)
            ...
            [12535, 2213]
            [2003]
            [2183]
            [2000]
            [3006]
            """
            inputs = config.TOKENIZER.encode(s, add_special_tokens = False)
            input_len = len(inputs)
            #we are using the extend instead of append. We want to  append the individual elemets of the list to the ids not the list itself.
            ids.extend(inputs)
            target_pos.extend([pos[i]] * input_len)
            target_tag.extend([tag[i]] * input_len)

        ids = ids[:config.MAX_LEN - 2]
        target_tag = target_tag[:config.MAX_LEN - 2]
        target_pos = target_pos[:config.MAX_LEN -2]

        #[101] is "cls" token and [102] is "sep" token
        ids = [101] + ids + [102]
        target_tag = [0] + target_tag + [0]
        target_pos = [0] + target_pos + [0]

        #mask is used to identify that which token are from the text and which tokens are padded to make them of equal length.If a token is padded then mask would be 0 otherwise 1.
        mask = [1] * len(ids)
        #token_type_ids are used to sepaarte the context from the question when we have question-answering task at hand. 0 is used for context and 1 is used to question.
        token_type_ids = [0] * len(ids)

        padding_len = config.MAX_LEN - len(ids)

        ids = ids + ([0] * padding_len)
        target_tag = target_tag + ([0] * padding_len)
        target_pos = target_pos + ([0] * padding_len)
        mask = mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)

        return {
        "ids" : torch.tensor(ids, dtype=torch.long),
        "target_tag" : torch.tensor(target_tag, dtype=torch.long),
        "target_pos" : torch.tensor(target_pos, dtype=torch.long),
        "mask" : torch.tensor(mask, dtype=torch.long),
        "token_type_ids" : torch.tensor(token_type_ids, dtype=torch.long)
        }
