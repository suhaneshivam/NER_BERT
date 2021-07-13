import config
import torch
import transformers
import torch.nn as nn

def loss_fn(output, target, mask, num_labels):
    """
    torch.where(condition, x, y) → Tensor
    The operation is defined as :
    out_i = x_i if condition else y_i
    >>> lfn.ingore happens to be -100. It would replace False with -100 and True with the correspondig Target. lfn would simply ignore where it finds -100.
    """
    lfn = nn.CrossEntropyLoss()
    #.view() creates a copy of the tensor(shallow copy) and shape is specified as argument. In this particular case, it would also compare the elements of mask and make a list of True and False according to the condition.
    active_loss = mask.view(-1) == 1
    active_logits = output.view(-1, num_labels)

    active_labels = torch.where(active_loss, target.view(-1), torch.tensor(lfn.ignore_index).type_as(target))

    loss = lfn(active_logits, active_labels)
    return loss

class EntityModel(nn.Module):
    def __init__(self, num_pos, num_tag):
        super(EntityModel, self).__init__()
        self.num_tag = num_tag
        self.num_pos = num_pos
        self.bert = transformers.BertModel.from_pretrained("bert-base-uncased", return_dict = False)
        self.bert_drop1 = nn.Dropout(0.3)
        self.bert_drop2 = nn.Dropout(0.3)
        self.out_tag = nn.Linear(768, self.num_tag)
        self.out_pos = nn.Linear(768, self.num_pos)

    def forward(self, ids, target_tag, target_pos, mask, token_type_ids):
        #size of o1 is (number of tokens in each text X 768)

        o1, _ = self.bert(ids, attention_mask = mask, token_type_ids = token_type_ids) #128 X 768

        bo_tag = self.bert_drop1(o1) #batch_size X 128 X 768
        bo_pos = self.bert_drop2(o1) #batch_size X 128 X 768

        #Each of the token would have its sepaarte output which is of dimension num_tag and num_pos for tag and pos respectively; where num_pos is total number of possible pos classes and num_tag is total number of possible tag classes.
        tag = self.out_tag(bo_tag) #128 X 768 X 768 X num_tag = 128 X num_tag =batch_size X  128 X 17
        pos = self.out_pos(bo_pos) #128 X 768 X 768 X num_pos = 128 X num_pos =batch_size X  128 X 42

        loss_tag = loss_fn(tag, target_tag, mask, self.num_tag)
        loss_pos = loss_fn(pos, target_pos, mask, self.num_pos)

        loss = (loss_tag + loss_pos) / 2

        return tag, pos, loss
