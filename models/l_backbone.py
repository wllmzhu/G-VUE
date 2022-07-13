import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel


class RoBERTa(nn.Module):
    def __init__(self, cache_dir=None):
        super().__init__()
        if cache_dir is None:
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            self.model = RobertaModel.from_pretrained('roberta-base')
        else:
            self.tokenizer = RobertaTokenizer.from_pretrained(cache_dir)
            self.model = RobertaModel.from_pretrained(cache_dir)

    def forward(self, sentences, device=None):
        token_inputs = self.tokenizer(
            sentences,
            padding=True,
            return_tensors='pt')
        
        if device is None:
            token_inputs = {k: v.cuda() for k, v in token_inputs.items()}
        else:
            token_inputs = {k: v.cuda(device) for k, v in token_inputs.items()}
        
        outputs = self.model(**token_inputs)
        return outputs[0], token_inputs


class RoBERTa_R2R(nn.Module):
    "Different settings compared to the general RoBERTa above. Skip tokenization, output pooler_output and sentence embedding"
    def __init__(self, cache_dir=None):
        super().__init__()
        if cache_dir is None:
            self.model = RobertaModel.from_pretrained('roberta-base')
        else:
            self.model = RobertaModel.from_pretrained(cache_dir)
        
    def forward(self, token_inputs, position_ids=None, token_type_ids=None, device=None, ):
        if device is None:
            token_inputs = token_inputs.cuda()
        else:
            token_inputs = token_inputs.cuda(device)
        
        outputs = self.model(token_inputs, output_hidden_states=True, position_ids=position_ids, token_type_ids=token_type_ids) 
        
        # return pooler_output, sentence embedding(first element of hidden_states)
        return outputs[1], outputs[2][0]


if __name__=='__main__':
    roberta = RoBERTa_R2R().to('cuda:0')
    # seq_pair = roberta.tokenizer.encode("__bbox_begin__ pos_1 pos_10 pos_20 pos_30 __bbox_end__")
    # print(roberta.tokenizer.decode(seq_pair))
    # print(roberta(['How do you do?','I am fine thank you.']))
    print(roberta([
        'locate a blue turtle-like pokemon with round head with box',
        'locate "a blue turtle-like pokemon with round head" with box',
        'locate " a blue turtle-like pokemon with round head " with box'
    ], device='cuda:0'))
