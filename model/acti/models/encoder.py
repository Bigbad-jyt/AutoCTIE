"""Transformer, first-WordPiece pooling, BiLSTM and self-attention."""
import math
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from transformers import BertModel


def load_pretrained_encoder(path):
    model,info=BertModel.from_pretrained(path,local_files_only=True,use_safetensors=True,
        add_pooling_layer=False,output_loading_info=True)
    if info['missing_keys'] or info.get('mismatched_keys') or info.get('error_msgs'):
        raise ValueError(f'Incomplete pretrained encoder: {info}')
    if any(not x.startswith(('cls.','pooler.')) for x in info['unexpected_keys']):
        raise ValueError(f'Unexpected non-MLM encoder weights: {info}')
    return model,info


class SharedEncoder(nn.Module):
    def __init__(self,transformer,lstm_hidden=256,lstm_layers=1,dropout=0.1):
        super().__init__();self.transformer=transformer;self.dropout=nn.Dropout(dropout)
        self.lstm=nn.LSTM(transformer.config.hidden_size,lstm_hidden,num_layers=lstm_layers,
            bidirectional=True,batch_first=True,dropout=dropout if lstm_layers>1 else 0)
        self.output_size=2*lstm_hidden
        self.query=nn.Linear(self.output_size,self.output_size)
        self.key=nn.Linear(self.output_size,self.output_size)
        self.value=nn.Linear(self.output_size,self.output_size)
        self.attention_dropout=nn.Dropout(dropout)
        self.post_attention_norm=nn.LayerNorm(self.output_size)

    def forward(self,batch):
        pieces=self.transformer(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],
            token_type_ids=batch['token_type_ids']).last_hidden_state
        indices=batch['first_wordpiece_indices'][:,:,None].expand(-1,-1,pieces.shape[-1])
        words=pieces.gather(1,indices)
        mask=batch['word_mask'];lengths=mask.long().sum(1)
        packed=pack_padded_sequence(self.dropout(words),lengths.cpu(),batch_first=True,enforce_sorted=False)
        packed,_=self.lstm(packed)
        words,_=pad_packed_sequence(packed,batch_first=True,total_length=mask.shape[1])
        q,k,v=self.query(words),self.key(words),self.value(words)
        scores=q@k.transpose(1,2)/math.sqrt(self.output_size)
        weights=scores.masked_fill(~mask[:,None,:],float('-inf')).softmax(dim=-1)
        context=self.attention_dropout(weights)@v
        # Mask padding after residual addition and layer normalization.
        output=self.post_attention_norm(words+self.dropout(context))
        return output*mask[:,:,None],weights*mask[:,:,None]
