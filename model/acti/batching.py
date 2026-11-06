"""Pad batches and sample training pairs."""
import torch
from acti.relation_samples import training_pairs


def collate_records(records,pad_token_id=0,training=False,seed=43,epoch=0):
    if not records:raise ValueError('Empty batch')
    pieces=max(len(r['input_ids']) for r in records);words=max(len(r['words']) for r in records)
    if words==0:raise ValueError('Empty word sequence')
    def padded(key,length,fill=0):
        return torch.tensor([list(r[key])+[fill]*(length-len(r[key])) for r in records],dtype=torch.long)
    batch={'input_ids':padded('input_ids',pieces,pad_token_id),
        'attention_mask':padded('attention_mask',pieces),
        'token_type_ids':padded('token_type_ids',pieces),
        'first_wordpiece_indices':padded('first_wordpiece_indices',words),
        'word_mask':torch.tensor([[True]*len(r['words'])+[False]*(words-len(r['words'])) for r in records]),
        'records':records,'training_pairs':None}
    if training:
        if any('label_ids' not in r or 'entities' not in r or 'relations' not in r for r in records):
            raise ValueError('Gold missing from training batch')
        batch['label_ids']=padded('label_ids',words)
        batch['training_pairs']=[training_pairs(r,seed=seed,epoch=epoch) for r in records]
    return batch


def to_device(batch,device):
    return {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
