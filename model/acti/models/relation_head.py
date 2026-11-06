"""Mean entity pooling and ordered-pair relation MLP."""
import torch
from torch import nn
from acti.relation_samples import RELATION_LABELS


class RelationHead(nn.Module):
    def __init__(self,hidden_size,mlp_hidden=256,dropout=0.1):
        super().__init__();self.output_size=len(RELATION_LABELS)
        self.mlp=nn.Sequential(nn.Linear(4*hidden_size,mlp_hidden),nn.ReLU(),nn.Dropout(dropout),
                               nn.Linear(mlp_hidden,self.output_size))

    def forward(self,hidden,entities,pairs):
        representations={}
        for e in entities:
            a,b=e['word_start'],e['word_end']
            if not 0<=a<b<=len(hidden):raise ValueError('Invalid entity word span')
            representations[e['id']]=hidden[a:b].mean(dim=0)
        if not pairs:return hidden.new_empty((0,self.output_size))
        entity_ids=list(representations)
        index={eid:i for i,eid in enumerate(entity_ids)}
        head_indices=[];tail_indices=[]
        for pair in pairs:
            h,t=pair['head'],pair['tail']
            if h==t or h not in representations or t not in representations:raise ValueError('Invalid pair')
            head_indices.append(index[h]);tail_indices.append(index[t])
        pooled=torch.stack([representations[eid] for eid in entity_ids])
        head=pooled[torch.tensor(head_indices,device=hidden.device)]
        tail=pooled[torch.tensor(tail_indices,device=hidden.device)]
        return self.mlp(torch.cat([head,tail,head*tail,head-tail],dim=-1))
