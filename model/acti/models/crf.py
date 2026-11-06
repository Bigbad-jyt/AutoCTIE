"""Linear-chain CRF with BIOES transition constraints."""
import torch
from torch import nn
from acti.tokenization import LABELS


class BIOESCRF(nn.Module):
    def __init__(self, labels=LABELS):
        super().__init__()
        self.labels=tuple(labels); n=len(labels)
        self.start=nn.Parameter(torch.zeros(n));self.end=nn.Parameter(torch.zeros(n))
        self.transitions=nn.Parameter(torch.zeros(n,n))  # previous -> next
        def split(label):return ('O',None) if label=='O' else tuple(label.split('-',1))
        kinds=[split(x) for x in labels]
        start=torch.tensor([p in ('O','B','S') for p,t in kinds])
        end=torch.tensor([p in ('O','E','S') for p,t in kinds])
        valid=torch.tensor([[(q in ('O','B','S') if p in ('O','E','S') else q in ('I','E') and t==u)
                            for q,u in kinds] for p,t in kinds])
        self.register_buffer('valid_start',start);self.register_buffer('valid_end',end)
        self.register_buffer('valid_transitions',valid)

    def _check(self, emissions, mask, tags=None):
        if emissions.ndim!=3 or emissions.shape[-1]!=len(self.labels) or mask.shape!=emissions.shape[:2]:
            raise ValueError('CRF shape mismatch')
        if mask.dtype!=torch.bool or mask.shape[1]==0 or not mask[:,0].all() or (mask[:,1:]&~mask[:,:-1]).any():
            raise ValueError('CRF requires nonempty contiguous prefix mask')
        if tags is not None:
            if tags.shape!=mask.shape or ((tags[mask]<0)|(tags[mask]>=len(self.labels))).any():
                raise ValueError('Invalid CRF labels')
            for row,m in zip(tags,mask):
                seq=row[m]
                if not self.valid_start[seq[0]] or not self.valid_end[seq[-1]] or (len(seq)>1 and not self.valid_transitions[seq[:-1],seq[1:]].all()):
                    raise ValueError('Illegal gold BIOES path')

    def _scores(self):
        return (self.start.masked_fill(~self.valid_start,float('-inf')),
                self.end.masked_fill(~self.valid_end,float('-inf')),
                self.transitions.masked_fill(~self.valid_transitions,float('-inf')))

    def log_partition(self, emissions, mask):
        self._check(emissions,mask)
        start,end,trans=self._scores();alpha=emissions[:,0]+start
        for i in range(1,emissions.shape[1]):
            nxt=torch.logsumexp(alpha[:,:,None]+trans[None,:,:],dim=1)+emissions[:,i]
            alpha=torch.where(mask[:,i,None],nxt,alpha)
        return torch.logsumexp(alpha+end,dim=-1)

    def neg_log_likelihood(self, emissions, tags, mask):
        self._check(emissions,mask,tags)
        # Replace padding labels before indexing.
        tags=tags.masked_fill(~mask,0)
        start,end,trans=self._scores()
        batch=torch.arange(len(tags),device=tags.device)
        gold=start[tags[:,0]]+emissions[batch,0,tags[:,0]]
        for i in range(1,tags.shape[1]):
            step=trans[tags[:,i-1],tags[:,i]]+emissions[batch,i,tags[:,i]]
            gold=gold+torch.where(mask[:,i],step,torch.zeros_like(step))
        last=tags.gather(1,(mask.long().sum(1)-1)[:,None]).squeeze(1)
        gold=gold+end[last]
        return (self.log_partition(emissions,mask)-gold).mean()

    @torch.no_grad()
    def decode(self, emissions, mask):
        self._check(emissions,mask)
        start,end,trans=self._scores();paths=[]
        for row,m in zip(emissions,mask):
            length=int(m.sum());score=row[0]+start;history=[]
            for i in range(1,length):
                best,back=(score[:,None]+trans).max(dim=0)
                score=best+row[i];history.append(back)
            last=int((score+end).argmax());path=[last]
            for back in reversed(history):last=int(back[last]);path.append(last)
            paths.append(list(reversed(path)))
        return paths
