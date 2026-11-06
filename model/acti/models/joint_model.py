"""Joint CRF entity tagging and ordered-pair relation classification."""
from dataclasses import asdict,dataclass
from pathlib import Path
import json
import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertConfig,BertModel
from acti.io_utils import digest
from acti.tokenization import LABELS,decode_bioes
from acti.relation_samples import RELATION_LABELS,directed_pairs,decode_relation_logits
from acti.models.encoder import SharedEncoder,load_pretrained_encoder
from acti.models.entity_head import EntityHead
from acti.models.relation_head import RelationHead


@dataclass
class JointConfig:
    lstm_hidden:int=256
    lstm_layers:int=1
    relation_hidden:int=256
    dropout:float=0.1
    lambda_relation:float=1.0


class JointModel(nn.Module):
    def __init__(self,transformer,config=None):
        super().__init__();self.config=config or JointConfig();self.pretrained_identity=None;self.head_initialization_seed=None
        if self.config.lambda_relation<0:raise ValueError('Negative loss weight')
        self.shared=SharedEncoder(transformer,self.config.lstm_hidden,self.config.lstm_layers,self.config.dropout)
        self.entity=EntityHead(self.shared.output_size)
        self.relation=RelationHead(self.shared.output_size,self.config.relation_hidden,self.config.dropout)

    @classmethod
    def from_pretrained_encoder(cls,path,config=None,head_seed=43):
        encoder,info=load_pretrained_encoder(path)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(head_seed)
            model=cls(encoder,config)
        model.head_initialization_seed=head_seed
        model.pretrained_identity={'path':str(Path(path).resolve()),'weights_sha256':digest(Path(path)/'model.safetensors'),'config_sha256':digest(Path(path)/'config.json')}
        return model,info

    def forward(self,batch,compute_loss=False):
        hidden,attention=self.shared(batch);emissions=self.entity(hidden)
        result={'emissions':emissions,'hidden':hidden,'attention':attention}
        if compute_loss:
            if batch['training_pairs'] is None or 'label_ids' not in batch:
                raise ValueError('Loss is training-only; use explicit gold training batch')
            entity_loss=self.entity.crf.neg_log_likelihood(emissions,batch['label_ids'],batch['word_mask'])
            logits=[];targets=[]
            for i,(record,pairs) in enumerate(zip(batch['records'],batch['training_pairs'])):
                scores=self.relation(hidden[i,:len(record['words'])],record['entities'],pairs)
                if pairs:
                    logits.append(scores);targets.extend(p['target'] for p in pairs)
            if logits:
                relation_logits=torch.cat(logits)
                relation_targets=torch.tensor(targets,device=hidden.device,dtype=relation_logits.dtype)
                relation_loss=F.binary_cross_entropy_with_logits(relation_logits,relation_targets,reduction='none').mean(dim=1).mean()
            else:
                relation_logits=hidden.new_empty((0,len(RELATION_LABELS)))
                relation_loss=hidden.sum()*0
            result.update(entity_loss=entity_loss,relation_loss=relation_loss,relation_logits=relation_logits,
                loss=entity_loss+self.config.lambda_relation*relation_loss)
        return result

    @torch.no_grad()
    def predict(self,batch):
        if self.training:raise ValueError('Call eval() before deterministic inference')
        output=self(batch);paths=self.entity.crf.decode(output['emissions'],batch['word_mask']);predictions=[]
        for i,(record,path) in enumerate(zip(batch['records'],paths)):
            words=record['words'];entities=[]
            for j,(t,a,b) in enumerate(decode_bioes([LABELS[k] for k in path],words)):
                ws=next(k for k,w in enumerate(words) if w['start']==a)
                we=next(k+1 for k,w in enumerate(words) if w['end']==b)
                entities.append({'id':f'P{j+1}','type':t,'start':a,'end':b,'text':record['text'][a:b],
                    'word_start':ws,'word_end':we})
            pairs=[{'head':a,'tail':b} for a,b in directed_pairs(entities)]
            scores=self.relation(output['hidden'][i,:len(words)],entities,pairs)
            relations=decode_relation_logits(pairs,scores)
            predictions.append({'sample_id':record['sample_id'],'report_id':record['report_id'],
                'char_start':record['char_start'],'entities':entities,'relations':relations})
        return predictions

    def parameter_report(self):
        groups={}
        for name,module in [('transformer',self.shared.transformer),('shared_word_layers',self.shared.lstm),
                            ('attention',nn.ModuleList([self.shared.query,self.shared.key,self.shared.value,self.shared.post_attention_norm])),
                            ('entity',self.entity),('relation',self.relation)]:
            groups[name]={'parameters':sum(p.numel() for p in module.parameters()),
                         'trainable':sum(p.numel() for p in module.parameters() if p.requires_grad)}
        return {'total':sum(p.numel() for p in self.parameters()),
                'trainable':sum(p.numel() for p in self.parameters() if p.requires_grad),
                'frozen_parameters':[n for n,p in self.named_parameters() if not p.requires_grad],
                'groups':groups,'config':asdict(self.config),'pretrained_identity':self.pretrained_identity,'head_initialization_seed':self.head_initialization_seed}

    def save(self,path):
        path=Path(path)
        if path.exists():raise FileExistsError(path)
        path.mkdir(parents=True)
        config={'format_version':2,'architecture':'residual_postnorm_ten_multilabel','relation_threshold':0.5,'joint':asdict(self.config),
            'transformer':self.shared.transformer.config.to_dict(),'pretrained_identity':self.pretrained_identity,'head_initialization_seed':self.head_initialization_seed,
            'entity_labels':list(LABELS),'relation_labels':list(RELATION_LABELS)}
        (path/'config.json').write_text(json.dumps(config,indent=2,sort_keys=True)+'\n',encoding='utf-8')
        torch.save(self.state_dict(),path/'model.pt')

    @classmethod
    def load(cls,path,map_location='cpu'):
        path=Path(path);config=json.loads((path/'config.json').read_text(encoding='utf-8'))
        if config.get('format_version')!=2:raise ValueError('Unsupported checkpoint format; legacy eleven-class checkpoints cannot load')
        if config['entity_labels']!=list(LABELS) or config['relation_labels']!=list(RELATION_LABELS):
            raise ValueError('Checkpoint label vocabulary mismatch')
        model=cls(BertModel(BertConfig.from_dict(config['transformer']),add_pooling_layer=False),JointConfig(**config['joint']))
        model.pretrained_identity=config.get('pretrained_identity')
        model.head_initialization_seed=config.get('head_initialization_seed')
        model.load_state_dict(torch.load(path/'model.pt',map_location=map_location,weights_only=True),strict=True)
        return model.to(map_location)
