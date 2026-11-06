"""Train or validate one AutoCTIE configuration."""
import argparse,json
from _bootstrap import ROOT,local_path,settings

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='config.yaml')
    p.add_argument('--condition',choices=['C1','C2','C3','C4'],default='C4')
    p.add_argument('--fold',type=int,choices=range(1,6),default=1)
    p.add_argument('--encoder-path',help='Local pretrained BertModel directory (config.json + model.safetensors)')
    p.add_argument('--device',default='cuda')
    p.add_argument('--output',help='New output directory; relative paths use the repository root')
    p.add_argument('--dry-run',action='store_true',help='Validate data and configuration; do not load weights or train')
    a=p.parse_args();cfg=settings(a.config);condition=cfg['conditions'][a.condition]
    from acti.train import TrainConfig,load_training_inputs,run_training
    from acti.io_utils import digest
    encoder=condition['encoder'];folder=local_path(cfg['encoded'])/encoder/f'fold_{a.fold}'
    if not folder.exists():p.error('Encoded data missing. Run preprocess.py for this encoder and fold first.')
    frozen=folder/'augmentation';augment=frozen/f'{encoder}.jsonl' if condition['augmentation'] else None
    name=f'fold{a.fold}_{a.condition}'
    config=TrainConfig(run_id=name,configuration=a.condition,encoder_name=encoder,
        encoder_path=str(local_path(a.encoder_path)) if a.encoder_path else '',
        train_path=str(folder/'train.jsonl'),validation_path=str(folder/'validation.jsonl'),test_path=str(folder/'test.jsonl'),
        augmentation_path=str(augment) if augment else None,
        augmentation_source_sha256=digest(frozen/'raw_pool.jsonl') if augment else None,
        fold=a.fold,seed=42+a.fold,run_kind='formal',device=a.device,joint=cfg['joint'],**cfg['training'])
    original,validation,training=load_training_inputs(config)
    if a.dry_run:
        print(json.dumps({'training_started':False,'condition':a.condition,'fold':a.fold,'original':len(original),'validation':len(validation),'total_training':len(training)}));return
    if not a.encoder_path:p.error('--encoder-path is required for training')
    for n in ['config.json','model.safetensors']:
        if not (local_path(a.encoder_path)/n).is_file():p.error(f'Pretrained encoder file missing: {n}')
    output=local_path(a.output) if a.output else ROOT/'outputs'/name
    print(json.dumps(run_training(config,output,ROOT/'outputs/registry.jsonl'),indent=2))
if __name__=='__main__':main()
