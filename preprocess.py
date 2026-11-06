"""Encode data with word-aligned BIOES labels."""
import argparse,json
from _bootstrap import ROOT,local_path,settings

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='config.yaml')
    p.add_argument('--encoder',choices=['BERT','CySecBERT'],default='CySecBERT')
    p.add_argument('--tokenizer',help='Local BertTokenizerFast directory; defaults to the supplied snapshot')
    p.add_argument('--fold',type=int,choices=range(1,6),help='Omit to encode all five folds')
    a=p.parse_args()
    from acti.tokenization import load_tokenizer,encode_sample
    from acti.io_utils import read_jsonl,write_jsonl,write_json,digest
    cfg=settings(a.config);data=local_path(cfg['data']);out=local_path(cfg['encoded'])/a.encoder
    tok=load_tokenizer(str(local_path(a.tokenizer) if a.tokenizer else ROOT/'model/tokenizers'/a.encoder))
    originals={r['sample_id']:r for r in read_jsonl(data/'original.jsonl')}
    for fold in ([a.fold] if a.fold else range(1,6)):
        target=out/f'fold_{fold}'
        if target.exists():raise FileExistsError(f'Refusing to overwrite encoded data: {target}')
        split=json.loads((data/f'splits/fold_{fold}.json').read_text(encoding='utf-8'))
        prepared={part:[encode_sample(originals[sid],tok) for sid in ids] for part,ids in split['sample_ids'].items()}
        raw=data/f'augmented/fold_{fold}.jsonl';augraw=read_jsonl(raw)
        augmented=[]
        for row in augraw:
            encoded=encode_sample(row,tok)
            encoded.update({k:row[k] for k in ('software_test_only','source')})
            augmented.append(encoded)
        # Validate all records before writing.
        target.mkdir(parents=True)
        for part,records in prepared.items():write_jsonl(target/f'{part}.jsonl',records)
        frozen=target/'augmentation';frozen.mkdir()
        (frozen/'raw_pool.jsonl').write_bytes(raw.read_bytes())
        write_jsonl(frozen/f'{a.encoder}.jsonl',augmented)
        write_json(frozen/'manifest.json',{'fold':fold,'selected_samples':len(augmented),'files_sha256':{n:digest(frozen/n) for n in ['raw_pool.jsonl',f'{a.encoder}.jsonl']}})
        write_json(target/'manifest.json',{'fold':fold,'encoder':a.encoder,'counts':{**{k:len(v) for k,v in prepared.items()},'augmented':len(augmented)},'files_sha256':{x.name:digest(x) for x in target.glob('*.jsonl')}})
        print(json.dumps({'fold':fold,'output':str(target),'original':{k:len(v) for k,v in prepared.items()},'augmented':len(augmented)}))
if __name__=='__main__':main()
