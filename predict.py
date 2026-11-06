"""Extract entities and directed relations from a checkpoint."""
import argparse
from types import SimpleNamespace
from _bootstrap import local_path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--input',required=True,help='JSONL with sample_id, report_id, char_start and text')
    p.add_argument('--tokenizer',required=True,help='Local tokenizer matching the training encoder')
    p.add_argument('--output',required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--batch-size',type=int,default=2)
    a=p.parse_args()
    if a.batch_size<1:p.error('--batch-size must be positive')
    output=local_path(a.output)
    if output.exists():raise FileExistsError(output)
    from acti.models.joint_model import JointModel
    from acti.tokenization import load_tokenizer,encode_sample
    from acti.train import predict_records
    from acti.io_utils import read_jsonl,write_jsonl
    tokenizer=load_tokenizer(str(local_path(a.tokenizer)));rows=[]
    for r in read_jsonl(local_path(a.input)):
        clean={k:r[k] for k in ['sample_id','report_id','char_start','text']}
        clean.update(entities=[],relations=[])
        rows.append(encode_sample(clean,tokenizer))
    model=JointModel.load(local_path(a.checkpoint),a.device);model.eval()
    predictions=predict_records(model,rows,SimpleNamespace(device=a.device,evaluation_batch=a.batch_size))
    output.parent.mkdir(parents=True,exist_ok=True);write_jsonl(output,predictions)
    print(f'Wrote {len(predictions)} predictions to {output}')
if __name__=='__main__':main()
