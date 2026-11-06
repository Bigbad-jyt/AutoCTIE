"""Evaluate entities and directed triples by exact matching."""
import argparse,json
from _bootstrap import local_path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gold',required=True);p.add_argument('--predictions',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();output=local_path(a.output)
    if output.exists():raise FileExistsError(output)
    from acti.evaluate import evaluate_rows,VERSION
    from acti.io_utils import read_jsonl,write_json,write_jsonl
    metrics,samples,errors=evaluate_rows(read_jsonl(local_path(a.gold)),read_jsonl(local_path(a.predictions)))
    output.mkdir(parents=True);write_json(output/'metrics.json',metrics)
    write_jsonl(output/'per_sample.jsonl',samples);write_jsonl(output/'errors.jsonl',errors)
    print(json.dumps({'evaluator':VERSION,'entity':metrics['entity']['micro'],'relation':metrics['relation']['micro']},indent=2))
if __name__=='__main__':main()
