"""Five-fold report splitting with duplicate-group isolation."""
from collections import Counter, defaultdict
import math
from pathlib import Path
import random
import unicodedata
from .schema import ENTITY_TYPES, RELATION_TYPES
from .io_utils import read_jsonl, write_json, write_jsonl, digest

LABELS = tuple('entity:'+x for x in ENTITY_TYPES)+tuple('relation:'+x for x in RELATION_TYPES)


def normalized_text(text):
    return ' '.join(unicodedata.normalize('NFKC',text).casefold().split())


def duplicate_components(reports, samples):
    parent={r['report_id']:r['report_id'] for r in reports}
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b: parent[max(a,b)]=min(a,b)
    buckets=defaultdict(list)
    for s in samples: buckets[normalized_text(s['text'])].append(s)
    for rows in buckets.values():
        ids=sorted({s['report_id'] for s in rows})
        for rid in ids[1:]: union(ids[0],rid)
    groups=defaultdict(list)
    for rid in sorted(parent): groups[find(rid)].append(rid)
    return list(groups.values()), buckets


def assign_groups(groups, vectors, proportions, seed):
    """Greedily balance rare labels and report counts across groups.

    Use inverse label-frequency weights, report-count weight 10 and seeded ties.
    """
    rng=random.Random(seed)
    groups=[tuple(sorted(g)) for g in groups]
    total=Counter()
    gv={}
    for g in groups:
        v=Counter()
        for rid in g: v.update(vectors[rid])
        gv[g]=v;total.update(v)
    n=sum(map(len,groups));tie={g:rng.random() for g in sorted(groups)}
    groups.sort(key=lambda g:(-sum(v/total[k] for k,v in gv[g].items()),-len(g),tie[g]))
    out=[[] for _ in proportions];loads=[Counter() for _ in proportions];sizes=[0]*len(proportions)
    for g in groups:
        def cost(i):
            p=proportions[i]
            score=sum(((loads[i][k]+v)**2-loads[i][k]**2)/(max(total[k],1)*p)
                      for k,v in gv[g].items())
            return score+10*((sizes[i]+len(g))**2-sizes[i]**2)/(n*p)
        best=min(range(len(proportions)),key=lambda i:(cost(i),sizes[i]/proportions[i],i))
        out[best].append(g);loads[best].update(gv[g]);sizes[best]+=len(g)
    if any(not part for part in out):
        raise ValueError('Not enough independent groups for nonempty partitions')
    return out


def annotation_signature(s):
    # Exact text preserves character offsets; entity IDs do not affect the signature.
    entities={e['id']:(e['type'],tuple(tuple(x) for x in e['segments'])) for e in s['entities']}
    return (s['text'],tuple(sorted(entities.values())),
            tuple(sorted((r['type'],entities[r['head']],entities[r['tail']]) for r in s['relations'])))


def dedup_training(rows):
    buckets=defaultdict(list)
    for s in rows:buckets[normalized_text(s['text'])].append(s)
    keep, audit=[],[]
    for key, bucket in sorted(buckets.items()):
        bucket=sorted(bucket,key=lambda s:s['sample_id'])
        if len(bucket)==1:
            keep.extend(bucket);continue
        if len({annotation_signature(s) for s in bucket})==1:
            keep.append(bucket[0])
            for s in bucket[1:]:audit.append({'sample_id':s['sample_id'],'kept_id':bucket[0]['sample_id'],'reason':'identical_text_and_annotation'})
        else:
            for s in bucket:audit.append({'sample_id':s['sample_id'],'kept_id':None,'reason':'conflicting_duplicate_group_excluded_from_training'})
    return sorted(keep,key=lambda s:s['sample_id']),audit


def build_folds(dataset, output, seed=42):
    dataset,output=Path(dataset),Path(output)
    if output.exists() and any(output.iterdir()):raise ValueError('Refusing to overwrite fixed folds')
    reports=read_jsonl(dataset/'reports.jsonl');samples=read_jsonl(dataset/'sentences.jsonl')
    vectors={r['report_id']:tuple(sorted(set(['entity:'+e['type'] for e in r['entities']]+['relation:'+e['type'] for e in r['relations']]))) for r in reports}
    groups,buckets=duplicate_components(reports,samples)
    outer=assign_groups(groups,vectors,[.2]*5,seed)
    output.mkdir(parents=True,exist_ok=True)
    report_ids={r['report_id'] for r in reports}
    duplicates=[{'normalized_text':text,'sample_ids':[s['sample_id'] for s in rows],
                 'report_ids':sorted({s['report_id'] for s in rows})}
                for text,rows in sorted(buckets.items()) if len(rows)>1]
    write_jsonl(output/'duplicate_texts.jsonl',duplicates)
    write_json(output/'components.json', {'components':groups,'largest_component_reports':max(map(len,groups))})
    global_counts=Counter(k for rid in sorted(vectors) for k in vectors[rid])
    totals=[];test_all=[];deviations=[]
    for f,test_groups in enumerate(outer,1):
        remaining=[g for i,part in enumerate(outer,1) if i!=f for g in part]
        train_groups,val_groups=assign_groups(remaining,vectors,[.9,.1],seed+f)
        parts={'train':sorted(r for g in train_groups for r in g),
               'validation':sorted(r for g in val_groups for r in g),
               'test':sorted(r for g in test_groups for r in g)}
        sets={k:set(v) for k,v in parts.items()}
        assert set.union(*sets.values())==report_ids
        assert all(not sets[a]&sets[b] for a,b in [('train','validation'),('train','test'),('validation','test')])
        rows={k:[s for s in samples if s['report_id'] in ids] for k,ids in sets.items()}
        textsets={k:{normalized_text(s['text']) for s in ss} for k,ss in rows.items()}
        assert all(not textsets[a]&textsets[b] for a,b in [('train','validation'),('train','test'),('validation','test')])
        before=len(rows['train']);rows['train'],removed=dedup_training(rows['train'])
        fold=output/f'fold_{f}';fold.mkdir()
        for name in parts:
            write_json(fold/f'{name}_reports.json',parts[name])
            write_json(fold/f'{name}_sample_ids.json',[s['sample_id'] for s in rows[name]])
            write_jsonl(fold/f'{name}.jsonl',rows[name])
        write_jsonl(fold/'training_exclusions.jsonl',removed)
        dist={}
        for name,ids in parts.items():
            rc=Counter(label for rid in ids for label in vectors[rid])
            actual=Counter(r['type'] for s in rows[name] for r in s['relations'])
            dist[name]={'reports':len(ids),'samples':len(rows[name]),'report_label_support':{k:rc[k] for k in LABELS},
                        'sentence_relation_instances':{k:actual[k] for k in RELATION_TYPES},
                        'sentence_entity_mentions':dict(Counter(e['type'] for s in rows[name] for e in s['entities']))}
            for label in LABELS:
                deviations.append({'fold':f,'partition':name,'label':label,'support_reports':rc[label],
                                   'global_support_reports':global_counts[label],
                                   'prevalence_difference_pp':round(100*(rc[label]/len(ids)-global_counts[label]/len(reports)),4)})
        write_json(fold/'distribution.json',dist)
        budget=math.ceil(.2*len(rows['train']))
        rec={'fold':f,'seed':seed+f, 'reports':{k:len(v) for k,v in parts.items()},
             'samples':{k:len(v) for k,v in rows.items()},'train_samples_before_dedup':before,
             'train_exclusions':len(removed),'augmentation_acceptance_budget':budget,
             'report_overlaps':0,'normalized_text_overlaps':0,
             'train_exclusion_reasons':dict(Counter(x['reason'] for x in removed))}
        totals.append(rec);test_all.extend(parts['test']);write_json(fold/'summary.json',rec)
    assert len(test_all)==len(set(test_all))==len(reports)
    summary={'protocol':'report_duplicate_component_multilabel_greedy_v1','seed':seed,
             'reports':len(reports),'components':len(groups),'largest_component_reports':max(map(len,groups)),
             'duplicate_text_groups':len(duplicates),'cross_report_duplicate_text_groups':sum(len(x['report_ids'])>1 for x in duplicates),
             'test_reports_once':True,'all_report_partitions_disjoint':True,'all_normalized_text_partitions_disjoint':True,
             'augmentation_ratio':.2,'augmentation_denominator':'post-dedup original train sentences including negatives',
             'folds':totals}
    write_jsonl(output/'label_distribution_audit.jsonl',deviations)
    write_json(output/'summary.json',summary)
    write_json(output/'manifest.json',{'source_reports_sha256':digest(dataset/'reports.jsonl'),
                                      'source_sentences_sha256':digest(dataset/'sentences.jsonl'),
                                      'files':{p.relative_to(output).as_posix():digest(p) for p in sorted(output.rglob('*')) if p.is_file()}})
    return summary
