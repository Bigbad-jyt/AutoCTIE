"""Exact character-span entity and directed-relation evaluation."""
from collections import Counter
from acti.schema import ENTITY_TYPES,RELATION_TYPES

VERSION='strict_character_graph_v1'


def _integer(value):return isinstance(value,int) and not isinstance(value,bool)


def _index(rows,kind):
    index={}
    for row in rows:
        sid=row.get('sample_id')
        if not isinstance(sid,str) or not sid:raise ValueError(f'{kind}: missing sample_id')
        if sid in index:raise ValueError(f'{kind}: duplicate sample_id {sid}')
        index[sid]=row
    return index


def canonical_graph(row,reference):
    """Map entity IDs to types and absolute half-open spans."""
    if row.get('report_id')!=reference['report_id']:raise ValueError('Report identity mismatch')
    base=reference['char_start']
    if not _integer(base) or base<0:raise ValueError('Invalid original character offset')
    if row.get('char_start')!=base:raise ValueError('Sentence original offset mismatch')
    if 'text' in row and row['text']!=reference['text']:raise ValueError('Sentence text mismatch')
    if not isinstance(row.get('entities'),list) or not isinstance(row.get('relations'),list):
        raise ValueError('Explicit entities/relations lists required, including empty predictions')
    entities={};keys=[]
    for entity in row['entities']:
        eid=entity.get('id');kind=entity.get('type');a,b=entity.get('start'),entity.get('end')
        if not isinstance(eid,str) or not eid:raise ValueError('Missing entity ID')
        if kind not in ENTITY_TYPES:raise ValueError('Unknown entity type')
        if not _integer(a) or not _integer(b) or not 0<=a<b<=len(reference['text']):
            raise ValueError('Invalid entity span')
        if 'segments' in entity and entity['segments']!=[[a,b]]:
            raise ValueError('Evaluator scope requires a single continuous entity span')
        if 'text' in entity and entity['text']!=reference['text'][a:b]:raise ValueError('Entity text/span mismatch')
        key=(kind,base+a,base+b)
        if eid in entities and entities[eid]!=key:raise ValueError('Conflicting entity ID')
        entities[eid]=key;keys.append(key)
    relations=[]
    for relation in row['relations']:
        kind=relation.get('type');head=relation.get('head');tail=relation.get('tail')
        if kind not in RELATION_TYPES:raise ValueError('Unknown relation type (no_relation must be omitted)')
        if head not in entities or tail not in entities:raise ValueError('Dangling relation endpoint')
        if entities[head]==entities[tail]:raise ValueError('Self relation outside task scope')
        relations.append((kind,entities[head],entities[tail]))
    return set(keys),set(relations),{'entity_duplicates':len(keys)-len(set(keys)),
        'relation_duplicates':len(relations)-len(set(relations))}


def _counts(gold,predicted):
    return {'tp':len(gold&predicted),'fp':len(predicted-gold),'fn':len(gold-predicted)}


def score_sets(gold,predicted,classes):
    return {'counts':_counts(gold,predicted),'per_class':{
        kind:_counts({x for x in gold if x[0]==kind},{x for x in predicted if x[0]==kind}) for kind in classes}}


def _overlap(a,b):return max(a[1],b[1])<min(a[2],b[2])


def _entity_category(item,other,side):
    if any(item[1:]==x[1:] for x in other):return 'entity_type_error'
    if any(item[0]==x[0] and _overlap(item,x) for x in other):return 'entity_boundary_error'
    if any(_overlap(item,x) for x in other):return 'entity_type_and_boundary_error'
    return 'spurious_entity' if side=='fp' else 'missing_entity'


def _relation_category(item,other,side):
    kind,head,tail=item
    if (kind,tail,head) in other:return 'direction_error'
    if any(head==h and tail==t for k,h,t in other):return 'relation_type_error'
    if any(kind==k and head[1:]==h[1:] and tail[1:]==t[1:] for k,h,t in other):return 'endpoint_type_error'
    if any(kind==k and head[0]==h[0] and tail[0]==t[0] and _overlap(head,h) and _overlap(tail,t) for k,h,t in other):
        return 'endpoint_boundary_error'
    return 'unmatched_relation_fp' if side=='fp' else 'unmatched_relation_fn'


def _sample_result(sid,report_id,char_start,ge,pe,gr,pr,duplicates):
    row={'sample_id':sid,'report_id':report_id,'char_start':char_start,
        'entity':dict(score_sets(ge,pe,ENTITY_TYPES),gold=sorted(ge),predicted=sorted(pe)),
        'relation':dict(score_sets(gr,pr,RELATION_TYPES),gold=sorted(gr),predicted=sorted(pr)),
        'duplicate_records':duplicates}
    errors=[]
    for task,gold,predicted,classifier in [('entity',ge,pe,_entity_category),('relation',gr,pr,_relation_category)]:
        for side,items,other in [('fp',predicted-gold,gold),('fn',gold-predicted,predicted)]:
            errors.extend({'sample_id':sid,'report_id':report_id,'task':task,'side':side,
                           'category':classifier(item,other,side),'item':item} for item in sorted(items))
    return row,errors


def _metric(counts,support_zero_as_null=False):
    tp,fp,fn=(counts[x] for x in ('tp','fp','fn'));support=tp+fn;predicted=tp+fp
    if support_zero_as_null and support==0:
        precision=recall=f1=None
    else:
        precision=tp/predicted if predicted else 0.
        recall=tp/support if support else 0.
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
    return dict(counts,support=support,predicted=predicted,precision=precision,recall=recall,f1=f1)


def metrics_from_count_rows(rows):
    metrics={}
    for task,classes in [('entity',ENTITY_TYPES),('relation',RELATION_TYPES)]:
        total={k:sum(r[task]['counts'][k] for r in rows) for k in ('tp','fp','fn')}
        per_class={kind:_metric({k:sum(r[task]['per_class'][kind][k] for r in rows) for k in ('tp','fp','fn')},True) for kind in classes}
        supported=[c for c in classes if per_class[c]['support']>0]
        metrics[task]={'micro':_metric(total),'per_class':per_class,
            'macro':{**{k:sum(per_class[c][k] for c in supported)/len(supported) if supported else None
                for k in ('precision','recall','f1')},'included_classes':supported,
                'excluded_zero_support_classes':[c for c in classes if c not in supported]}}
    return metrics


def evaluate_rows(gold_rows,prediction_rows):
    gold=_index(gold_rows,'gold');predicted=_index(prediction_rows,'prediction')
    if gold.keys()!=predicted.keys():
        raise ValueError(f"Prediction coverage mismatch: missing={sorted(gold.keys()-predicted.keys())}, extra={sorted(predicted.keys()-gold.keys())}")
    samples=[];errors=[]
    for sid in sorted(gold):
        ge,gr,gd=canonical_graph(gold[sid],gold[sid]);pe,pr,pd=canonical_graph(predicted[sid],gold[sid])
        row,err=_sample_result(sid,gold[sid]['report_id'],gold[sid]['char_start'],ge,pe,gr,pr,
            {'gold':gd,'prediction':pd})
        samples.append(row);errors.extend(err)
    metrics=metrics_from_count_rows(samples)
    metrics['evaluation']={'version':VERSION,'samples':len(samples),
        'offsets':'original_report_unicode_codepoints_half_open',
        'unit':'unique_entity_and_directed_relation_sets_within_each_sample',
        'micro_undefined_denominator':0.,'zero_gold_support_class_metrics':None,
        'macro':'arithmetic_mean_over_classes_with_nonzero_gold_support',
        'duplicates_collapsed':{source:{kind:sum(r['duplicate_records'][source][kind] for r in samples)
            for kind in ('entity_duplicates','relation_duplicates')} for source in ('gold','prediction')},
        'diagnostic_errors':dict(sorted(Counter(f"{e['task']}:{e['side']}:{e['category']}" for e in errors).items())),
        'diagnostic_note':'Categories are deterministic non-paired hints; they do not affect exact set scoring.'}
    return metrics,samples,errors


def _entity_key(value):
    if not isinstance(value,(list,tuple)) or len(value)!=3:raise ValueError('Invalid canonical entity')
    kind,a,b=value
    if kind not in ENTITY_TYPES or not _integer(a) or not _integer(b) or not 0<=a<b:
        raise ValueError('Invalid canonical entity')
    return kind,a,b


def _relation_key(value):
    if not isinstance(value,(list,tuple)) or len(value)!=3 or value[0] not in RELATION_TYPES:
        raise ValueError('Invalid canonical relation')
    head,tail=_entity_key(value[1]),_entity_key(value[2])
    if head==tail:raise ValueError('Invalid canonical self relation')
    return value[0],head,tail


def aggregate_saved_samples(rows):
    """Recompute counts from saved entity and relation sets."""
    index=_index(rows,'per-sample audit');checked=[]
    for sid,row in sorted(index.items()):
        for task,classes,convert in [('entity',ENTITY_TYPES,_entity_key),('relation',RELATION_TYPES,_relation_key)]:
            gold={convert(x) for x in row[task]['gold']};predicted={convert(x) for x in row[task]['predicted']}
            expected=score_sets(gold,predicted,classes)
            if expected['counts']!=row[task]['counts'] or expected['per_class']!=row[task]['per_class']:
                raise ValueError('Saved counters disagree with canonical sets')
        for side in ('gold','predicted'):
            entity_set={_entity_key(x) for x in row['entity'][side]}
            if any(h not in entity_set or t not in entity_set for k,h,t in {_relation_key(x) for x in row['relation'][side]}):
                raise ValueError('Saved relation has dangling canonical endpoint')
        checked.append(row)
    return metrics_from_count_rows(checked)
