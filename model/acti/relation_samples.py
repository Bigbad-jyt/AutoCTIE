"""Ordered entity-pair generation and negative sampling."""
import hashlib
from itertools import permutations
from acti.schema import RELATION_TYPES

RELATION_LABELS = RELATION_TYPES


def decode_relation_logits(pairs, logits, threshold=0.5):
    """Threshold relation probabilities and deduplicate directed triples."""
    if threshold != 0.5:raise ValueError('Core protocol fixes threshold at 0.5')
    if tuple(logits.shape)!=(len(pairs),len(RELATION_LABELS)):
        raise ValueError('Expected pair_count by ten raw relation logits')
    relations=[];seen=set()
    for pair, active in zip(pairs,(logits.sigmoid()>=threshold).tolist()):
        for label,enabled in zip(RELATION_LABELS,active):
            triple=(pair['head'],pair['tail'],label)
            if enabled and triple not in seen:
                relations.append(dict(head=pair['head'],tail=pair['tail'],type=label));seen.add(triple)
    return relations


def directed_pairs(entities):
    return list(permutations([e['id'] for e in entities], 2))


def training_pairs(sample, seed, epoch=0, negative_ratio=3):
    if negative_ratio < 0:
        raise ValueError('negative_ratio must be nonnegative')
    positive = {}
    entity_ids = {e['id'] for e in sample['entities']}
    for r in sample['relations']:
        pair = (r['head'], r['tail'])
        if pair[0] not in entity_ids or pair[1] not in entity_ids or pair[0] == pair[1]:
            raise ValueError('Invalid relation endpoints')
        if r['type'] not in RELATION_TYPES:
            raise ValueError('Unknown relation type')
        positive.setdefault(pair,set()).add(r['type'])
    negative = [p for p in directed_pairs(sample['entities']) if p not in positive]
    active = {eid for p in positive for eid in p}
    # Retain all hard negatives, even above the sampling ratio.
    hard = [p for p in negative if p[0] in active or p[1] in active]
    easy = [p for p in negative if p not in hard]
    def rank(pair):
        return hashlib.sha256(f"{seed}|{epoch}|{sample['sample_id']}|{pair[0]}|{pair[1]}".encode()).hexdigest()
    target = negative_ratio * max(1, len(positive))
    selected = set(hard + sorted(easy, key=rank)[:max(0, target - len(hard))])
    return [{'head': a, 'tail': b,
             'labels': [t for t in RELATION_LABELS if t in positive.get((a,b),set())],
             'target': [float(t in positive.get((a,b),set())) for t in RELATION_LABELS],
             'hard_negative': (a, b) in hard}
            for a, b in directed_pairs(sample['entities']) if (a, b) in positive or (a, b) in selected]
