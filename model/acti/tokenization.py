"""Word tokenization, offsets and BIOES alignment."""
import re
from transformers import BertTokenizerFast
from acti.schema import ENTITY_TYPES

LABELS = ('O',) + tuple(f'{p}-{t}' for t in ENTITY_TYPES for p in ('B', 'I', 'E', 'S'))
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
WORD_PATTERN = r'\w+|[^\w\s]'


def words_with_offsets(text):
    return [{'text': m.group(), 'start': m.start(), 'end': m.end()}
            for m in re.finditer(WORD_PATTERN, text)]


def boundary_errors(sample):
    words = words_with_offsets(sample['text'])
    starts = {w['start'] for w in words}
    ends = {w['end'] for w in words}
    return [e['id'] for e in sample['entities']
            if len(e['segments']) != 1 or e['start'] not in starts or e['end'] not in ends]


def load_tokenizer(path):
    # Load the local tokenizer snapshot.
    tokenizer = BertTokenizerFast.from_pretrained(path, local_files_only=True)
    if not tokenizer.is_fast:
        raise ValueError('Fast tokenizer required')
    return tokenizer


def decode_bioes(labels, words):
    result = []
    open_entity = None
    for i, label in enumerate(labels):
        if label not in LABEL_TO_ID:
            raise ValueError(f'Unknown label {label}')
        if label == 'O':
            if open_entity is not None:
                raise ValueError('Unclosed BIOES entity')
            continue
        prefix, entity_type = label.split('-', 1)
        if prefix in ('B', 'S'):
            if open_entity is not None:
                raise ValueError('Overlapping BIOES entity')
            if prefix == 'S':
                result.append((entity_type, words[i]['start'], words[i]['end']))
            else:
                open_entity = (entity_type, i)
        else:
            if open_entity is None or open_entity[0] != entity_type:
                raise ValueError('Illegal BIOES transition')
            if prefix == 'E':
                result.append((entity_type, words[open_entity[1]]['start'], words[i]['end']))
                open_entity = None
    if open_entity is not None:
        raise ValueError('Unclosed BIOES entity')
    return result


def encode_sample(sample, tokenizer, max_length=512, allow_training_truncation=False):
    if boundary_errors(sample):
        raise ValueError(f"Unrepresentable word boundary: {sample['sample_id']}")
    words = words_with_offsets(sample['text'])
    if not words:
        raise ValueError('No lexical words')
    entities = []
    for e in sample['entities']:
        if e['type'] not in ENTITY_TYPES:
            raise ValueError('Unknown entity type')
        indices = [i for i, w in enumerate(words) if e['start'] <= w['start'] and w['end'] <= e['end']]
        entities.append(dict(e, word_start=indices[0], word_end=indices[-1] + 1))
    encoded = tokenizer([w['text'] for w in words], is_split_into_words=True,
                        return_offsets_mapping=True, truncation=False)
    full_length = len(encoded['input_ids'])
    keep = len(words)
    if full_length > max_length:
        if not allow_training_truncation:
            raise ValueError('Evaluation sample exceeds max_length; gold must not be shortened')
        budget = max_length - tokenizer.num_special_tokens_to_add(pair=False)
        piece_counts = [encoded.word_ids().count(i) for i in range(len(words))]
        used = 0
        keep = 0
        for size in piece_counts:
            if used + size > budget:
                break
            used += size
            keep += 1
        # Move the truncation boundary before a partially retained entity.
        for e in entities:
            if e['word_start'] < keep < e['word_end']:
                keep = e['word_start']
        if not keep:
            raise ValueError('No complete word/entity fits max_length')
        encoded = tokenizer([w['text'] for w in words[:keep]], is_split_into_words=True,
                            return_offsets_mapping=True, truncation=False)
    retained = [e for e in entities if e['word_end'] <= keep]
    ids = {e['id'] for e in retained}
    labels = ['O'] * keep
    for e in retained:
        n = e['word_end'] - e['word_start']
        tags = ['S'] if n == 1 else ['B'] + ['I'] * (n - 2) + ['E']
        for i, tag in zip(range(e['word_start'], e['word_end']), tags):
            if labels[i] != 'O':
                raise ValueError('Overlapping entity labels')
            labels[i] = f"{tag}-{e['type']}"
    if sorted(decode_bioes(labels, words[:keep])) != sorted((e['type'],e['start'],e['end']) for e in retained):
        raise ValueError('BIOES roundtrip failed')
    word_ids = encoded.word_ids()
    first = [word_ids.index(i) for i in range(keep)]
    offsets = []
    for wid, (a, b) in zip(word_ids, encoded['offset_mapping']):
        if wid is None:
            offsets.append(None)
        else:
            if not 0 <= a < b <= len(words[wid]['text']):
                raise ValueError('Invalid WordPiece offset')
            offsets.append([words[wid]['start'] + a, words[wid]['start'] + b])
    relations = [r for r in sample['relations'] if r['head'] in ids and r['tail'] in ids]
    return {'sample_id': sample['sample_id'], 'report_id': sample['report_id'],
        'text': sample['text'], 'char_start': sample['char_start'], 'words': words[:keep],
        'input_ids': encoded['input_ids'], 'attention_mask': encoded['attention_mask'],
        'token_type_ids': encoded.get('token_type_ids', [0] * len(word_ids)),
        'word_ids': word_ids, 'wordpiece_char_offsets': offsets, 'first_wordpiece_indices': first,
        'labels': labels, 'label_ids': [LABEL_TO_ID[x] for x in labels],
        'entities': retained, 'relations': relations,
        'truncation': {'full_wordpieces': full_length, 'removed_words': len(words) - keep,
            'removed_entity_ids': [e['id'] for e in entities if e['id'] not in ids],
            'removed_relation_ids': [r['id'] for r in sample['relations'] if r not in relations]}}


def first_wordpiece_pool(hidden, indices):
    """Select the first WordPiece embedding for each word."""
    import torch
    return hidden.index_select(0, torch.as_tensor(indices, device=hidden.device, dtype=torch.long))
