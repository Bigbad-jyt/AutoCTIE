# Data

Acti source: https://doi.org/10.6084/m9.figshare.27916758.v1

- `original.jsonl`: the union of the effective original sentence records used
  by the five prepared folds. It is a model-view derivative of Acti, not its full
  lossless BRAT archive.
- `splits/fold_N.json`: exact sample lists and report identities for training,
  validation and testing. Training partitions preserve the source project's
  fold-specific deduplication and eligibility decisions.
- `augmented/fold_N.jsonl`: the selected, author-accepted synthetic pool for
  fold N. These files are byte-identical copies of the frozen selected pools.
  Each pool belongs only to its corresponding training fold. It is not a
  validation or test dataset. Candidate reserves and generation/review logs
  are not included in this repository.
- `labels.json`: the ten entity and ten directed relation types.
- `summary.json`: record counts in this preparation copy.

Spans use Python Unicode character offsets and half-open intervals `[start,end)`
relative to each sentence. `char_start` is its position in the source report.
Relations refer to entity identifiers through `head` and `tail`; direction is
semantic and is not the order of appearance in the sentence.

`preprocess.py` performs WordPiece encoding and reversible BIOES alignment on
these supplied records. It does not re-create the lossless BRAT-to-sentence
archive or silently reconstruct omitted records.
