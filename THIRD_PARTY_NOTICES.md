# Third-party resources

- Acti corpus: Wang et al., *A dataset for cyber threat intelligence modeling of
  connected autonomous vehicles*, Scientific Data 12, 366 (2025).
  Article: https://doi.org/10.1038/s41597-025-04439-5
  Dataset: https://doi.org/10.6084/m9.figshare.27916758.v1
  Figshare version 1 declares CC BY 4.0:
  https://creativecommons.org/licenses/by/4.0/
  `data/original.jsonl` and its partition indexes are derived from this corpus.
  Changes include sentence-level scope selection, token-boundary alignment and
  fold-specific training deduplication. Original report/sample identifiers and
  character-span annotations are retained. These data are not an unchanged
  redistribution of the complete original Acti archive.
- Original Acti implementation: https://github.com/AutoCS-wyh/Automotive-cyber-threat-intelligence-corpus
  Its Apache-2.0 license does not automatically license our separate implementation.
- BERT / Hugging Face Transformers: https://github.com/huggingface/transformers
- CySecBERT: https://huggingface.co/markusbayer/CySecBERT
  Pretrained model weights are not included. The small tokenizer resources are
  local snapshots used by the source experiment project. Retain their upstream
  terms when redistributing. CySecBERT source revision recorded by that project:
  `c9e04e355a2bc0690b6e55833a9895d7591afcd0`.
