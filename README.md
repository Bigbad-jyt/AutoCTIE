# AutoCTIE

Core implementation and prepared data for **AutoCTIE: A Framework for Joint
Entity and Relation Extraction from Automotive Cyber Threat Intelligence**.

The extractor combines a pretrained encoder, BiLSTM, self-attention, a BIOES
CRF entity head and an independent multilabel relation head for ordered pairs.
Inference uses predicted entities. The evaluator requires exact entity spans,
entity types, relation types and direction.

## Contents

| Path | Contents |
| --- | --- |
| `data/` | Model-view Acti records, five-fold splits, selected augmentation and labels |
| `model/acti/` | Core model, training, batching, alignment and evaluation implementation |
| `model/tokenizers/` | Small tokenizer snapshots; no pretrained weights |
| `preprocess.py` | Prepare encoded inputs for one encoder and one or all folds |
| `train.py` | Train a single condition/fold, or validate inputs with `--dry-run` |
| `predict.py` | Checkpoint-based entity and directed relation extraction |
| `evaluate.py` | Strict entity/triple evaluation from JSONL |
| `config.yaml` | Model parameters and C1-C4 training settings |

## Environment

Python 3.10-3.12. Install dependencies in a separate environment:

```bash
python -m pip install -r requirements.txt
```

Select the PyTorch build appropriate for your CPU/CUDA installation. The source
project used PyTorch 2.8.0 and Transformers 4.56.2. No package installation or
model download occurs automatically when these entry points are imported.

## Data and encoders

The original Acti corpus is at https://doi.org/10.6084/m9.figshare.27916758.v1.
See `data/README.md` for our prepared subset, exact splits and annotation schema.

Obtain a local pretrained BERT or CySecBERT snapshot separately. The encoder
loader expects `config.json` and `model.safetensors` readable by Transformers
`BertModel`. BERT corresponds to `bert-base-uncased`; CySecBERT corresponds to
https://huggingface.co/markusbayer/CySecBERT. Use the supplied matching tokenizer
snapshot during preprocessing and inference. Weights and trained AutoCTIE
checkpoints are not bundled.

## Run

Prepare a fold without downloading a model or starting training:

```bash
python preprocess.py --encoder CySecBERT --fold 1
python train.py --condition C4 --fold 1 --dry-run
```

Omit `--fold` from preprocessing to prepare all five folds. Prepare BERT inputs
with `--encoder BERT` for C1/C2. Conditions are C1 = BERT, C2 = BERT plus
augmentation, C3 = CySecBERT, C4 = CySecBERT plus augmentation.

Start training only when intended, after obtaining pretrained weights:

```bash
python train.py --condition C4 --fold 1 --encoder-path weights/CySecBERT --device cuda
```

The maximum budget is 120 original-training-set-equivalent epochs, batch 16,
microbatch 2 and patience 20. Selection uses validation triple micro-F1, then
entity micro-F1; test data are evaluated after selection. Relative paths are
resolved from the repository root. Existing outputs are not overwritten.

Predict from a checkpoint produced by this implementation, then evaluate:

```bash
python predict.py --checkpoint outputs/fold1_C4/checkpoints/step_000145 --input data/encoded/CySecBERT/fold_1/test.jsonl --tokenizer model/tokenizers/CySecBERT --output outputs/predictions.jsonl
python evaluate.py --gold data/encoded/CySecBERT/fold_1/test.jsonl --predictions outputs/predictions.jsonl --output outputs/evaluation
```

Prediction ignores any gold entity/relation fields in its input.

## Scope

This repository provides the AutoCTIE implementation, prepared original and
selected synthetic data, and necessary configuration.

## License and attribution

Original AutoCTIE source code is licensed under Apache-2.0. Third-party data
and model resources retain their original terms. See `LICENSE` and
`THIRD_PARTY_NOTICES.md`.
