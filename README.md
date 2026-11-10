# EHR-MCTS

official implementation for "EHR-MCTS: Knowledge-Enhanced Process Reward Guided Monte Carlo Tree Search for Longitudinal EHR Event Forecasting"

## Setup

Use a Python environment with CUDA-enabled PyTorch. PRM training and inference also require FlashAttention 2.

```bash
pip install -r requirements.txt
python -m pip install -e .
```

## Data and configs

Obtain UK Biobank and MIMIC-III data through their respective access processes. Process the data with our provided scripts.
See `prepare_data/prepare_ukb.py` and `prepare_data/prepare_mimic.py`

## Training

Train the event generator for the selected dataset:

```bash
python -u train.py --config configs/yaml/ehrcs_config_ukb.yaml
```

Train the PRM using the shared config after setting its dataset-specific paths:

```bash
python -m reward_model.prm_training -c configs/yaml/prm_config.yaml
```

## Inference

Set `RAG_JSONL` to the matching knowledge-base file and use the trained PRM checkpoint:

```bash
python -m reason.inference.run_mcts \
  --config configs/yaml/ehrcs_config_ukb.yaml \
  --rm_config configs/yaml/prm_config.yaml \
  --rm_model_path checkpoints/reward_model_v1 \
  --task_type ukb --rm_label_jsonl data/ICD.jsonl --rm_rag_jsonl "$RAG_JSONL" \
  --histories observed.jsonl --output predictions.jsonl
```
