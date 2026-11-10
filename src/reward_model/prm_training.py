#!/usr/bin/env python
# coding: utf-8
"""
Fine-tune LLM on PRM dataset with optional RAG and MedTok support.
"""

# =====================================================================
# 1. Import Libraries
# =====================================================================
import argparse
import yaml
import torch
import json
import os

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset
from reward_model.review import Review, PROJECTOR_FILE, META_FILE
from reward_model.medtok_features import load_medtok_table
from reward_model.training_data import normalize_train_json_paths, build_dataset_from_json, split_json_data
from reward_model.trainer import AutoRegressiveTrainer
from reward_model.losses import (
    PRM_LOSS_NAME, PRM_LOSS_REDUCTION,
)

# Optional online logging.
try:
    import wandb
except ImportError:
    wandb = None

# =====================================================================
# 2. Argument Parser
# =====================================================================
def parse_args(argv=None, *, default_filtering="no"):
    parser = argparse.ArgumentParser(
        description="Fine-tune LLM model on PRM dataset."
    )

    # Config file support
    parser.add_argument("-c", "--config", type=str, help="Path to YAML configuration file.")

    # Model related
    parser.add_argument("--model_path", type=str, help="Path or name of the LLM model.")
    parser.add_argument("--device", type=str, help="Device name")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type (default: bfloat16).")
    parser.add_argument("--max_token_len", type=int, default=1024, help="Maximum token length (default: 1024).")

    # Data related
    parser.add_argument("--train_json", type=str, nargs="+", help="Path(s) to training JSON file(s). Supports one or multiple JSON files.")
    parser.add_argument("--train_ratio", type=float, default=1.0, help="Ratio for train/valid split (default: 1.0).")

    # Training hyperparameters
    parser.add_argument("--output_dir", type=str, help="Output directory.")
    parser.add_argument("--logging_steps", type=int, help="Logging steps.")
    parser.add_argument("--num_train_epochs", type=int, help="Number of epochs.")
    parser.add_argument("--learning_rate", type=float, help="Learning rate.")
    parser.add_argument("--gradient_accumulation_steps", type=int, help="Gradient accumulation steps.")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="Learning rate scheduler type.")
    parser.add_argument("--per_device_train_batch_size", type=int, help="Training batch size per device.")
    parser.add_argument("--run_name", type=str, help="Name for the run in logs.")
    parser.add_argument("--save_steps", type=int, help="Interval steps for saving checkpoints.")

    # Label types
    parser.add_argument("--train_label", type=str,
                        choices=["prm_soft_label", "prm_hard_label", "gemini_label", "llama_label", "orm_label", "prm_target_label"],
                        help="Which training label to use.")

    # Filtering
    parser.add_argument("--do_filtering", type=str, default=default_filtering,
                        help="Whether to perform filtering for orm_label=0/1. 'yes' or 'no'")

    # RAG usage
    parser.add_argument("--use_rag", type=str, default="no",
                        choices=["yes", "no"],
                        help="Include related docs (RAG) if 'yes'.")

    # Online related
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--wandb_token", type=str, help="Wandb API token for online mode.")
    parser.add_argument("--wandb_project", type=str, help="Wandb project name for online mode.")
    parser.add_argument("--hf_token", type=str, help="HuggingFace Hub token for online mode.")

    # MedTok integration
    parser.add_argument("--medtok_emb_npy",type=str,required=False,default=None,help="Path to precomputed MedTok embeddings.npy",)
    parser.add_argument("--medtok_vocab_size", type=int, default=None, help="Expected number of MedTok token rows.")
    parser.add_argument("--medtok_concat_len", type=int, default=512,
                        help="Total padded sequence length for Review wrapper (default: 512).")
    parser.add_argument("--medtok_projector_path", type=str, default=None,
                        help="Path to load a pre-trained MedTok projector.pth (optional).")

    parser.add_argument(
        "--init_from_rm_checkpoint", type=str,
        help="Initialize PRM weights/projector/tokenizer; start a new optimizer and scheduler.",
    )
    args, _ = parser.parse_known_args(argv)

    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)
            if yaml_config:
                parser.set_defaults(**yaml_config)

    args = parser.parse_args(argv)

    if not args.medtok_emb_npy:
        parser.error("--medtok_emb_npy is required because MedTok is always enabled.")

    if not args.init_from_rm_checkpoint and not args.model_path:
        parser.error("--model_path is required unless --init_from_rm_checkpoint is set.")
    if args.init_from_rm_checkpoint and args.medtok_projector_path:
        parser.error("A checkpoint already supplies the projector; do not override it.")
    return args


# =====================================================================
# 3. Model & Tokenizer Loader
# =====================================================================
def load_model(model_name="meta-llama/Llama-3.1-8B-Instruct", dtype="bfloat16"):
    print("Loading model...")
    print(f"Model being loaded: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Tokenizer loaded: {tokenizer.__class__.__name__}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if dtype == "bfloat16" else torch.float16,
        attn_implementation="flash_attention_2",
    )
    print(f"Model loaded: {model.__class__.__name__}, Parameters: {model.num_parameters():,}")
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token to eos_token: {tokenizer.pad_token}")

    print("Model loaded successfully.")
    return model, tokenizer

def load_review_checkpoint_for_training(checkpoint_dir: str, dtype: str = "bfloat16"):
    """
    Load an already trained reward-model checkpoint for continued training,
    but ONLY restore model/projector/tokenizer weights.
    Optimizer/scheduler states are intentionally NOT restored.
    """
    print(f"Loading trained RM checkpoint for continued training: {checkpoint_dir}")

    meta_path = os.path.join(checkpoint_dir, META_FILE)
    projector_path = os.path.join(checkpoint_dir, PROJECTOR_FILE)

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing meta file: {meta_path}")
    if not os.path.exists(projector_path):
        raise FileNotFoundError(f"Missing projector file: {projector_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("medtok_prefix_version") != 2:
        raise ValueError("Legacy one-vector PRM checkpoint cannot be continued with the two-summary architecture")

    base_model, tokenizer = load_model(checkpoint_dir, dtype)

    embedding_dim = int(meta["embedding_dim"])
    model_hidden_size = int(meta["model_hidden_size"])
    concat_length = int(meta["concat_length"])
    filler_token_id = int(meta["filler_token_id"])

    model = Review(
        model=base_model,
        embedding_dim=embedding_dim,
        model_hidden_size=model_hidden_size,
        filler_token_id=filler_token_id,
        concat_length=concat_length,
    )
    model.load_projector(checkpoint_dir)

    print(
        "[Resume RM] Loaded Review-wrapped checkpoint "
        f"(emb_dim={embedding_dim}, hidden={model_hidden_size}, concat={concat_length})"
    )
    return model, tokenizer


def main(argv=None, *, default_filtering="no", continuation_defaults=False):
    args = parse_args(argv, default_filtering=default_filtering)

    use_wandb = args.online

    if use_wandb:
        if wandb is None:
            raise ImportError("wandb is not installed, but --online was provided.")

        if not args.wandb_token:
            raise ValueError("WANDB token is required when --online is set.")

        os.environ["WANDB_API_KEY"] = args.wandb_token
        wandb.login(key=args.wandb_token)

        wandb.init(
            project=args.wandb_project or "Med-PRM",
            name=args.run_name,
            config={
                "model_path": args.model_path,
                "train_json": args.train_json,
                "train_label": args.train_label,
                "use_rag": args.use_rag,
                "max_token_len": args.max_token_len,
                "num_train_epochs": args.num_train_epochs,
                "learning_rate": args.learning_rate,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "per_device_train_batch_size": args.per_device_train_batch_size,
                "lr_scheduler_type": args.lr_scheduler_type,
                "do_filtering": args.do_filtering,
                "bf16": args.dtype == "bfloat16",
                "use_medtok": True,
            }
        )

    if args.init_from_rm_checkpoint:
        model, tokenizer = load_review_checkpoint_for_training(args.init_from_rm_checkpoint, args.dtype)
    else:
        model, tokenizer = load_model(args.model_path, args.dtype)

    step_token = " ки"
    added_step_token = step_token not in tokenizer.get_vocab()
    if added_step_token:
        tokenizer.add_special_tokens({"additional_special_tokens": [step_token]})
    if added_step_token or not args.init_from_rm_checkpoint:
        model.resize_token_embeddings(len(tokenizer))
    step_tag_id = tokenizer.encode(step_token)[-1]

    # 2b) MedTok setup
    medtok_emb_table = load_medtok_table(args.medtok_emb_npy, args.medtok_vocab_size)
    medtok_emb_dim = medtok_emb_table.shape[1]
    print(f"[MedTok] Loaded precomputed embeddings from {args.medtok_emb_npy}, shape={medtok_emb_table.shape}")

    auto_concat_len = 2 + args.max_token_len
    if args.medtok_concat_len != auto_concat_len:
        print(f"[MedTok] Overriding medtok_concat_len: {args.medtok_concat_len} -> {auto_concat_len}")
        args.medtok_concat_len = auto_concat_len
    
    # 3) Build datasets
    train_json_paths = normalize_train_json_paths(args.train_json)
    print(f"Training JSON files: {train_json_paths}")
    train_raw, valid_raw = split_json_data(train_json_paths, args=args)

    if isinstance(model, Review):
        if model.embedding_dim != medtok_emb_dim:
            raise ValueError("MedTok table dimension differs from checkpoint projector")
        if model.concat_length != args.medtok_concat_len:
            raise ValueError("Checkpoint context length differs from max_token_len + 2")
    else:
        model = Review(
            model=model,
            embedding_dim=medtok_emb_dim,
            model_hidden_size=model.config.hidden_size,
            filler_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
            concat_length=args.medtok_concat_len,
        )
        if args.medtok_projector_path:
            model.projector.load_state_dict(
                torch.load(args.medtok_projector_path, map_location="cpu", weights_only=True)
            )

    train_dict = build_dataset_from_json(
        train_raw, tokenizer, step_tag_id, args, medtok_emb_table=medtok_emb_table,
    )
    train_ds = Dataset.from_dict(train_dict)

    if len(valid_raw) == 0:
        valid_ds = None
    else:
        valid_dict = build_dataset_from_json(
            valid_raw, tokenizer, step_tag_id, args,
            medtok_emb_table=medtok_emb_table,
        )
        valid_ds = Dataset.from_dict(valid_dict)

    # 4) Padding/Cutting
    # Two summary vectors precede the padded text input.
    max_len = args.max_token_len

    def pad_fn(ex, max_len=max_len, pad_val=0):
        iid, attn, lab, val = ex["input_ids"], ex["attention_mask"], ex["labels"], ex["values"]
        if len(iid) < max_len:
            pad = max_len - len(iid)
            iid += [pad_val] * pad
            attn += [0] * pad
            lab += [-100] * pad
            val += [0] * pad
        else:
            iid, attn, lab, val = iid[:max_len], attn[:max_len], lab[:max_len], val[:max_len]
        result = {
            "input_ids": iid,
            "attention_mask": attn,
            "labels": lab,
            "values": val,
        }
        result["history_embeddings"] = ex["history_embeddings"]
        result["prefix_embeddings"] = ex["prefix_embeddings"]
        return result

    train_ds = train_ds.map(pad_fn)
    if valid_ds is not None:
        valid_ds = valid_ds.map(pad_fn)

    # 5) Trainer
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )
    run_name = (
        args.run_name
        or f"{os.path.basename(args.init_from_rm_checkpoint or args.model_path)}_{args.train_label}_{args.learning_rate}"
    )

    # save settings
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    if args.save_steps and args.save_steps > 0:
        save_strategy = "steps"
        save_steps = args.save_steps
    else:
        save_strategy = "epoch"
        save_steps = None

    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        eval_strategy="no",
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,
        save_strategy="no",
        save_steps=None,
        save_total_limit=None,
        logging_steps=args.logging_steps,
        run_name=run_name,
        bf16=(args.dtype=="bfloat16"),
        remove_unused_columns=False,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        report_to=["wandb"] if use_wandb else ["none"],
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=0.05,

        **({"optim": "paged_adamw_8bit", "ddp_find_unused_parameters": False}
           if not args.init_from_rm_checkpoint and not continuation_defaults else {}),
    )

    trainer = AutoRegressiveTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=data_collator,
        my_tokenizer=tokenizer,
        objective_metadata={
            "loss": PRM_LOSS_NAME,
            "reduction": PRM_LOSS_REDUCTION,
            "train_label": args.train_label,
            "train_json": normalize_train_json_paths(args.train_json),
            "target_origin": "existing input labels; no automatic relabeling",
        },
    )

    # 6) Training
    print(f"● Fine-tuning start | lr={args.learning_rate} | filter={args.do_filtering} | RAG={args.use_rag}")

    try:
        trainer.train()
        print("✅ Fine-tuning finished")

        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
    finally:
        if use_wandb and wandb is not None:
            wandb.finish()

    print("🎉 Task completed!")

if __name__ == "__main__":
    main()
