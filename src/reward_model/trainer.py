"""Reward-marker objective and Review checkpoint saving."""

import json
import os
import torch
from transformers import Trainer
from transformers.trainer import TRAINING_ARGS_NAME
from reward_model.review import Review
from reward_model.losses import reward_marker_mse, PRM_OBJECTIVE_FILE


class AutoRegressiveTrainer(Trainer):
    def __init__(self, my_tokenizer=None, objective_metadata=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.my_tokenizer = my_tokenizer
        self.objective_metadata = objective_metadata

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        values = inputs.pop("values")

        outputs = model(**inputs, labels=labels)
        logits = outputs["logits"]

        plus_id = self.my_tokenizer.encode(" +", add_special_tokens=False)[-1]
        minus_id = self.my_tokenizer.encode(" -", add_special_tokens=False)[-1]

        seq_len_logits = logits.size(1)
        seq_len_labels = labels.size(1)

        if seq_len_logits == seq_len_labels + 2:
            pad_label = torch.full(
                (labels.size(0), 2), -100, dtype=labels.dtype, device=labels.device
            )
            pad_value = torch.zeros(
                (values.size(0), 2), dtype=values.dtype, device=values.device
            )
            labels = torch.cat([pad_label, labels], dim=1)
            values = torch.cat([pad_value, values], dim=1)
        elif seq_len_logits != seq_len_labels:
            raise ValueError(
                f"Sequence length mismatch: logits={seq_len_logits}, labels={seq_len_labels}"
            )

        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()
        values = values[..., 1:].contiguous().float()

        plus_logits = logits[:, :, plus_id]
        minus_logits = logits[:, :, minus_id]
        chosen = labels != -100

        loss = reward_marker_mse(plus_logits, minus_logits, values, chosen)

        return (loss, outputs) if return_outputs else loss

    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        raw_model = self.model
        while hasattr(raw_model, "module"):
            raw_model = raw_model.module

        if not isinstance(raw_model, Review):
            raise TypeError(
                f"Expected Review-wrapped model, got {type(raw_model)}. "
                "PRM training now assumes MedTok is always enabled."
            )

        raw_model.save_pretrained(
            output_dir,
            tokenizer=self.my_tokenizer,
            safe_serialization=getattr(self.args, "save_safetensors", False),
        )

        if self.objective_metadata is not None:
            with open(os.path.join(output_dir, PRM_OBJECTIVE_FILE), "w", encoding="utf-8") as f:
                json.dump(self.objective_metadata, f, ensure_ascii=False, indent=2)

        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        print(f"✅ Saved checkpoint to: {output_dir}")
