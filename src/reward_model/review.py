# src/reward_model/review.py
import json
import os
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECTOR_FILE = "medtok_projector.bin"
META_FILE = "custom_training_meta.json"


class Review(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        embedding_dim: int = 256,
        model_hidden_size: int = 4096,
        filler_token_id: int = 128009,
        concat_length: int = 512,
        prefix_count: int = 2,
    ) -> None:
        super().__init__()
        self.llama_model = model
        self.embedding_dim = int(embedding_dim)
        self.model_hidden_size = int(model_hidden_size)
        self.concat_length = int(concat_length)
        self.filler_token_id = int(filler_token_id)
        if prefix_count not in (1, 2):
            raise ValueError("prefix_count must be 1 or 2")
        self.prefix_count = prefix_count
        self.projector = nn.Linear(self.embedding_dim, self.model_hidden_size)

    @property
    def base_model(self):
        m = self.llama_model
        while hasattr(m, "module"):
            m = m.module
        return m

    @property
    def config(self):
        m = self.base_model
        if hasattr(m, "config"):
            return m.config
        raise AttributeError("Cannot find config on wrapped model.")

    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.base_model, "gradient_checkpointing_enable"):
            self.base_model.gradient_checkpointing_enable(**kwargs)

    @property
    def is_gradient_checkpointing(self):
        return getattr(self.base_model, "is_gradient_checkpointing", False)

    def get_input_embeddings(self):
        if hasattr(self.base_model, "get_input_embeddings"):
            return self.base_model.get_input_embeddings()
        return None

    def resize_token_embeddings(self, *args, **kwargs):
        if hasattr(self.base_model, "resize_token_embeddings"):
            return self.base_model.resize_token_embeddings(*args, **kwargs)
        raise AttributeError("Wrapped model does not support resize_token_embeddings().")

    @property
    def device(self):
        return self.projector.weight.device

    def _get_embed_tokens(self):
        m = self.base_model
        for attr_chain in [
            ("model", "embed_tokens"),
            ("model", "model", "embed_tokens"),
            ("base_model", "model", "embed_tokens"),
        ]:
            obj = m
            try:
                for a in attr_chain:
                    obj = getattr(obj, a)
                return obj
            except AttributeError:
                continue
        raise AttributeError(f"Cannot find embed_tokens in wrapped model: {type(m)}")

    def build_meta(self) -> Dict[str, Any]:
        return {
            "wrapped_with_review": True,
            "concat_length": self.concat_length,
            "embedding_dim": self.embedding_dim,
            "model_hidden_size": self.model_hidden_size,
            "filler_token_id": self.filler_token_id,
            "medtok_prefix_version": self.prefix_count,
        }

    def save_pretrained(
        self,
        output_dir: str,
        tokenizer=None,
        safe_serialization: bool = False,
        **kwargs,
    ):
        os.makedirs(output_dir, exist_ok=True)

        # 1) save plain HF causal LM
        self.base_model.save_pretrained(
            output_dir,
            safe_serialization=safe_serialization,
            **kwargs,
        )

        # 2) save tokenizer
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)

        # 3) save projector
        torch.save(
            self.projector.state_dict(),
            os.path.join(output_dir, PROJECTOR_FILE),
        )

        # 4) save metadata
        with open(os.path.join(output_dir, META_FILE), "w", encoding="utf-8") as f:
            json.dump(self.build_meta(), f, ensure_ascii=False, indent=2)

        print(f"[Review] Saved checkpoint to {output_dir}")

    def load_projector(self, checkpoint_dir: str):
        projector_path = os.path.join(checkpoint_dir, PROJECTOR_FILE)
        if not os.path.exists(projector_path):
            raise FileNotFoundError(f"Missing projector file: {projector_path}")

        try:
            state = torch.load(projector_path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(projector_path, map_location="cpu")
        self.projector.load_state_dict(state)

    def _summary(self, kwargs, name: str, device: torch.device) -> torch.Tensor:
        pooled = kwargs.pop(f"{name}_embeddings", None)
        if pooled is None:
            raise ValueError(f"Missing {name}_embeddings")
        pooled = pooled.to(device=device, dtype=self.projector.weight.dtype)
        return self.projector(F.normalize(pooled, p=2, dim=-1))

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        embed_fn = self._get_embed_tokens()
        dev = self.device
        bz = input_ids.shape[0]
        if self.prefix_count == 1:
            summaries = (self._summary(kwargs, "medtok", dev),)
        else:
            summaries = (self._summary(kwargs, "history", dev),
                         self._summary(kwargs, "prefix", dev))

        filler_emb = embed_fn(torch.tensor(self.filler_token_id, device=dev))

        attention_mask_new = torch.zeros((bz, self.concat_length), device=dev, dtype=torch.long)
        concat_embeds = (
            filler_emb.unsqueeze(0).unsqueeze(0)
            .expand(bz, self.concat_length, -1)
            .clone()
        )
        new_labels = torch.full(
            (bz, self.concat_length),
            fill_value=-100,
            dtype=torch.long,
            device=dev,
        )

        for i in range(bz):
            query_embeds = embed_fn(input_ids[i])

            input_embeds_i = torch.cat(
                tuple(summary[i:i + 1].to(query_embeds.dtype) for summary in summaries)
                + (query_embeds,), dim=0
            )

            concat_embeds[i, -input_embeds_i.shape[0]:] = input_embeds_i
            attention_mask_new[i, -input_embeds_i.shape[0]:] = 1
            if attention_mask is not None:
                attention_mask_new[i, -query_embeds.shape[0]:] = attention_mask[i].to(dev)

            if labels is not None:
                new_labels[i, -query_embeds.shape[0]:] = labels[i, :query_embeds.shape[0]]

        return self.base_model(
            input_ids=None,
            attention_mask=attention_mask_new,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=concat_embeds,
            labels=new_labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
