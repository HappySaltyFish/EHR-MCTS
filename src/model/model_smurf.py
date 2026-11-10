import math
import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
from torch.nn import functional as F

from model.model import (
    Delphi, DelphiConfig, LayerNorm, CausalSelfAttention, MLP, Block, AgeEncoding, new_gelu
)
from dataset.dataset import GAP_THRESH
from utils import (
    TIME_UNCERTAINTY_BETA,
    ATOMIC_CANDIDATE_POOL_SIZE,
    TIME_UNCERTAINTY_CLIP_MAX,
    TIME_UNCERTAINTY_EPS,
    LANGEVIN_SAMPLES_PER_CANDIDATE,
    LANGEVIN_STEPS,
)
TERMINATION_TOKENS = [1145]


def get_obj(t_var, score, non_pad_mask):
    grad_t = torch.autograd.grad(score.sum(), t_var, retain_graph=True)[0].squeeze(-1)
    if grad_t.ndim == 1:
        grad_t = grad_t.unsqueeze(-1)
    grad_t = grad_t.masked_fill_(~non_pad_mask.squeeze(-1)[:,1:].bool(), 0.0)
    obj = grad_t + 0.5 * score ** 2
    obj *= non_pad_mask.squeeze(-1)[:,1:]
    return obj


def get_obj_denoise(
    t_true: torch.Tensor,
    t_noisy: torch.Tensor,
    score: torch.Tensor,
    var_noise: float,
    non_pad_mask: torch.Tensor
) -> torch.Tensor:
    # Target score: (t_true - t_noisy) / var_noise^2
    target = (t_true[:, :, None] - t_noisy) / (var_noise ** 2)
    
    # MSE loss between predicted and target scores
    obj = 0.5 * (score - target) ** 2
    obj *= var_noise ** 2
    obj *= non_pad_mask[:, 1:, :]  # Mask padding positions
    obj = obj.mean(dim=2)  # Average over noise samples
    
    return obj



@dataclass
class DelphiSMURFConfig(DelphiConfig):
    # SMURF Score Matching parameters
    use_score_matching: bool = True
    parametrize: str = "intensity"  # "intensity" or "score"
    
    # Noise parameters for denoising score matching
    add_noise: str = "denoise"  # "denoise" or "None"
    noise_type: str = "normal"  # "normal" or "truncated"
    var_noise: float = 0.1
    num_noise: int = 100
    
    # Langevin dynamics sampling parameters
    langevin_step: float = 3e-2
    n_langevin_steps: int = 300
    n_samples: int = 100
    denoise_step: bool = True
    
    # Loss weighting
    loss_lambda: float = 30.0  # Weight for score matching loss

    normalize: str = "normal"
    visit_token: Optional[int] = None


class ScoreDecoder(nn.Module):
    def __init__(self, n_embd: int, d_inner: Optional[int] = None):
        super().__init__()
        d_inner = d_inner or n_embd
        self.d_inner = d_inner
        
        # Base and affect layers for time-dependent computation
        self.base_layer = nn.Linear(n_embd, d_inner, bias=True)
        self.affect_layer = nn.Sequential(
            nn.Linear(n_embd, d_inner, bias=True),
            nn.Tanh()
        )
        
        # Intensity output layer
        self.intensity_layer = nn.Sequential(
            nn.Linear(d_inner, 1, bias=True),
            nn.Softplus(beta=1.0)
        )
        
        # Cached values from forward pass
        self.affect: Optional[torch.Tensor] = None
        self.base: Optional[torch.Tensor] = None
    
    def forward(self, hidden: torch.Tensor) -> None:
        self.affect = self.affect_layer(hidden)
        self.base = self.base_layer(hidden)
    
    def get_score(
        self,
        t: torch.Tensor,
        non_pad_mask: torch.Tensor
    ) -> torch.Tensor:
        if self.affect is None or self.base is None:
            raise RuntimeError("Must call forward() before get_score()")
        
        if t.ndim == 2:
            t = t.unsqueeze(2)
        assert t.ndim == 3, f"Expected t.ndim == 3, got {t.ndim}"
        
        if not t.requires_grad:
            t = t.clone().detach().requires_grad_(True)

        intensity = torch.tanh(
            self.affect[:, :-1, None, :] * t.unsqueeze(3) + 
            self.base[:, :-1, None, :]
        )
        
        intensity = self.intensity_layer(intensity).squeeze(3)  # (batch, seq_len-1, n_samples)
        intensity = intensity * non_pad_mask[:, 1:, :]
        
        intensity_log = (intensity + 1e-10).log() * non_pad_mask[:, 1:, :]
        
        intensity_grad_t = torch.autograd.grad(
            intensity_log.sum(), t, retain_graph=True, create_graph=True
        )[0] * non_pad_mask[:, 1:, :]
        
        score = intensity_grad_t - intensity
        
        return score


class ScoreDecoderDirect(nn.Module):    
    def __init__(self, n_embd: int, d_inner: Optional[int] = None):
        super().__init__()
        d_inner = d_inner or n_embd
        self.d_inner = d_inner
        
        self.base_layer = nn.Linear(n_embd, d_inner, bias=True)
        self.affect_layer = nn.Sequential(
            nn.Linear(n_embd, d_inner, bias=True),
            nn.Tanh()
        )
        self.score_layer = nn.Sequential(
            nn.Linear(d_inner, 1, bias=True),
            nn.Softplus(beta=1.0)
        )
        
        self.affect: Optional[torch.Tensor] = None
        self.base: Optional[torch.Tensor] = None
    
    def forward(self, hidden: torch.Tensor) -> None:
        self.affect = self.affect_layer(hidden)
        self.base = self.base_layer(hidden)
    
    def get_score(
        self,
        t: torch.Tensor,
        non_pad_mask: torch.Tensor
    ) -> torch.Tensor:
        if self.affect is None or self.base is None:
            raise RuntimeError("Must call forward() before get_score()")
        
        if t.ndim == 2:
            t = t.unsqueeze(2)
        assert t.ndim == 3
        
        # Direct score computation
        score = torch.tanh(
            self.affect[:, :-1, None, :] * t.unsqueeze(3) +
            self.base[:, :-1, None, :]
        )
        score = self.score_layer(score).squeeze(3)
        score = score * non_pad_mask[:, 1:, :]
        
        return score


class TypePredictorWithTime(nn.Module):
    
    def __init__(self, n_embd: int, vocab_size: int):
        super().__init__()
        d_inner = n_embd // 2
        
        self.base_layer = nn.Linear(n_embd, d_inner, bias=True)
        self.affect_layer = nn.Sequential(
            nn.Linear(n_embd, d_inner, bias=True),
            nn.Tanh()
        )
        self.classifier = nn.Linear(d_inner, vocab_size, bias=True)
        
        self.affect: Optional[torch.Tensor] = None
        self.base: Optional[torch.Tensor] = None
    
    def forward(self, hidden: torch.Tensor) -> None:
        self.affect = self.affect_layer(hidden)
        self.base = self.base_layer(hidden)
    
    def get_type(
        self,
        t: torch.Tensor,
        non_pad_mask: torch.Tensor
    ) -> torch.Tensor:
        if self.affect is None or self.base is None:
            raise RuntimeError("Must call forward() before get_type()")
        
        if t.ndim == 2:
            t = t.unsqueeze(2)
        assert t.ndim == 3
        
        out = torch.tanh(
            self.affect[:, :-1, None, :] * t.unsqueeze(3) +
            self.base[:, :-1, None, :]
        )
        
        # Project to vocabulary
        out = self.classifier(out)  # (batch, seq_len-1, n_samples, vocab_size)
        out = out * non_pad_mask[:, 1:, :, None]
        
        return out


class DelphiSMURF(Delphi):
    generation_count = 0

    def _defer_parameter_report(self):
        return True

    def clear_gen_cnt(self):
        self.generation_count = 0
    
    def __init__(self, config: DelphiSMURFConfig):
        super().__init__(config)
        self.config: DelphiSMURFConfig = config
        
        # self.type_predictor = TypePredictorWithTime(
        #     config.n_embd, config.vocab_size
        # )
        self.num_event_types = config.vocab_size - 1
        self.type_predictor = TypePredictorWithTime(
            config.n_embd,
            self.num_event_types,
        )
        
        # Score decoder for time prediction
        if config.parametrize == "intensity":
            self.score_decoder = ScoreDecoder(config.n_embd, config.n_embd)
        else:  # "score"
            self.score_decoder = ScoreDecoderDirect(config.n_embd, config.n_embd)
        
        # Initialize weights for new modules
        self.type_predictor.apply(self._init_weights)
        self.score_decoder.apply(self._init_weights)
        
        print(f"number of parameters: {self.get_num_params()}")
    
    def transformer_forward(
        self,
        idx: torch.Tensor,
        age: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run the transformer backbone and return hidden states.
        
        Args:
            idx: Token indices, shape (batch, seq_len)
            age: Age/time values, shape (batch, seq_len)
        
        Returns:
            Tuple of (hidden_states, attention_mask)
            - hidden_states: shape (batch, seq_len, n_embd)
            - attn_mask: shape (batch, 1, seq_len, seq_len)
        """
        device = idx.device
        b, t = idx.size()

        # torch.cuda.synchronize()
        # print("idx min/max", idx.min().item(), idx.max().item())
        # print("embedding size", self.transformer.wte.num_embeddings)
        # print("age finite", torch.isfinite(age[age > -5]).all().item())
        # torch.cuda.synchronize()
        # tok_emb = self.transformer.wte(idx)
        torch.cuda.synchronize()
        
        # Embeddings
        tok_emb = self.transformer.wte(idx)
        age_emb = self.transformer.wae(age.unsqueeze(-1))
        
        x = self.transformer.token_drop(tok_emb) * (1 - self.config.token_dropout)
        x = x + age_emb
        x = self.transformer.drop(x)
        
        # Build attention mask
        attn_mask = (idx > 0).view(b, 1, 1, t) * (idx > 0).view(b, 1, t, 1)
        attn_mask = attn_mask * (torch.tril(torch.ones(t, t, device=device))[None, None, :, :] > 0)
        attn_mask = attn_mask + (idx == 0).view(b, 1, 1, t) * torch.diag(torch.ones(t, device=device)) > 0
        attn_mask = attn_mask * (torch.tril(torch.ones(t, t, device=device))[None, None, :, :] > 0)
        
        # Transformer blocks
        for block in self.transformer.h:
            x, _ = block(x, attn_mask)
        
        x = self.transformer.ln_f(x)
        
        return x, attn_mask
    
    def get_non_pad_mask(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Get non-padding mask from token indices.
        
        Args:
            idx: Token indices, shape (batch, seq_len)
        
        Returns:
            Non-padding mask, shape (batch, seq_len, 1)
        """
        return (idx > 0).float().unsqueeze(-1)
    
    def forward(
        self,
        idx: torch.Tensor,
        age: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        targets_age: Optional[torch.Tensor] = None,
        time_gap: Optional[torch.Tensor] = None,
        validation_loss_mode: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        """
        Forward pass for DelphiSMURF.
        
        Training mode (targets provided):
        - Computes denoising score matching loss for time prediction
        - Computes cross-entropy loss for type prediction
        
        Inference mode (no targets):
        - Returns logits from time-dependent type predictor using mean time
        
        Args:
            idx: Input token indices, shape (batch, seq_len)
            age: Input ages, shape (batch, seq_len)
            targets: Target token indices, shape (batch, seq_len), optional
            targets_age: Target ages, shape (batch, seq_len), optional
            time_gap: Precomputed time gap between events, shape (batch, seq_len-1), optional
                      If provided, uses this directly instead of computing from targets_age - age.
                      This follows the smurf_thp data format.
            validation_loss_mode: Whether in validation mode
        
        Returns:
            Tuple of (logits, loss_dict, attention_weights)
        """
        device = idx.device
        b, t = idx.size()
        
        # 1. Transformer forward pass
        hidden, attn_mask = self.transformer_forward(idx, age)
        non_pad_mask = self.get_non_pad_mask(idx)
        
        # 2. Cache affect/base for type predictor and score decoder
        self.type_predictor(hidden)
        self.score_decoder(hidden)
        
        # Placeholder for attention (for compatibility)
        att = torch.zeros(self.config.n_layer, b, self.config.n_head, t, t, device=device)
        
        if time_gap is not None:
            dt = time_gap * non_pad_mask[:, 1:].squeeze(-1)
        else:
            dt = dt[:, 1:]  # Shift to predict next event's time gap
            dt = dt * non_pad_mask[:, 1:].squeeze(-1)
        
        # --- Score Matching Loss ---
        if self.config.add_noise == "denoise":
            # Use fewer noise samples during evaluation to save memory
            # Full num_noise for training, reduced for validation
            # num_noise = self.config.num_noise if not validation_loss_mode else min(10, self.config.num_noise)
            num_noise = self.config.num_noise
            # Add noise for denoising score matching
            noise = self.config.var_noise * torch.randn(
                *dt.size(), num_noise, device=device
            )
            t_noisy = dt[:, :, None] + noise  # (batch, seq_len-1, num_noise)
            # Handle truncated noise (ensure positive times)
            if self.config.noise_type == "truncated":
                while ((t_noisy < 0) * non_pad_mask[:, 1:, :].bool()).any():
                    noise_new = self.config.var_noise * torch.randn(
                        *dt.size(), num_noise, device=device
                    )
                    t_noisy_new = dt[:, :, None] + noise_new
                    t_noisy[t_noisy < 0] = t_noisy_new[t_noisy < 0]
            
            # Make t_noisy require grad for intensity parametrization
            if self.config.parametrize == "intensity":
                t_noisy = t_noisy.clone().detach().requires_grad_(True)
            
            # Compute score
            score = self.score_decoder.get_score(t_noisy, non_pad_mask)
            
            # ============ DEBUG: Gradient hooks ============
            # Store debug info as instance attributes for inspection
            self._debug_grads = {}
            def save_grad(name):
                def hook(grad):
                    self._debug_grads[name] = grad.clone() if grad is not None else None
                return hook
            
            # Register hooks on key tensors
            score.register_hook(save_grad('score'))
            t_noisy.register_hook(save_grad('t_noisy'))
            
            # Store forward values for inspection
            self._debug_forward = {
                'score': score.detach().clone(),
                't_noisy': t_noisy.detach().clone(),
                'dt': dt.detach().clone(),
                'score_requires_grad': score.requires_grad,
                'score_grad_fn': str(score.grad_fn),
                'affect_grad_fn': str(self.score_decoder.affect.grad_fn) if self.score_decoder.affect is not None else None,
            }
            # ============ END DEBUG ============
            
            # Denoising score matching objective
            obj = get_obj_denoise(
                dt, t_noisy, score, self.config.var_noise, non_pad_mask
            )
            # loss_score = obj.sum() / non_pad_mask[:, 1:].sum()
            type_prediction = self.type_predictor.get_type(t_noisy, non_pad_mask)
        else:
            # Non-denoising score matching (not recommended)
            # t_var = dt.unsqueeze(-1).clone().detach().requires_grad_(True)
            t_var = dt.clone().detach().requires_grad_(True)
            score = self.score_decoder.get_score(t_var, non_pad_mask).squeeze(-1)
            obj = get_obj(t_var, score, non_pad_mask)
            # type_prediction = self.type_predictor.get_type(dt, non_pad_mask).squeeze(2)
            type_prediction = self.type_predictor.get_type(dt, non_pad_mask)
        
        loss = self.compute_loss(obj, idx, type_prediction)
        logits = type_prediction
            
        
        return logits, loss, att
    
    @torch.no_grad()
    def langevin_sample(
        self,
        hidden: torch.Tensor,
        non_pad_mask: torch.Tensor,
        n_samples: Optional[int] = None,
        n_steps: Optional[int] = None,
        step_size: Optional[float] = None,
        t_init: Optional[torch.Tensor] = None,
        apply_denoise: Optional[bool] = None
    ) -> torch.Tensor:
        """
        Sample time using Langevin dynamics.
        
        Langevin dynamics iteratively samples from p(t|H) by:
        t_{k+1} = t_k + step_size/2 * score(t_k) + sqrt(step_size) * z
        
        where z ~ N(0, 1) and score = ∇_t log p(t|H).
        
        Args:
            hidden: Hidden states from transformer, shape (batch, seq_len, n_embd)
            non_pad_mask: Padding mask, shape (batch, seq_len, 1)
            n_samples: Number of samples per position (default: config.n_samples)
            n_steps: Number of Langevin steps (default: config.n_langevin_steps)
            step_size: Langevin step size (default: config.langevin_step)
            t_init: Initial time values (optional)
            apply_denoise: Whether to apply final denoising step (default: config.denoise_step)
        
        Returns:
            Sampled times, shape (batch, seq_len-1, n_samples)
        """
        n_samples = n_samples or self.config.n_samples
        n_steps = n_steps or self.config.n_langevin_steps
        step_size = step_size or self.config.langevin_step
        apply_denoise = apply_denoise if apply_denoise is not None else self.config.denoise_step
        
        batch_size, seq_len = hidden.shape[:2]
        device = hidden.device
        
        # Cache affect and base values
        self.score_decoder(hidden)
        
        # Initialize samples
        if t_init is not None:
            t = t_init
        else:
            # Initialize uniformly between 5th and 95th percentile
            t_05 = self.config.time_quantile_05
            t_95 = self.config.time_quantile_95
            t = torch.rand(batch_size, seq_len - 1, n_samples, device=device)
            t = t * (t_95 - t_05) + t_05
        
        # Langevin dynamics iterations
        # Need to enable grad temporarily for score computation
        with torch.enable_grad():
            for _ in range(n_steps):
                t_var = t.clone().detach().requires_grad_(True)
                score = self.score_decoder.get_score(t_var, non_pad_mask)
                
                # Langevin update
                z = torch.randn_like(t)
                t = t + 0.5 * step_size * score.detach() + math.sqrt(step_size) * z
                
                # Clamp to positive values
                # t = torch.clamp(t, min=1e-5)
        
        # Optional denoising step
        if apply_denoise:
            with torch.enable_grad():
                t_var = t.clone().detach().requires_grad_(True)
                score = self.score_decoder.get_score(t_var, non_pad_mask)
                t = t + (self.config.var_noise ** 2) * score.detach()
                # t = torch.clamp(t, min=1e-5)
        
        return t
    
    @torch.no_grad()
    def predict_time(
        self,
        idx: torch.Tensor,
        age: torch.Tensor,
        n_samples: int = 100,
        n_steps: int = 300,
        step_size: Optional[float] = None,
        apply_denoise: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict time gaps using Langevin sampling.
        
        Args:
            idx: Input token indices, shape (batch, seq_len)
            age: Input ages, shape (batch, seq_len)
            n_samples: Number of Langevin samples
            n_steps: Number of Langevin steps
            step_size: Langevin step size (default: config.langevin_step)
            apply_denoise: Whether to apply final denoising step
        
        Returns:
            Tuple of (t_samples, t_mean):
            - t_samples: All sampled times, shape (batch, seq_len-1, n_samples)
            - t_mean: Mean of samples, shape (batch, seq_len-1)
        """
        step_size = step_size or self.config.langevin_step
        
        # Run transformer forward pass
        hidden, _ = self.transformer_forward(idx, age)
        non_pad_mask = self.get_non_pad_mask(idx)
        
        # Run Langevin sampling (outputs normalized time)
        t_samples = self.langevin_sample(
            hidden, non_pad_mask,
            n_samples=n_samples,
            n_steps=n_steps,
            step_size=step_size,
            apply_denoise=apply_denoise
        )
        
        # The langevin results t_samples seems to be in the actual distribution (denormalized)
        
        # Compute mean prediction
        t_mean = t_samples.mean(dim=2)  # (batch, seq_len-1)
        
        return t_samples, t_mean
    
    def _encode_age_for_transformer(
        self,
        age: torch.Tensor,
        age_is_normalized: bool = True,
    ) -> torch.Tensor:
        if age_is_normalized:
            return age
        mode = getattr(self.config, "normalize", "normal")
        if mode == "normal":
            time_mean = max(float(getattr(self.config, "time_mean", 1.0)), 1e-8)
            return age / time_mean
        if mode == "log":
            return age
        raise ValueError(f"Unsupported normalize mode: {mode}")
    
    def _age_to_raw_for_generation(
        self,
        age: torch.Tensor,
        age_is_normalized: bool,
    ) -> torch.Tensor:
        if not age_is_normalized:
            return age

        mode = self._get_time_mode()
        if mode == "normal":
            return age * float(self.config.time_mean)
        if mode == "log":
            return age

        raise ValueError(f"Unsupported normalize mode: {mode}")

    def _type_loss_smurf(self, prediction: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        truth = targets[:, 1:] - 1  # official SMURF convention

        if prediction.ndim == 4:
            truth = truth.unsqueeze(2).expand(-1, -1, prediction.size(2))  # (B, L-1, N)

        loss = F.cross_entropy(
            prediction.reshape(-1, prediction.size(-1)),
            truth.reshape(-1),
            ignore_index=-1,
            reduction="none",
        ).view(*truth.shape)

        if prediction.ndim == 4:
            loss = loss.mean(dim=2)  # mean over noise/sample dimension

        return loss.sum()
    
    def _get_time_mode(self):
        return getattr(self.config, "normalize", "normal")

    def _gap_model_to_raw(self, t_model: torch.Tensor) -> torch.Tensor:
        mode = self._get_time_mode()
        if mode == "normal":
            # t_model = gap / time_mean
            return torch.clamp(t_model, min=0.0) * self.config.time_mean
        elif mode == "log":
            log_mean = float(getattr(self.config, "log_mean", 0.0))
            log_std = max(float(getattr(self.config, "log_std", 1.0)), 1e-8)
            # t_model = (log(gap / time_mean) - log_mean) / log_std
            return torch.exp(t_model * log_std + log_mean) * self.config.time_mean
        else:
            raise ValueError(f"Unsupported normalize mode: {mode}")

    def _gap_raw_to_model(self, gap_raw: torch.Tensor) -> torch.Tensor:
        mode = self._get_time_mode()
        if mode == "normal":
            return torch.clamp(gap_raw, min=0.0) / self.config.time_mean
        elif mode == "log":
            log_mean = float(getattr(self.config, "log_mean", 0.0))
            log_std = max(float(getattr(self.config, "log_std", 1.0)), 1e-8)
            return (torch.log(torch.clamp(gap_raw, min=1e-8) / self.config.time_mean + 1e-9) - log_mean) / log_std
        else:
            raise ValueError(f"Unsupported normalize mode: {mode}")

    def _transition_logprobs_from_logits(
        self,
        idx: torch.Tensor,
        age: torch.Tensor,
        logits: torch.Tensor,
        mask_time: float = -10000.0,
    ) -> torch.Tensor:
        """return log-prob for each transition,shape = (B, T-1), in which logits[:, j, :] predicts idx[:, j+1]"""
        if idx.shape[1] <= 1:
            return torch.zeros(idx.shape[0], 0, device=idx.device, dtype=logits.dtype,)

        # logits[:, :-1, :] -> predict idx[:, 1:]
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)

        # SMURF logits are 0-indexed, tokens are 1-indexed
        target = torch.clamp(idx[:, 1:] - 1, 0, self.num_event_types - 1)

        token_logprobs = log_probs.gather(
            dim=-1,
            index=target.unsqueeze(-1)
        ).squeeze(-1)

        valid_mask = (idx[:, 1:] > 0) & (age[:, 1:] > mask_time)
        token_logprobs = token_logprobs.masked_fill(~valid_mask, 0.0)

        return token_logprobs

    def compute_loss(
        self,
        obj: torch.Tensor,
        targets: torch.Tensor,
        prediction: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        event_loss = obj.sum()
        pred_loss = self._type_loss_smurf(prediction, targets)
        total_loss = self.config.loss_lambda * event_loss + pred_loss

        denom = (targets[:, 1:] > 0).sum().clamp_min(1)

        return {
            "loss_ce": pred_loss / denom,
            "loss_score": event_loss / denom,
            "loss_dt": torch.zeros((), device=targets.device),
            "loss_dt_mse": torch.zeros((), device=targets.device),
            "loss_total": total_loss,
        }

    
    def _estimate_time_probability_from_samples(
        self,
        gap_raw_samples: torch.Tensor,
        chosen_gap_raw: torch.Tensor,
        time_prob_window: Optional[float] = None,
        time_prob_window_scale: float = 0.5,
        min_time_prob_window: float = 1.0,
        eps: float = 1e-12,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        samples = gap_raw_samples.float()
        chosen = chosen_gap_raw.float()

        valid = torch.isfinite(samples) & (samples >= 0.0)
        valid_count = valid.sum(dim=-1).clamp_min(1)

        safe_samples = torch.where(valid, samples, torch.zeros_like(samples))
        sample_mean = safe_samples.sum(dim=-1) / valid_count

        centered = torch.where(
            valid,
            samples - sample_mean.unsqueeze(-1),
            torch.zeros_like(samples),
        )
        sample_var = (centered ** 2).sum(dim=-1) / valid_count
        sample_std = torch.sqrt(sample_var + eps)

        if time_prob_window is None:
            window = torch.clamp(
                sample_std * float(time_prob_window_scale),
                min=float(min_time_prob_window),
            )
        else:
            window = torch.full_like(chosen, float(time_prob_window))

        hit = valid & (
            torch.abs(samples - chosen.unsqueeze(-1)) <= window.unsqueeze(-1)
        )
        time_prob = hit.float().sum(dim=-1) / valid_count
        time_prob = time_prob.clamp_min(eps).clamp_max(1.0)

        return time_prob, window
    
    @torch.no_grad()
    def generate_mcts(
        self,
        idx: torch.Tensor,
        age: torch.Tensor,
        max_new_tokens: int = 100,
        max_age: float = 85 * 365.25,
        no_repeat: bool = False,
        termination_tokens: Optional[list] = None,
        top_k: Optional[int] = None,
        temperature: float = 0.0,
        use_langevin: bool = True,
        n_samples: int = LANGEVIN_SAMPLES_PER_CANDIDATE,
        n_langevin_steps: int = LANGEVIN_STEPS,
        time_uncertainty_beta: float = TIME_UNCERTAINTY_BETA,
        atomic_candidate_pool_size: int = ATOMIC_CANDIDATE_POOL_SIZE,
        time_uncertainty_calibration: Optional[dict] = None,
        langevin_step: Optional[float] = None,
        apply_denoise: bool = True,
        age_is_normalized: bool = True,
        mask_time: float = -10000.0,
        time_prob_window: Optional[float] = None,
        time_prob_window_scale: float = 0.5,
        min_time_prob_window: float = 1.0,
        return_samples: bool = False,
        device: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        if termination_tokens is None:
            termination_tokens = TERMINATION_TOKENS

        if device is not None:
            idx = idx.to(device)
            age = age.to(device)
        else:
            device = idx.device

        idx = idx.long()
        age = age.float()

        if max_new_tokens == -1:
            max_new_tokens = 128

        if not use_langevin:
            raise NotImplementedError("generate_mcts currently requires Langevin sampling.")

        if time_uncertainty_calibration is None:
            raise ValueError("Training-split time uncertainty calibration is required")
        if n_samples < 2 or atomic_candidate_pool_size < 2:
            raise ValueError("At least two Langevin samples and two candidates are required")

        from model.atomic_prior import atomic_candidate_log_prior

        batch_size = idx.size(0)
        eps = 1e-12
        time_mode = self._get_time_mode()
        langevin_step = float(langevin_step or self.config.langevin_step)

        termination_tokens_t = torch.tensor(
            termination_tokens,
            dtype=torch.long,
            device=device,
        )

        if age_is_normalized:
            if time_mode == "normal":
                valid_age = age > GAP_THRESH
                age_raw = torch.where(valid_age, age * float(self.config.time_mean), age)
            elif time_mode == "log":
                age_raw = age
            else:
                raise ValueError(f"Unsupported normalize mode: {time_mode}")
        else:
            age_raw = age

        def raw_age_to_model_age(x_raw: torch.Tensor) -> torch.Tensor:
            if time_mode == "normal":
                valid_age = x_raw > GAP_THRESH
                return torch.where(valid_age, x_raw / float(self.config.time_mean), x_raw)
            elif time_mode == "log":
                return x_raw
            else:
                raise ValueError(f"Unsupported normalize mode: {time_mode}")

        def last_nonpad_positions(tokens: torch.Tensor) -> torch.Tensor:
            valid = tokens > 0
            pos = torch.arange(tokens.size(1), device=tokens.device).view(1, -1)
            last_pos = torch.where(valid, pos, torch.zeros_like(pos)).max(dim=1).values
            return last_pos.long()

        def score_from_last_state(
            t_model: torch.Tensor,
            score_affect_last: torch.Tensor,
            score_base_last: torch.Tensor,
        ) -> torch.Tensor:
            if t_model.ndim == 1:
                t_model = t_model.unsqueeze(1)

            if self.config.parametrize == "intensity":
                t_var = t_model.clone().detach().requires_grad_(True)

                h = torch.tanh(
                    score_affect_last[:, None, :] * t_var[:, :, None]
                    + score_base_last[:, None, :]
                )
                intensity = self.score_decoder.intensity_layer(h).squeeze(-1)
                intensity_log = torch.log(intensity + 1e-10)

                grad_t = torch.autograd.grad(
                    intensity_log.sum(),
                    t_var,
                    retain_graph=False,
                    create_graph=False,
                )[0]

                return grad_t - intensity

            else:
                h = torch.tanh(
                    score_affect_last[:, None, :] * t_model[:, :, None]
                    + score_base_last[:, None, :]
                )
                return self.score_decoder.score_layer(h).squeeze(-1)

        def sample_next_gap_from_last_state(
            hidden: torch.Tensor,
            last_pos: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            self.score_decoder(hidden)

            b_idx = torch.arange(batch_size, device=device)
            score_affect_last = self.score_decoder.affect[b_idx, last_pos]  # (B, D)
            score_base_last = self.score_decoder.base[b_idx, last_pos]      # (B, D)

            t_05 = float(getattr(self.config, "time_quantile_05", 0.0))
            t_95 = float(getattr(self.config, "time_quantile_95", 1.0))
            if not torch.isfinite(torch.tensor(t_05)):
                t_05 = 0.0
            if not torch.isfinite(torch.tensor(t_95)) or t_95 <= t_05:
                t_95 = t_05 + 1.0

            t_model = torch.rand(
                batch_size, atomic_candidate_pool_size * n_samples, device=device
            )
            t_model = t_model * (t_95 - t_05) + t_05

            with torch.enable_grad():
                for _ in range(n_langevin_steps):
                    score = score_from_last_state(
                        t_model,
                        score_affect_last,
                        score_base_last,
                    )
                    noise = torch.randn_like(t_model)
                    t_model = (
                        t_model
                        + 0.5 * langevin_step * score.detach()
                        + math.sqrt(langevin_step) * noise
                    )

                if apply_denoise:
                    score = score_from_last_state(
                        t_model,
                        score_affect_last,
                        score_base_last,
                    )
                    t_model = t_model + float(self.config.var_noise) ** 2 * score.detach()

            t_model = torch.nan_to_num(t_model, nan=0.0, posinf=t_95, neginf=t_05)

            if time_mode == "normal":
                t_model_pos = torch.clamp(t_model, min=0.0)
                gap_raw_samples = (t_model_pos * float(self.config.time_mean)).reshape(
                    batch_size, atomic_candidate_pool_size, n_samples
                )
                next_gap_raw = gap_raw_samples.mean(dim=-1)  # (B, M)
                t_query_model = next_gap_raw / float(self.config.time_mean)

            elif time_mode == "log":
                gap_raw_samples = self._gap_model_to_raw(t_model)
                gap_raw_samples = torch.nan_to_num(
                    gap_raw_samples,
                    nan=0.0,
                    posinf=float(max_age),
                    neginf=0.0,
                )
                gap_raw_samples = torch.clamp(gap_raw_samples, min=0.0)

                gap_raw_samples = gap_raw_samples.reshape(
                    batch_size, atomic_candidate_pool_size, n_samples
                )
                next_gap_raw = gap_raw_samples.mean(dim=-1)
                t_query_model = self._gap_raw_to_model(next_gap_raw)

            else:
                raise ValueError(f"Unsupported normalize mode: {time_mode}")

            return next_gap_raw, t_query_model, gap_raw_samples

        def type_logits_from_last_state(
            hidden: torch.Tensor,
            last_pos: torch.Tensor,
            t_query_model: torch.Tensor,
        ) -> torch.Tensor:
            self.type_predictor(hidden)

            b_idx = torch.arange(batch_size, device=device)
            affect_last = self.type_predictor.affect[b_idx, last_pos]  # (B, D)
            base_last = self.type_predictor.base[b_idx, last_pos]      # (B, D)

            h_type = torch.tanh(
                affect_last[:, None, :] * t_query_model[:, :, None]
                + base_last[:, None, :]
            )
            logits = self.type_predictor.classifier(h_type).squeeze(1)  # (B, C)
            return logits

        def mask_class_logits_by_token_ids(
            class_logits: torch.Tensor,
            token_ids,
        ) -> torch.Tensor:
            if token_ids is None:
                return class_logits

            if isinstance(token_ids, torch.Tensor):
                tok = token_ids.to(device=device, dtype=torch.long)
            else:
                tok = torch.tensor(token_ids, device=device, dtype=torch.long)

            C = class_logits.size(-1)
            valid = (tok >= 1) & (tok <= C)
            class_ids = tok[valid] - 1

            if class_ids.numel() > 0:
                class_logits[:, class_ids] = -torch.inf

            return class_logits

        def apply_no_repeat_class_mask(
            class_logits: torch.Tensor,
            tokens: torch.Tensor,
        ) -> torch.Tensor:
            C = class_logits.size(-1)
            for b in range(tokens.size(0)):
                seen_tokens = tokens[b]
                seen_tokens = seen_tokens[(seen_tokens >= 1) & (seen_tokens <= C)]
                if seen_tokens.numel() > 0:
                    seen_classes = seen_tokens - 1
                    class_logits[b, seen_classes] = -torch.inf
            return class_logits

        if termination_tokens_t.numel() > 0:
            finished = torch.isin(idx, termination_tokens_t).any(dim=1)
        else:
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        initial_length = idx.size(1)

        token_probs_list = []
        time_probs_list = []
        action_logprobs_list = []
        uncertainty_list = []
        if return_samples:
            time_samples_list = []

        for _ in range(max_new_tokens):
            if finished.all():
                break

            age_model = raw_age_to_model_age(age_raw)

            hidden, _ = self.transformer_forward(idx, age_model)

            last_pos = last_nonpad_positions(idx)
            last_age_raw = age_raw[
                torch.arange(batch_size, device=device),
                last_pos,
            ]

            # ---- next gap from last hidden state ----
            candidate_gap_raw, t_query_model, candidate_gap_samples = sample_next_gap_from_last_state(
                hidden=hidden,
                last_pos=last_pos,
            )
            next_gap_raw = candidate_gap_raw[:, 0]
            gap_raw_samples = candidate_gap_samples[:, 0, :]
            next_age_raw = last_age_raw + next_gap_raw

            # Diagnostic window mass; search prior comes from equation (2).
            valid_time_sample = torch.isfinite(gap_raw_samples) & (gap_raw_samples >= 0.0)
            samples_for_prob = gap_raw_samples.masked_fill(~valid_time_sample, float("nan"))

            time_prob, _ = self._estimate_time_probability_from_samples(
                gap_raw_samples=samples_for_prob,
                chosen_gap_raw=next_gap_raw,
                time_prob_window=time_prob_window,
                time_prob_window_scale=time_prob_window_scale,
                min_time_prob_window=min_time_prob_window,
                eps=eps,
            )

            # ---- next token logits from last hidden state ----
            candidate_logits = type_logits_from_last_state(
                hidden=hidden,
                last_pos=last_pos,
                t_query_model=t_query_model,
            )  # (B, M, C), class logits

            num_classes = candidate_logits.size(-1)
            flat_logits = candidate_logits.reshape(-1, num_classes)
            prior_logits = flat_logits.clone()

            # Correct masking: ignore_tokens are TOKEN ids, logits are CLASS ids.
            prior_logits = mask_class_logits_by_token_ids(
                prior_logits,
                getattr(self.config, "ignore_tokens", None),
            )

            if no_repeat:
                prior_logits = apply_no_repeat_class_mask(
                    prior_logits,
                    idx.repeat_interleave(atomic_candidate_pool_size, dim=0),
                )

            finite_row = torch.isfinite(prior_logits).any(dim=-1)
            if not finite_row.all():
                fallback_logits = flat_logits.clone()
                fallback_logits = mask_class_logits_by_token_ids(
                    fallback_logits,
                    getattr(self.config, "ignore_tokens", None),
                )
                prior_logits = torch.where(
                    finite_row[:, None],
                    prior_logits,
                    fallback_logits,
                )

            if not torch.isfinite(prior_logits).any(dim=-1).all():
                raise ValueError("No valid event type remains after applying masks")

            prior_probs = F.softmax(prior_logits, dim=-1)
            sample_logits = prior_logits.clone()
            if top_k is not None and top_k > 0:
                k = min(int(top_k), sample_logits.size(-1))
                topv, _ = torch.topk(sample_logits, k=k, dim=-1)
                pivot = topv[:, -1].unsqueeze(1)
                sample_logits = sample_logits.masked_fill(sample_logits < pivot, -torch.inf)

            if temperature is not None and temperature > 0:
                sample_probs = F.softmax(sample_logits / float(temperature), dim=-1)
                candidate_classes = torch.multinomial(sample_probs, num_samples=1).squeeze(1)
            else:
                sample_probs = prior_probs
                candidate_classes = torch.argmax(sample_probs, dim=-1)

            candidate_type_prob = prior_probs.gather(
                dim=-1,
                index=candidate_classes.unsqueeze(1),
            ).squeeze(1).clamp_min(eps).reshape(batch_size, atomic_candidate_pool_size)

            prefix_keys = [
                (tuple(idx[b].tolist()), tuple(age_raw[b].tolist()))
                for b in range(batch_size)
            ]
            prefix_ids = [prefix_keys.index(key) for key in prefix_keys]
            candidate_log_prior, candidate_uncertainty = atomic_candidate_log_prior(
                type_logprobs=torch.log(candidate_type_prob),
                gap_samples=candidate_gap_samples,
                beta=time_uncertainty_beta,
                calibration=time_uncertainty_calibration,
                clip_max=TIME_UNCERTAINTY_CLIP_MAX,
                eps=TIME_UNCERTAINTY_EPS,
                prefix_group_ids=prefix_ids,
            )
            next_class = candidate_classes.reshape(batch_size, atomic_candidate_pool_size)[:, 0]
            token_prob = candidate_type_prob[:, 0]
            atomic_logprob = candidate_log_prior[:, 0]
            uncertainty = candidate_uncertainty[:, 0]

            next_token = next_class + 1

            too_old = next_age_raw > float(max_age)

            next_token = torch.where(
                finished | too_old,
                torch.zeros_like(next_token),
                next_token,
            )
            next_age_raw = torch.where(
                finished | too_old,
                torch.full_like(next_age_raw, float(mask_time)),
                next_age_raw,
            )
            next_gap_raw = torch.where(
                finished | too_old,
                torch.zeros_like(next_gap_raw),
                next_gap_raw,
            )

            token_prob = torch.where(
                finished | too_old,
                torch.ones_like(token_prob),
                token_prob,
            )
            time_prob = torch.where(
                finished | too_old,
                torch.ones_like(time_prob),
                time_prob,
            )

            action_logprob = torch.where(
                finished | too_old,
                torch.zeros_like(atomic_logprob),
                atomic_logprob,
            )
            uncertainty = torch.where(
                finished | too_old,
                torch.zeros_like(uncertainty),
                uncertainty,
            )

            idx = torch.cat([idx, next_token.unsqueeze(1)], dim=1)
            age_raw = torch.cat([age_raw, next_age_raw.unsqueeze(1)], dim=1)

            valid_tokens = (next_token > 0).sum().item()
            self.generation_count += valid_tokens

            token_probs_list.append(token_prob)
            time_probs_list.append(time_prob)
            action_logprobs_list.append(action_logprob)
            uncertainty_list.append(uncertainty)

            if return_samples:
                time_samples_list.append(gap_raw_samples)

            if termination_tokens_t.numel() > 0:
                is_termination = torch.isin(next_token, termination_tokens_t)
            else:
                is_termination = torch.zeros_like(next_token, dtype=torch.bool)

            finished = finished | is_termination | too_old

        # Every probability field is transition-aligned: [B, T-1].
        length = max(idx.size(1) - 1, 0)
        full_type_logprobs = torch.zeros(batch_size, length, device=device)
        full_time_gap_logprobs = torch.zeros_like(full_type_logprobs)
        full_step_logprobs = torch.zeros_like(full_type_logprobs)
        full_uncertainties = torch.zeros_like(full_type_logprobs)

        if action_logprobs_list:
            # prefix_last -> gen_1, gen_1 -> gen_2, ...
            start = initial_length - 1
            end = start + len(action_logprobs_list)
            full_type_logprobs[:, start:end] = torch.log(torch.stack(token_probs_list, dim=1).clamp_min(eps))
            full_time_gap_logprobs[:, start:end] = torch.log(torch.stack(time_probs_list, dim=1).clamp_min(eps))
            full_step_logprobs[:, start:end] = torch.stack(action_logprobs_list, dim=1)
            full_uncertainties[:, start:end] = torch.stack(uncertainty_list, dim=1)

        output = {
            "events": idx,
            "ages": age_raw,
            "type_logprobs": full_type_logprobs,
            "time_gap_logprobs": full_time_gap_logprobs,
            "step_logprobs": full_step_logprobs,
            "time_uncertainties": full_uncertainties,
        }
        if return_samples:
            output["time_samples"] = (
                torch.stack(time_samples_list, dim=1) if time_samples_list
                else torch.empty(batch_size, 0, n_samples, device=device)
            )
        return output

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        age: torch.Tensor,
        max_new_tokens: int = 100,
        max_age: float = 85 * 365.25,
        no_repeat: bool = True,
        termination_tokens: Optional[list] = None,
        top_k: Optional[int] = None,
        temperature: float = 0,
        use_langevin: bool = True,
        n_samples: int = 100,
        n_langevin_steps: int = 100,
        age_is_normalized: bool = True,
        device: str = 'cuda'
    ):
        if termination_tokens is None:
            # warnings.warn('Using default termination token 1269.')
            termination_tokens = TERMINATION_TOKENS
        batch_size = idx.shape[0]
        
        termination_tokens = torch.tensor(termination_tokens, dtype=torch.int64, device=idx.device)
        mask_time = -10000
        
        if max_new_tokens == -1:
            max_new_tokens = 128

        # Normalize max_age for comparison (max_age is always in raw days)
        time_mode = self._get_time_mode()
        if time_mode == "normal":
            if not age_is_normalized:
                valid_mask = age > GAP_THRESH
                age = torch.where(valid_mask, age / self.config.time_mean, age)
            max_age_model = max_age / self.config.time_mean
        elif time_mode == "log":
            max_age_model = max_age
        else:
            raise ValueError(f"Unsupported normalize mode: {time_mode}")
        
        all_t_samples = torch.zeros(
            (max_new_tokens, batch_size, n_samples),
            dtype=torch.float32,
            device=idx.device
        )
        gen_token_len = 0
        
        for i in range(max_new_tokens):
            # Forward pass through transformer
            hidden, attn_mask = self.transformer_forward(idx, age)
            non_pad_mask = self.get_non_pad_mask(idx)
            
            if use_langevin:
                # Sample time using Langevin dynamics (outputs NORMALIZED time)
                t_samples = self.langevin_sample(
                    hidden, non_pad_mask,
                    n_samples=n_samples,
                    n_steps=n_langevin_steps
                )
                
                t_last_model = t_samples[:, -1, :]  # (batch, n_samples) normalized
                all_t_samples[i] = t_last_model
                gen_token_len += 1

                # 以及用于条件查询的归一化时间 (t_query_last)
                if time_mode == "normal":
                    t_next_for_age = torch.clamp(t_last_model, min=0.0).mean(dim=1, keepdim=True)
                    t_query_last = t_next_for_age  # normal 模式下两者尺度一致
                else:
                    # internal age is raw here
                    t_last_raw = self._gap_model_to_raw(t_last_model)
                    t_next_for_age = t_last_raw.mean(dim=1, keepdim=True) 
                    t_query_last = self._gap_raw_to_model(t_next_for_age)  # 转换回模型尺度以供预测器使用
                
                t_query = torch.zeros(
                    batch_size, 
                    idx.size(1) - 1, 
                    1, 
                    device=idx.device, 
                    dtype=t_samples.dtype
                )
                t_query[:, -1, 0] = t_query_last.squeeze(1)
                self.type_predictor(hidden)
                type_logits = self.type_predictor.get_type(t_query, non_pad_mask)
        
                logits = type_logits[:, -1, 0, :]
                
            else:
                raise NotImplementedError("Unimplemented Error for no langevain sample generation")

            # Apply masks
            if self.config.ignore_tokens:
                logits[:, self.config.ignore_tokens] = -torch.inf
            
            # if no_repeat:
            #     fill = idx.clone() - 1
            #     fill[fill == -1] = 0
            #     # fill[fill == 1] = 0
            #     logits = logits.scatter_(1, fill, -torch.inf)
            
            # Apply top-k
            if top_k is not None and top_k > 0:
                k = min(top_k, logits.size(-1))
                val, _ = torch.topk(logits, k)
                pivot = val[:, -1].unsqueeze(1)
                logits[logits < pivot] = -torch.inf
            
            # Apply temperature and sample
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1) + 1  # (batch, 1)
            else:
                probs = torch.softmax(logits, dim=-1)
                idx_next = probs.argmax(dim=-1, keepdim=True) + 1  # (batch, 1)
            
            
            # Compute next age (both are normalized, so addition is correct)
            age_next = age[..., [-1]] + t_next_for_age
            
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)
            age = torch.cat((age, age_next), dim=1)
            
            # Check termination (compare normalized ages with normalized max_age)
            has_terminated = torch.isin(idx, termination_tokens).any(-1)
            is_too_old = (age_next > max_age_model).squeeze(-1)
            if torch.logical_or(has_terminated, is_too_old).all():
                break
        
        t_samples = all_t_samples[:gen_token_len].squeeze(1)    # cut-off padding and batch_size=1
        # print(f"t_samples.shape={t_samples.shape}")
        # Post-processing: mask out tokens after termination
        is_terminated = torch.isin(idx, termination_tokens)
        pad_mask = (torch.cumsum(torch.cumsum(is_terminated, 1).bool().int(), 1) > 1)
        pad_mask = pad_mask + (age > max_age_model)  # Use normalized max_age
        
        # Final forward pass for logits
        hidden, _ = self.transformer_forward(idx, age)
        self.type_predictor(hidden)
        non_pad_mask = self.get_non_pad_mask(idx)

        if idx.shape[1] >= 2:
            age_diff = age[:, 1:] - age[:, :-1]
            if time_mode == "normal":
                gap_model = torch.clamp(age_diff, min=0.0)
            else:
                gap_model = self._gap_raw_to_model(age_diff)

            event_logits = self.type_predictor.get_type(gap_model.unsqueeze(-1), non_pad_mask).squeeze(2)
            event_logits = F.pad(event_logits, (0, 0, 1, 0), value=0)
        else:
            event_logits = torch.zeros(idx.shape[0], idx.shape[1], self.num_event_types, device=device)
        
        idx[pad_mask] = 0
        age[pad_mask] = mask_time
        
        # if no_repeat:
        #     fill = idx.clone() - 1
        #     fill[fill == -1] = 0
        #     event_logits = torch.stack([
        #         event_logits[:, j].scatter_(1, fill[:, :j + 1], -torch.inf)
        #         for j in range(fill.shape[1])
        #     ]).transpose(0, 1)
        # event_logits = event_logits[:, :gen_token_len, :]
        
        if time_mode == "normal":    # always restore
            valid_mask = age != mask_time
            age = torch.where(valid_mask, age * self.config.time_mean, age)
        
        transition_logprobs = self._transition_logprobs_from_logits(
            idx=idx,
            age=age,
            logits=event_logits,
            mask_time=mask_time,
        )

        # return idx, age, event_logits, t_samples
        return idx, age, transition_logprobs, t_samples



def create_delphi_smurf(
    vocab_size: int,
    n_layer: int = 6,
    n_head: int = 8,
    n_embd: int = 512,
    block_size: int = 1024,
    dropout: float = 0.1,
    use_score_matching: bool = True,
    var_noise: float = 0.1,
    num_noise: int = 100,
    langevin_step: float = 3e-2,
    n_langevin_steps: int = 300,
    loss_lambda: float = 3.0,
    **kwargs
) -> DelphiSMURF:
    """
    Factory function to create a DelphiSMURF model.
    
    Args:
        vocab_size: Size of vocabulary (number of event types)
        n_layer: Number of transformer layers
        n_head: Number of attention heads
        n_embd: Embedding dimension
        block_size: Maximum sequence length
        dropout: Dropout rate
        use_score_matching: Whether to use score matching for time prediction
        var_noise: Noise variance for denoising score matching
        num_noise: Number of noise samples
        langevin_step: Step size for Langevin dynamics
        n_langevin_steps: Number of Langevin steps
        loss_lambda: Weight for score matching loss
        **kwargs: Additional config parameters
    
    Returns:
        Configured DelphiSMURF model
    """
    config = DelphiSMURFConfig(
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=block_size,
        dropout=dropout,
        use_score_matching=use_score_matching,
        var_noise=var_noise,
        num_noise=num_noise,
        langevin_step=langevin_step,
        n_langevin_steps=n_langevin_steps,
        loss_lambda=loss_lambda,
        **kwargs
    )
    
    return DelphiSMURF(config)
