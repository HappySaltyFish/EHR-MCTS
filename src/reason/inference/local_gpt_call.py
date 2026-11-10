"""Local DelphiSMURF trajectory generation for EHR search."""

import torch
from model import DelphiSMURFConfig, DelphiSMURF
from reason.inference.lm_call import LanguageModelCallingFunction
from reason.inference.types import TrajectoryGenerationConfig, TrajectoryGenerationResult
from utils import get_time_uncertainty_calibration


class LocalGPTCallerSmurf(LanguageModelCallingFunction):
    """
    Delphi caller for SMURF models (DelphiSMURF).
    Ages are always kept in raw days throughout the MCTS pipeline.
    The model normalizes/denormalizes internally via age_is_normalized=False.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype: torch.dtype = torch.float32,
        lm_step_tag: str = None,
    ):
        super().__init__(lm_step_tag)

        self.checkpoint_path = checkpoint_path
        self.device = device
        self.torch_dtype = torch_dtype

        print(f"Loading DelphiSMURF model: {checkpoint_path}")
        print(f"Device: {device}, dtype: {torch_dtype}")

        checkpoint = torch.load(checkpoint_path, map_location=device)

        self.config = DelphiSMURFConfig(**checkpoint['model_args'])
        self.model = DelphiSMURF(self.config)

        state_dict = checkpoint['model']
        self.model.load_state_dict(state_dict)

        self.model.eval()
        self.model = self.model.to(device)

        if torch_dtype != torch.float32:
            self.model = self.model.to(torch_dtype)

        self.time_mean = self.config.time_mean
        print(f"DelphiSMURF model loaded! time_mean={self.time_mean}")
        print(f"Model params: {self.model.get_num_params()/1e6:.2f}M")

    @torch.no_grad()
    def forward(
        self, input_ehr: tuple[torch.Tensor, torch.Tensor],
        config: TrajectoryGenerationConfig,
    ) -> TrajectoryGenerationResult:
        """Generate futures in raw days with equation (2) search priors."""
        events, ages = (tensor.to(self.device) for tensor in input_ehr)
        if config.n < 1:
            raise ValueError("n must be positive")
        if events.shape[0] != 1:
            raise ValueError("The search caller accepts one observed history at a time")
        initial_length = events.shape[1]
        if config.n > 1:
            events = events.repeat(config.n, 1)
            ages = ages.repeat(config.n, 1)

        output = self.model.generate_mcts(
            idx=events, age=ages,
            max_new_tokens=config.max_new_tokens, max_age=config.max_age,
            termination_tokens=config.termination_tokens, no_repeat=config.no_repeat,
            temperature=config.temperature, top_k=config.top_k,
            age_is_normalized=config.age_is_normalized,
            n_samples=config.n_samples, n_langevin_steps=config.n_langevin_steps,
            time_uncertainty_beta=config.time_uncertainty_beta,
            atomic_candidate_pool_size=config.atomic_candidate_pool_size,
            time_uncertainty_calibration=get_time_uncertainty_calibration(
                config.calibration_dataset or ("mimic" if config.config_type == "Mimic" else "ukb")
            ),
        )
        events, ages = output["events"], output["ages"]
        expected_shape = (events.shape[0], max(events.shape[1] - 1, 0))
        for name in ("type_logprobs", "time_gap_logprobs", "step_logprobs", "time_uncertainties"):
            if output[name].shape != expected_shape:
                raise ValueError(f"{name} must have transition shape {expected_shape}")

        num_tokens, finish_reasons = [], []
        for sample_events, sample_ages in zip(events, ages):
            num_valid = int(((sample_events > 0) & (sample_ages > -1000)).sum().item())
            num_new = max(0, num_valid - initial_length)
            num_tokens.append(num_new)
            finish_reasons.append(
                "length" if num_valid == len(sample_events) and num_new >= config.max_new_tokens else "stop"
            )
        return TrajectoryGenerationResult(
            events=events, ages=ages, num_tokens=num_tokens, finish_reason=finish_reasons,
            step_logprob=output["step_logprobs"], type_logprob=output["type_logprobs"],
            time_gap_logprob=output["time_gap_logprobs"], time_uncertainty=output["time_uncertainties"],
        )
