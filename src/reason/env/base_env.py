import abc
from typing import Optional
import numpy as np
import copy
import torch
from reason.inference.types import TrajectoryGenerationConfig, TrajectoryGenerationResult
from reason.env.prior import macro_action_log_score
from utils import MACRO_PRIOR_ETA, MACRO_PRIOR_EPS, MACRO_PRIOR_TEMPERATURE
from reason.env.utils import normalize_action, flatten_action, build_action_groups
from reason.observer import MCTSObserver, NullObserver

INVALID_ANS = "[invalid]"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEFAULT_TERMINATION_TOKENS = [1145]  # Default death token for UKB

class NoLegalActionException(Exception):
    pass


class ResetException(Exception):
    pass


class BaseEnv(abc.ABC):
    """Basic environment to use for MCTS"""

    @abc.abstractmethod
    def reset(self, update_legal_action: bool):
        raise NotImplementedError

    @abc.abstractmethod
    def step(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def legal_actions(self):
        raise NotImplementedError

    @abc.abstractmethod
    def copy(self):
        raise NotImplementedError

    @staticmethod
    def build_query_input(
        problem_input: list,
    ):
        return problem_input


class EhrEnv(BaseEnv):
    """Search state and macro actions for observed EHR histories."""

    def get_reward(self):
        """To implement based on learned reward model"""
        return 0
    
    def build_query_input(self, ehr_record):
        events = [event[0] for event in ehr_record]
        ages = [event[1] for event in ehr_record]
        
        events = torch.tensor(events, device=DEVICE).unsqueeze(0)
        ages = torch.tensor(ages, device=DEVICE).unsqueeze(0)
        return events, ages

    def __init__(
        self,
        config,
        ehr_records,
        llm_gen_fn,
        reset=True,
        observer: Optional["MCTSObserver"] = None,
    ):
        self.config = config
        self.mcts_mode = "play_with_bot_mode"
        self.ehr_records = ehr_records
        self.llm_gen_fn = llm_gen_fn
        self.action_history = None
        self.ehr_record = None
        self._legal_actions = None

        # Termination configuration
        self.use_termination_tokens = config.get("use_termination_tokens", True)
        self.termination_tokens = config.get("termination_tokens", DEFAULT_TERMINATION_TOKENS)
        if not self.use_termination_tokens:
            self.termination_tokens = []  # Empty list = no termination tokens
        self.mcts_termination_tokens = config.get(
            "mcts_termination_tokens",
            self.termination_tokens,
        )
        self.mcts_max_age_days = config.get("mcts_max_age_days")

        self.observer = observer if observer is not None else NullObserver()

        if reset:
            self.reset(update_legal_action=True)

    def reset(self, update_legal_action=True):
        # reset environment to problem idx
        self.set_problem(idx=0)
        self.action_history = []
        self._next_state_terminated = {}
        self._legal_actions = None
        self._init_query = self.ehr_record["question"]

        # Observer hook: env reset
        self.observer.on_env_reset({
            "question_length": len(self._init_query),
            "initial_state_length": len(self._init_query),
        })

        api_completion_token = 0
        if update_legal_action:
            cnt = 0
            while cnt < 3:
                cnt += 1
                try:
                    self._legal_actions, api_completion_token = (
                        self.update_legal_actions()
                    )
                    break
                except NoLegalActionException as e:
                    if cnt == 3:
                        raise ResetException
        info = {"api_completion_token": api_completion_token}
        return self.get_state(), info

    def step(self, action, update_legal_action=True):
        # self.action_history.extend(list[Any](action))
        self.action_history.append(action)
        state = self.get_state()
        reward = self.get_reward()
        terminated, truncated, info = self.get_done_and_info()

        # Observer hook: step taken
        self.observer.on_step({
            "chosen_action": str(action)[:100],
            "terminated": terminated,
            "truncated": truncated,
            "action_history_length": len(self.action_history),
        })

        # update legal actions
        if not (terminated or truncated) and update_legal_action:
            cnt = 0
            while cnt < 3:
                cnt += 1
                try:
                    self._legal_actions, api_completion_token = self.update_legal_actions()
                    info["api_completion_token"] = api_completion_token
                    break
                except NoLegalActionException as e:
                    if cnt == 3:
                        terminated = True
                        reward = 0
                        self._legal_actions = None
                        info["api_completion_token"] = 0
                    else:
                        pass
        else:
            self._legal_actions = None
            info["api_completion_token"] = 0
        return state, reward, terminated, truncated, info
    
    @property
    def act_his(self):
        return flatten_action(self.action_history)

    def get_state(self):
        # 合并初始查询和动作历史
        # action_history中的每个action都是[(event, age), ...]格式
        ret = self._init_query.copy() if isinstance(self._init_query, list) else self._init_query
        ret.extend(flatten_action(self.action_history))
        ret = self.build_query_input(ret)
        return ret

    def post_process_act(self, action):
        # action是[(event, age), ...]格式的列表
        # 对于EHR，直接返回action
        action = normalize_action(action)
        return action

    def _mcts_termination_status(self, action):
        has_termination = any(
            event in self.mcts_termination_tokens for event, _ in action
        )
        max_age_reached = self.mcts_max_age_days is not None and any(
            age >= float(self.mcts_max_age_days) for _, age in action
        )
        return has_termination, max_age_reached

    def _truncate_action(self, action):
        kept = []
        for event, age in action:
            if self.mcts_max_age_days is not None and age > self.mcts_max_age_days:
                break
            kept.append((event, age))
            if event in self.mcts_termination_tokens:
                break
        return kept

    def update_legal_actions(self):
        # generate actions
        gen_config = TrajectoryGenerationConfig(
            n=self.config["max_actions"],
            **self.config["generation_config"],
            termination_tokens=self.termination_tokens,
        )
        cur_state = self.get_state()
        result: TrajectoryGenerationResult = self.llm_gen_fn.forward(
            input_ehr=cur_state,
            config=gen_config
        )

        events = result.events  # [n_samples, seq_len]
        ages = result.ages  # [n_samples, seq_len]
        token_len = result.num_tokens

        action_list, num_token_list = [], []
        finish_reason_list = []
        next_state_terminated = {}
        raw_candidates = []

        step_logprob_list = []
        type_logprob_list = []
        time_gap_logprob_list = []

        # 获取当前状态的长度（初始序列长度）
        current_state = self.get_state()
        initial_length = current_state[0].shape[1]  # events tensor的长度

        for i in range(len(result.finish_reason)):
            # 提取第i个样本的序列
            sample_events = events[i]
            sample_ages = ages[i]

            # 找到有效token（非padding）
            valid_mask = (sample_events > 0) & (sample_ages > -1000)
            num_valid = valid_mask.sum().item()

            # 提取新生成的部分（排除初始prompt）
            if num_valid > initial_length:
                new_events = sample_events[initial_length:num_valid]
                new_ages = sample_ages[initial_length:num_valid]

                # 将token序列转换为action格式（列表的元组）
                action = [(int(e.item()), float(a.item())) for e, a in zip(new_events, new_ages)]

                # 检查是否终止
                # MCTS-only stopping controls are independent from the tokens
                # passed to the underlying trajectory generator.
                has_termination, max_age_reached = (
                    self._mcts_termination_status(action)
                )
                terminated = (
                    has_termination
                    or max_age_reached
                )
                
                # 处理action
                processed_act = self.post_process_act(self._truncate_action(action))

                # Store raw candidate for observer
                dedup_dropped = processed_act in action_list
                raw_candidates.append({
                    "generated_fragment_raw": action,
                    "generated_fragment_norm": processed_act,
                    "finish_reason": result.finish_reason[i],
                    "num_new_tokens": len(action),
                    "has_termination": has_termination,
                    "max_age_reached": max_age_reached,
                    "dedup_dropped": dedup_dropped,
                })

                # 检查是否为有效的唯一action
                if (
                    len(processed_act) > 0
                    and not dedup_dropped
                    # only stop is valid, otherwise the output action is truncated actually
                    and result.finish_reason[i] in ["stop", "length"]
                ):
                    start = initial_length - 1
                    end = start + len(processed_act)

                    if result.step_logprob is None:
                        raise ValueError("MCTS generation must return atomic search log priors")
                    step_lps = result.step_logprob[i, start:end]

                    step_lps = [float(x) for x in step_lps.detach().cpu().tolist()]

                    if len(step_lps) != len(processed_act):
                        raise ValueError(
                            "Atomic priors are not aligned with the macro action: "
                            f"{len(step_lps)} priors for {len(processed_act)} events"
                        )

                    if result.type_logprob is not None:
                        type_lps = result.type_logprob[i, start:end]
                        type_lps = [float(x) for x in type_lps.detach().cpu().tolist()]
                    else:
                        type_lps = []

                    if result.time_gap_logprob is not None:
                        time_lps = result.time_gap_logprob[i, start:end]
                        time_lps = [float(x) for x in time_lps.detach().cpu().tolist()]
                    else:
                        time_lps = []

                    action_list.append(processed_act)
                    num_token_list.append(len(processed_act))
                    finish_reason_list.append(result.finish_reason[i])
                    next_state_terminated[processed_act] = terminated

                    step_logprob_list.append(step_lps)
                    type_logprob_list.append(type_lps)
                    time_gap_logprob_list.append(time_lps)

        if not action_list:
            print(f"No possible action have been generated error.")
            raise NoLegalActionException("No possible action have been generated.")

        score_list = []

        eta = float(self.config.get("prior_eta", MACRO_PRIOR_ETA))
        prior_eps = float(self.config.get("prior_eps", MACRO_PRIOR_EPS))
        prior_temperature = float(self.config.get("prior_temperature", MACRO_PRIOR_TEMPERATURE))
        if prior_temperature <= 0:
            raise ValueError("prior_temperature must be positive")
        for step_logprob in step_logprob_list:
            score_list.append(macro_action_log_score(step_logprob, eta, prior_eps))
        action_prob_list = np.exp(score_list)

        score_arr = np.asarray(score_list, dtype=np.float64)
        score_arr = score_arr / prior_temperature
        score_arr = score_arr - np.max(score_arr)

        prob_arr = np.exp(score_arr)
        prob_arr = prob_arr / np.sum(prob_arr)

        _legal_actions = [
            {
                "action": action,
                "prob": float(prob),
                "num_token": n_token,
                "finish_reason": finish_reason,
                "action_prob": float(action_prob),
                "action_score": float(score),
                "step_logprobs": step_logprob,
                "type_logprobs": type_logprob,
                "time_gap_logprobs": time_gap_logprob,
            }
            for action, prob, n_token, finish_reason, action_prob, score, step_logprob, type_logprob, time_gap_logprob in zip(
                action_list,
                prob_arr,
                num_token_list,
                finish_reason_list,
                action_prob_list,
                score_list,
                step_logprob_list,
                type_logprob_list,
                time_gap_logprob_list,
            )
        ]

        # Observer hook: LM expansion complete
        self.observer.on_lm_expand({
            "state_len_before_expand": initial_length,
            "n_requested": gen_config.n,
            "completion_tokens": result.completion_tokens,
            "raw_candidates": raw_candidates,
            "num_legal_actions": len(_legal_actions),
        })

        self._next_state_terminated = next_state_terminated
        return _legal_actions, result.completion_tokens

    def set_problem(self, idx):
        self.ehr_record = self.ehr_records[idx]

    @property
    def query(self):
        return self._init_query
    
    @property
    def question(self) -> list[tuple[int, float]]:
        return self.ehr_record["question"]

    # @property
    # def answer(self):
    #     return flatten_action(self.action_history)

    @property
    def answer(self):
        return build_action_groups(self.action_history)

    @property
    def flatten_answer(self):
        return flatten_action(self.action_history)

    def get_done_and_info(self):
        info = {}
        
        if len(self.action_history) == 0:
            return False, False, info
            
        last_action = self.action_history[-1]
        
        # 检查是否终止
        # action_key = str(last_action)
        action_key = last_action
        if action_key in self._next_state_terminated:
            terminated = self._next_state_terminated[action_key]
        else:
            # 如果找不到，检查action本身是否包含终止条件
            if isinstance(last_action, (list, tuple)):
                termination_status = self._mcts_termination_status(last_action)
                terminated = any(termination_status)
            else:
                terminated = False

        # 计算总的事件数量（所有actions展开后的长度）
        total_events = sum(
            len(action) if isinstance(action, (list, tuple)) else 1
            for action in self.action_history
        )
        truncated = total_events >= self.config["max_length"]
        
        return terminated, truncated, info

    def copy(self):
        env = self.__class__(
            self.config,
            self.ehr_records,
            self.llm_gen_fn,
            reset=False,
            observer=self.observer,
        )
        env.ehr_record = copy.deepcopy(self.ehr_record)
        env._legal_actions = copy.deepcopy(self._legal_actions)
        env.action_history = copy.deepcopy(self.action_history)
        env._init_query = copy.deepcopy(self._init_query)
        env._next_state_terminated = copy.deepcopy(self._next_state_terminated)
        return env

    @property
    def legal_actions(self):
        return self._legal_actions


class DelphiEnv(EhrEnv):
    """Delphi trajectory search environment."""
