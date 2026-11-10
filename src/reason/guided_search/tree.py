"""
The Node and MCTS class for AlphaZero.
"""

import copy
import json
import math

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple, Union, Callable, Type
from reason.env.base_env import EhrEnv
from reason.env.utils import normalize_action
import pdb
from tqdm import tqdm
import heapq
from loguru import logger
from utils import KERNEL_PREFIX_LAMBDA, KERNEL_INITIAL_COUNT
from reason.guided_search.action_kernel import segment_kernel

from reason.observer import MCTSObserver, NullObserver


class Node(object):
    """
    Overview:
        The node base class for tree_search.
    """

    def __init__(
        self, parent: "Node" = None, prior_p: float = 1.0, initial_value: float = 0.0
    ) -> None:
        self._parent = parent
        self._children = {}
        self._visit_count = 0
        self._value_sum = 0
        self.prior_p = prior_p
        self.prior_p_ori = prior_p

        self._initial_value = initial_value
        self._terminated = False

    def __lt__(self, other):
        return self._initial_value < other._initial_value

    @property
    def terminated(self):
        return self._terminated

    def set_as_terminate_node(self):
        self._terminated = True

    @property
    def value(self) -> float:
        """
        Overview:
            The value of the current node.
        Returns:
            - output (:obj:`Int`): Current value, used to compute ucb score.
        """
        if self._visit_count == 0:
            # if not visited, return the initial value
            return self._initial_value
        return self._value_sum / self._visit_count

    @property
    def initial_value(self) -> float:
        return self._initial_value

    def update(self, value: float) -> None:
        """
        Overview:
            Updata the current node information, such as visit_count and value_sum.
        Arguments:
            - value (:obj:`Int`): The value of the node.
        """
        self._visit_count += 1
        self._value_sum += value

    def update_recursive(self, leaf_value: float, mcts_mode: str) -> None:
        """
        Overview:
            Update node information recursively.
        Arguments:
            - leaf_value (:obj:`Int`): The value of the node.
        """
        if mcts_mode == "self_play_mode":
            self.update(leaf_value)
            if self.is_root():
                return
            self._parent.update_recursive(-leaf_value, mcts_mode)
        if mcts_mode == "play_with_bot_mode":
            self.update(leaf_value)
            if self.is_root():
                return
            self._parent.update_recursive(leaf_value, mcts_mode)

    def is_leaf(self) -> Dict:
        """
        Overview:
            Check if the current node is a leaf node or not.
        Returns:
            - output (:obj:`Dict`): Dict type children node.
        """
        return self._children == {}

    def is_root(self) -> bool:
        """
        Overview:
            Check if the current node is a root node or not.
        Returns:
            - output (:obj:`Bool`): Whether it is the parent node.
        """
        return self._parent is None

    @property
    def parent(self) -> None:
        return self._parent

    @property
    def children(self) -> None:
        return self._children

    @property
    def visit_count(self) -> None:
        return self._visit_count

    def get_info(self):
        # return [
        #     "visit_cnt: {}, value: {:.6f}, prior: {:.6f}".format(
        #         self.visit_count, self.value, self.prior_p)
        # ]
        return {
            "visit_cnt": self.visit_count,
            "value": self.value,
            "prior_p": float(self.prior_p_ori),
            "initial_value": self._initial_value,
            "terminated": self.terminated,
        }

    def clear(self):
        self._visit_count = 0
        self._value_sum = 0
        self.prior_p = self.prior_p_ori

    def to_json(self):
        childrens = {}
        for name, child_node in self.children.items():
            childrens[name] = child_node.to_json()

        rets = {"children": childrens, "info": self.get_info()}
        return rets
    
    def __str__(self) -> str:
        if self.is_root():
            return "root"
        else:
            return "child: value: {:.3f}, prior: {:.3f}".format(
                self.last_action, self.value, self.prior_p
            )


class LanguageNode(Node):
    text_state: Optional[str] = None
    last_action: Optional[str] = None
    num_generated_token: Optional[int] = None
    macro_steps: Optional[List[Dict[str, Any]]] = None

    def __init__(
        self,
        parent: Node = None,
        prior_p: float = 1.0,
        prm_value: Optional[float] = None,
        text_state: Optional[str] = None,
        last_action: Optional[str] = None,
        initial_value: float = 0.0,
        num_generated_token: Optional[int] = None,
        macro_steps: Optional[List[Dict[str, Any]]] = None,
        step_rewards: Optional[List[float]] = None,
        reward_sum: float = 0.0,
        reward_count: int = 0,
        step_logprobs: Optional[List[float]] = None,
        type_logprobs: Optional[List[float]] = None,
        time_gap_logprobs: Optional[List[float]] = None,
        action_score: Optional[float] = None,
        action_logprob: Optional[float] = None,
    ) -> None:
        super().__init__(parent, prior_p, initial_value)
        self.text_state = text_state
        self.last_action = last_action
        self.prm_value = prm_value

        self.num_generated_token = num_generated_token
        self.has_collected_token_num = False

        self.macro_steps = macro_steps or []
        self.step_rewards = step_rewards or []
        self.reward_sum = float(reward_sum)
        self.reward_count = int(reward_count)
        self.step_logprobs = step_logprobs or []
        self.type_logprobs = type_logprobs or []
        self.time_gap_logprobs = time_gap_logprobs or []
        self.action_score = action_score
        self.action_logprob = action_logprob

    def get_path(self):
        ans = []
        node = self
        while not node.is_root():
            ans.append(node.last_action)
            node = node.parent
        return "\n".join(reversed(ans))

    def get_info(self):
        info_dict = super().get_info()
        if not self.is_root():
            info_dict["last_action"] = self.last_action
            info_dict["prm_value"] = self.prm_value
            info_dict["macro_steps"] = self.macro_steps
            info_dict["step_rewards"] = self.step_rewards
            info_dict["reward_sum"] = self.reward_sum
            info_dict["reward_count"] = self.reward_count
            info_dict["step_logprobs"] = self.step_logprobs
            info_dict["type_logprobs"] = self.type_logprobs
            info_dict["time_gap_logprobs"] = self.time_gap_logprobs
            info_dict["action_score"] = self.action_score
            info_dict["action_logprob"] = self.action_logprob
            # info_dict["acc_path_score"] = self.acc_path_score
        else:
            info_dict["text_state"] = self.text_state
        return info_dict
    
    def __str__(self):
        if self.is_root():
            return "root: {}".format(self.text_state)
        else: 
            return "action: {}, value: {:.3f}, prior: {:.3f}".format(
                self.last_action, self.value, self.prior_p
            )


def get_root(node: Node):
    while not node.is_root():
        node = node.parent
    return node


def _build_macro_steps(action, prev_age: float) -> List[Dict[str, Any]]:
    steps = []
    cur_age = prev_age
    items = action if isinstance(action, (list, tuple)) else [action]
    for item in items:
        event_type = int(item[0])
        age = float(item[1])
        # gap = age - cur_age
        gap = max(age - cur_age, 0)
        steps.append({"type": event_type, "gap": gap, "age": age})
        cur_age = age
    return steps


class SearchTree:
    def __init__(self, cfg, observer: Optional["MCTSObserver"] = None) -> None:
        self._cfg = cfg

        self._num_simulations = self._cfg.get("num_simulations", 20)

        # UCB formula
        self._pb_c_base = self._cfg.get("pb_c_base", 19652)  # 19652
        self._pb_c_init = self._cfg.get("pb_c_init", 1.25)  # 1.25

        # Root prior exploration noise.
        self._root_dirichlet_alpha = self._cfg.get(
            "root_dirichlet_alpha", 0.3
        )  # 0.3  # for chess, 0.03 for Go and 0.15 for shogi.
        self._root_noise_weight = self._cfg.get("root_noise_weight", 0.25)  # 0.25

        self.root = None

        self.visited_paths = None
        self.mask_non_terminal_node_value = self._cfg.get(
            "mask_non_terminal_node_value", False
        )

        self._init_critic_value = self._cfg.get("init_critic_value", True)

        self._completion_tokens = 0

        self.observer = observer if observer is not None else NullObserver()

    @property
    def num_generated_token(self):
        return self._completion_tokens

    def clear_node(self, node):
        assert node is not None
        node.clear()
        for child in node.children.values():
            self.clear_node(child)

    def get_next_action(
        self,
        simulate_env: Type[EhrEnv],
        reward_fn: Optional[Callable] = None,
        temperature: int = 1.0,
        sample: bool = True,
        return_tree=False,
    ) -> Tuple[int, List[float]]:
        if self.root is None:
            root = LanguageNode(text_state=simulate_env.get_state())
            self._expand_leaf_node(root, simulate_env, reward_fn)
            self.root = root
        else:
            root = self.root

        if root.is_leaf():
            # if root is leaf node, expand it
            # We have updated the environment legal action when we test the node is leaf node
            # So the expansion won't have bugs
            self._expand_leaf_node(root, simulate_env, reward_fn)

        if sample:
            self._add_exploration_noise(root)

        for n in range(self._num_simulations):
            simulate_env_copy = simulate_env.copy()
            simulate_env_copy.battle_mode = simulate_env_copy.mcts_mode
            self._simulate(root, simulate_env_copy, reward_fn)

        # for debugging
        # print('after simulation')
        # print('value= {}'.format([(k, v.value) for k,v in root.children.items()]))
        # print('visit_count= {}'.format([(k, v.visit_count) for k,v in root.children.items()]))

        action_visits = []
        for action_dict in simulate_env.legal_actions:
            action = action_dict["action"]
            if action in root.children:
                action_visits.append((action, root.children[action].visit_count))
            else:
                action_visits.append((action, 0))

        actions, visits = zip(*action_visits)
        action_probs = nn.functional.softmax(
            1.0
            / temperature
            * np.log(torch.as_tensor(visits, dtype=torch.float32) + 1e-10),
            dim=0,
        ).numpy()
        if sample:
            action = np.random.choice(actions, p=action_probs)
            self.reset_prior(root)
        else:
            action = actions[np.argmax(action_probs)]

        self.root = root
        if return_tree:
            return action, action_probs, root
        return action, action_probs

    def vanila_mcts(
        self,
        simulate_env: Type[EhrEnv],
        num_path: int,
        reward_model_fn: Optional[Callable] = None,
        select_by_prior: bool = False,
    ) -> List[Dict]:
        api_call_completion_tokens = 0
        _, info = simulate_env.reset(update_legal_action=True)
        api_call_completion_tokens += info["api_completion_token"]
        if self.root is None:
            root = LanguageNode(text_state=simulate_env.get_state())
            self._expand_leaf_node(root, simulate_env, reward_model_fn)
            self.root = root
            # Observer hook: root initialized
            self.observer.on_expand({"phase": "root_init", "node_id": "root", "runtime_node_id": id(root), "depth": 0})

        traj_list = []

        # TODO(ziyu): split with 1. select 2. expand 3. rollout 4. backprop
        #  for here is split the for loop with select and rollout
        #  so that arbitrary rollout function can be used here.
        
        for i_path in range(num_path):
            
            node = self.root
            env_copy = simulate_env.copy()
            done = False
            while not done:
                if node.visit_count > 0:
                    action, node = self._select_child(node, env_copy)
                else:
                    if select_by_prior:
                        print(f"[DEBUG] select_by_prior activated!")
                        action, node = self._select_by_prior(node, env_copy)
                    else:
                        action, node = self._select_child(node, env_copy)
                
                # sync terminated flag here
                # XXX(ziyu): find a more clean way
                env_copy._next_state_terminated = {}
                assert node.last_action == action
                env_copy._next_state_terminated[action] = node.terminated

                _, _, terminated, truncated, info = env_copy.step(
                    action, update_legal_action=node.is_leaf()
                )

                done = terminated or truncated

                if not done and node.is_leaf():
                    self._expand_leaf_node(node, env_copy, reward_model_fn)

                # record api_tokens, if not expand, info["api_completion_token"] is 0
                api_call_completion_tokens += info["api_completion_token"]
            else:
                if self._init_critic_value:
                    leaf_value = node.initial_value
                else:
                    leaf_value = reward_model_fn(env_copy.get_state()).item()
            # Observer hook: before backup
            self.observer.on_backup({"leaf_value": leaf_value, "path_idx": i_path})

            node.update_recursive(leaf_value, env_copy.mcts_mode)

            traj_data = {
                "path_idx": i_path,
                # "text": env_copy.answer,
                "text": env_copy.flatten_answer,
                "value": leaf_value,
                "api_completion_tokens": api_call_completion_tokens,
                "tree_completion_tokens": self._completion_tokens,
                "path_score": node.initial_value if self._init_critic_value else leaf_value,
            }

            traj_list.append(traj_data)

            # Observer hook: path finished
            self.observer.on_path_finish({
                "path_idx": i_path,
                "leaf_value": leaf_value,
                "flatten_answer": env_copy.flatten_answer,
                "action_groups": env_copy.answer,
                "api_completion_tokens": api_call_completion_tokens,
                "tree_completion_tokens": self._completion_tokens,
            })

            # reset api_call_completion_tokens
            api_call_completion_tokens = 0

        return traj_list

    def beam_search(
        self,
        simulate_env: EhrEnv,
        beam_size: int,
        max_step: int,
        reward_model_fn: Optional[Callable] = None,
    ) -> List[Dict]:
        """Beam Search implementation
        Args:
            simulate_env: The environment to simulate the search.
            beam_size: beam_size
            max_step: The maximum number of steps to search.
            reward_model_fn: The reward model function to evaluate the state.
        """
        api_call_completion_tokens = 0
        _, info = simulate_env.reset(update_legal_action=True)
        api_call_completion_tokens += info["api_completion_token"]
        if self.root is None:
            root = LanguageNode(text_state=simulate_env.get_state())
            self._expand_leaf_node(root, simulate_env, reward_model_fn)
            self.root = root

        end_nodes, top_k_nodes = [], [(-root._initial_value, root, simulate_env.copy())]
        k = beam_size

        for _ in range(max_step + 1):
            cur_nodes_to_search = top_k_nodes
            top_k_nodes = []
            for cur_neg_v, cur_node, cur_env in cur_nodes_to_search:
                if cur_node.terminated:
                    end_nodes.append((cur_neg_v, cur_node, cur_env))
                    k -= 1
                elif k > 0:
                    # select at most topk children add push to heap
                    assert (
                        len(cur_node.children) > 0
                    ), "in beam search you should expand this non-terminal node at first."

                    top_k_children = sorted(
                        [
                            (action, child, child._initial_value)
                            for action, child in cur_node.children.items()
                        ],
                        key=lambda x: x[2],
                        reverse=True,
                    )[:k]
                    for c_act, c_node, c_value in top_k_children:
                        new_env = cur_env.copy()
                        heapq.heappush(top_k_nodes, (-c_value, c_node, new_env))
            # nsmallest since we negate the value
            top_k_nodes = heapq.nsmallest(k, top_k_nodes)

            # expand selected nodes
            # XXX(ziyu): this could be optimized by batch expand
            for value, node, new_env in top_k_nodes:
                _, _, terminated, truncated, info = new_env.step(
                    node.last_action, update_legal_action=True
                )
                api_call_completion_tokens += info["api_completion_token"]
                if terminated or truncated:
                    node.set_as_terminate_node()
                else:
                    self._expand_leaf_node(node, new_env, reward_model_fn)

            if len(end_nodes) == beam_size:
                assert k == 0
                break

        traj_list = []
        for i, (neg_e_v, e_node, e_env) in enumerate(end_nodes):
            traj_list.append(
                {
                    "path_idx": i,
                    "text": e_env.flatten_answer,
                    "value": -neg_e_v,
                    "api_completion_tokens": 0,
                    "tree_completion_tokens": 0,
                    # num_generated_token is hard to compute for each single answer
                }
            )
        traj_list[-1]["tree_completion_tokens"] = self._completion_tokens
        traj_list[-1]["api_completion_tokens"] = api_call_completion_tokens
        return traj_list

    def _simulate(
        self,
        node: Node,
        simulate_env: Type[EhrEnv],
        reward_fn: Optional[Callable] = None,
    ) -> None:
        """
        Overview:
            Run a single playout from the root to the leaf, getting a value at the leaf and propagating it back through its parents.
            State is modified in-place, so a deepcopy must be provided.
        Arguments:
            - node (:obj:`Class Node`): Current node when performing mcts search.
            - simulate_env (:obj:`Class BaseGameEnv`): The class of simulate env.
            - reward_fn (:obj:`Function`): The Callable to compute the action probs and state value.
        """
        # XXX: fix the bug temporally, better implementation is required.
        done = False
        while not node.is_leaf():
            action, node = self._select_child(node, simulate_env)
            _, _, terminated, truncated, info = simulate_env.step(
                action, update_legal_action=(node.is_leaf() and node.visit_count == 1)
            )
            done = terminated or truncated

            # In original AlphaZero, the leaf node will be expanded once it is reached
            # In our setting, computing legal action is computational inefficient
            # Thus when we reach a leaf node, we will not directly expand it
            # Until the next time, when this node's children are required to be selected
            # In this case, node is leaf node and the visit count number of node is 1
            # Then we expand it

            if not done and node.is_leaf() and node.visit_count == 1:
                # Once we expand the node, the node will not be leaf node any more
                # And the while won't break
                self._expand_leaf_node(node, simulate_env, reward_fn)

        """
        in ``self_play_mode``, the leaf_value is calculated from the perspective of player ``simulate_env.current_player``.
        in ``play_with_bot_mode``, the leaf_value is calculated from the perspective of player 1.
        """
        if not done and self.mask_non_terminal_node_value:
            leaf_value = 0.0
        elif self._init_critic_value:
            leaf_value = node.initial_value
        else:
            leaf_value = reward_fn(simulate_env.get_state()).item()

        if done:
            node.set_as_terminate_node()
            if self.visited_paths is not None:
                self.visited_paths.append(
                    {
                        "text": simulate_env.flatten_answer,
                        "value": leaf_value,
                    }
                )

        # Update value and visit count of nodes in this traversal.
        if simulate_env.mcts_mode == "play_with_bot_mode":
            node.update_recursive(leaf_value, simulate_env.mcts_mode)

        elif simulate_env.mcts_mode == "self_play_mode":
            # NOTE: e.g.
            #       to_play: 1  ---------->  2  ---------->  1  ----------> 2
            #         state: s1 ---------->  s2 ---------->  s3 ----------> s4
            #                                     action    node
            #                                            leaf_value
            # leaf_value is calculated from the perspective of player 1, leaf_value = value_func(s3),
            # but node.value should be the value of E[q(s2, action)], i.e. calculated from the perspective of player 2.
            # thus we add the negative when call update_recursive().
            node.update_recursive(-leaf_value, simulate_env.mcts_mode)

    def _select_child(
        self, node: LanguageNode, simulate_env: Type[EhrEnv]
    ) -> Tuple[Union[int, float], Node]:
        """
        Overview:
            Select the child with the highest UCB score.
        Arguments:
            - node (:obj:`Class Node`): Current node.
        Returns:
            - action (:obj:`Int`): choose the action with the highest ucb score.
            - child (:obj:`Node`): the child node reached by executing the action with the highest ucb score.
        """

        action = None
        child = None
        best_score = -9999999
        candidate_scores = []

        for action_tmp, child_tmp in node.children.items():
            ucb_score = self._ucb_score(node, child_tmp)
            score = ucb_score
            candidate_scores.append({"action": action_tmp, "ucb": float(ucb_score)})
            if score > best_score:
                best_score = score
                action = action_tmp
                child = child_tmp

        if child is None:
            child = node  # child==None, node is leaf node in play_with_bot_mode.

        # Observer hook: selection made
        self.observer.on_select({
            "candidates": candidate_scores,
            "history": simulate_env.question + simulate_env.flatten_answer,
        })
        # self.observer.on_select({
        #     "candidates": candidate_scores,
        #     "chosen_action": str(action),
        #     "best_score": float(best_score),
        # })

        return action, child

    def _select_by_prior(self, node: Node, simulate_env):
        data_tmp = [
            (x_action, x_node.prior_p) for x_action, x_node in node.children.items()
        ]
        action_list, prior_list = list(zip(*data_tmp))
        chosen_action = np.random.choice(action_list, p=np.array(prior_list))
        chosen_node = node.children[chosen_action]

        return chosen_action, chosen_node

    def _expand_leaf_node(
        self,
        node: Node,
        simulate_env: Type[EhrEnv],
        reward_fn: Optional[Callable] = None,
    ) -> float:
        text_state = simulate_env.get_state()
        if not self._init_critic_value:
            leaf_value = reward_fn(text_state)
            child_step_rewards = [[] for _ in simulate_env.legal_actions]
            child_values = [0.0 for _ in simulate_env.legal_actions]

        else:
            leaf_value = node._initial_value
            assert len(simulate_env.legal_actions) > 0
            prefix = list(simulate_env.flatten_answer)
            sim_trajs = [
                (
                    simulate_env.question,
                    prefix + list(x["action"]),
                )
                for x in simulate_env.legal_actions
            ]
            prms = reward_fn(
                sim_trajs
            )
            child_step_rewards = []
            child_values = []
            
            if len(prms) != len(simulate_env.legal_actions):
                raise ValueError("PRM candidate count does not match legal actions")
            for action_dict, rs in zip(simulate_env.legal_actions, prms):
                act = action_dict.get('action')
                if not act or len(rs) != len(prefix) + len(act):
                    raise ValueError("PRM reward count must equal full generated prefix length")
                rs = [float(x) for x in rs[len(prefix):]]
                if not all(math.isfinite(x) for x in rs):
                    raise ValueError("PRM rewards must be finite")
                if act[-1][0] in getattr(simulate_env, "mcts_termination_tokens", []):
                    rs = rs[:-1]
                child_step_rewards.append(rs)
                parent_sum = getattr(node, "reward_sum", 0.0)
                parent_count = getattr(node, "reward_count", 0)
                count = parent_count + len(rs)
                child_values.append((parent_sum + sum(rs)) / count if count else 0.0)

        assert len(node.children) == 0
        children_info = []

        # Determine the last age in the current state for time_gap computation.
        # The state is (events_tensor, ages_tensor); take the last valid age.
        state_for_age = simulate_env.question + simulate_env.flatten_answer
        if len(state_for_age) > 0 and isinstance(state_for_age[-1], (list, tuple)):
            prev_age = float(state_for_age[-1][1])
        else:
            prev_age = 0.0

        for i, action_dict in enumerate(simulate_env.legal_actions):
            action, prob = action_dict["action"], action_dict["prob"]
            # Normalize action to make it hashable (convert list to tuple)
            action = normalize_action(action)

            if self._init_critic_value:
                child_value = child_values[i]
            else:
                # XXX(ziyu): consider turn off this branch, i.e. always assume
                #  `self._init_critic=True`, since with LLM
                child_value = 0.0

            node.children[action] = LanguageNode(
                parent=node,
                prior_p=prob,
                #  prm_value=prm_value,
                text_state=text_state,
                last_action=action,
                initial_value=child_value,
                num_generated_token=action_dict["num_token"],

                # new
                macro_steps=_build_macro_steps(action, prev_age),
                step_rewards=child_step_rewards[i],
                reward_sum=getattr(node, "reward_sum", 0.0) + sum(child_step_rewards[i]) if self._init_critic_value else 0.0,
                reward_count=getattr(node, "reward_count", 0) + len(child_step_rewards[i]) if self._init_critic_value else 0,
                step_logprobs=action_dict.get("step_logprobs", []),
                action_score=action_dict.get("action_score"),
                action_logprob=action_dict.get("action_logprob"),
            )
            if simulate_env._next_state_terminated[action]:
                node.children[action].set_as_terminate_node()

            # Collect child info for observer
            children_info.append({
                "runtime_node_id": id(node.children[action]),
                "action_key": action,
                "action_fragment_raw": action_dict["action"],
                "action_fragment_norm": action,
                "prior": float(prob),
                "initial_value": float(child_value),
                "terminated": simulate_env._next_state_terminated[action],
                "num_generated_token": action_dict["num_token"],
                "macro_steps": node.children[action].macro_steps,
            })

        if len(node.children) == 0:
            # print_rank_0(
            print("Prune all current children at node {}".format(node.last_action))

        # Observer hook: expansion complete
        self.observer.on_expand({
            "parent_runtime_id": id(node),
            "depth": self._node_depth(node),
            "children": children_info,
        })
        
        # collect num tokens
        if not node.has_collected_token_num:
            self._completion_tokens += sum(
                c.num_generated_token for c in node.children.values()
            )
            node.has_collected_token_num = True
        else:
            raise RuntimeError("Token number has been collected again.")

        return leaf_value

    @staticmethod
    def _node_depth(node: Node) -> int:
        depth = 0
        n = node
        while not n.is_root():
            n = n.parent
            depth += 1
        return depth

    def _ucb_score_old(self, parent: Node, child: Node) -> float:
        """
        Overview:
            Compute UCB score. The score for a node is based on its value, plus an exploration bonus based on the prior.
        Arguments:
            - parent (:obj:`Class Node`): Current node.
            - child (:obj:`Class Node`): Current node's child.
        Returns:
            - score (:obj:`Bool`): The UCB score.
        """
        pb_c = (
            math.log((parent.visit_count + self._pb_c_base + 1) / self._pb_c_base)
            + self._pb_c_init
        )
        pb_c *= math.sqrt(parent.visit_count + 1) / (child.visit_count + 1)

        prior_score = pb_c * child.prior_p
        value_score = child.value

        return prior_score + value_score
        # return value_score
    
    # ==== UCB score with kernel ====
    def _kernel_density(self, parent, target_child):
        count = float(self._cfg.get("kernel_init_count", KERNEL_INITIAL_COUNT))
        if not math.isfinite(count) or count < 0:
            raise ValueError("kernel_init_count must be finite and nonnegative")
        for other in parent.children.values():
            k = segment_kernel(
                target_child.macro_steps,
                other.macro_steps,
                prefix_lambda=self._cfg.get("kernel_prefix_lambda", KERNEL_PREFIX_LAMBDA),
            )
            count += k * other.visit_count
        return count

    def _kernel_value(self, parent, target_child):
        n0 = float(self._cfg.get("kernel_init_count", KERNEL_INITIAL_COUNT))
        value_sum = n0 * target_child.initial_value
        for other in parent.children.values():
            k = segment_kernel(
                target_child.macro_steps,
                other.macro_steps,
                prefix_lambda=self._cfg.get("kernel_prefix_lambda", KERNEL_PREFIX_LAMBDA),
            )
            value_sum += k * other._value_sum
        count = self._kernel_density(parent, target_child)
        return value_sum / count if count > 0 else target_child.initial_value

    def _ucb_score(self, parent: Node, child: Node) -> float:
        use_kernel = self._cfg.get("use_kernel_visit", True)

        if not use_kernel:
            pb_c = (
                math.log((parent.visit_count + self._pb_c_base + 1) / self._pb_c_base)
                + self._pb_c_init
            )
            pb_c *= math.sqrt(parent.visit_count + 1) / (child.visit_count + 1)
            return child.value + pb_c * child.prior_p

        n_child = self._kernel_density(parent, child)
        n_parent = sum(other.visit_count for other in parent.children.values())
        q_hat = self._kernel_value(parent, child)

        pb_c = (
            math.log((n_parent + self._pb_c_base + 1) / self._pb_c_base)
            + self._pb_c_init
        )
        pb_c *= math.sqrt(n_parent + 1.0) / (n_child + 1.0)

        return q_hat + pb_c * child.prior_p
    # ==== UCB score with kernel ====


    def reset_prior(self, node: Node) -> None:
        """
        Overview:
            Reset prior probability
        Arguments:
            - node (:obj:`Class Node`): Current node.
        """
        for a in node.children.keys():
            node.children[a].prior_p = node.children[a].prior_p_ori

    def _add_exploration_noise(self, node: Node) -> None:
        """
        Overview:
            Add exploration noise.
        Arguments:
            - node (:obj:`Class Node`): Current node.
        """
        # Get a list of actions corresponding to the child nodes.
        actions = list(node.children.keys())
        # Create a list of alpha values for Dirichlet noise.
        alpha = [self._root_dirichlet_alpha] * len(actions)
        # Generate Dirichlet noise using the alpha values.
        noise = np.random.dirichlet(alpha)
        # Compute the weight of the exploration noise.
        frac = self._root_noise_weight
        # Update the prior probability of each child node with the exploration noise.
        for a, n in zip(actions, noise):
            node.children[a].prior_p = node.children[a].prior_p * (1 - frac) + n * frac

    @classmethod
    def from_json(cls, cfg: dict, json_path: str, reset_visit_info: bool):
        tree_json = json.load(open(json_path, "r"))

        def build_tree(tree_dict: dict) -> Node:
            node_info = tree_dict["info"]
            current_node = LanguageNode(
                text_state=node_info.get("text_state", None),
                last_action=node_info.get("last_action", None),
                prior_p=node_info["prior_p"],
                prm_value=node_info.get("prm_value", None),
                initial_value=node_info.get("initial_value", 0.0),
                macro_steps=node_info.get("macro_steps", None),
                reward_sum=node_info.get("reward_sum", 0.0),
                reward_count=node_info.get("reward_count", 0),
            )

            if not reset_visit_info:
                current_node._visit_count = node_info["visit_cnt"]
                current_node._value_sum = node_info["value"] * current_node.visit_count
            if node_info.get("terminated", False):
                current_node.set_as_terminate_node()

            for name, child_dict in tree_dict["children"].items():
                child_node = build_tree(child_dict)
                current_node._children[name] = child_node
                child_node._parent = current_node

            return current_node

        root_node = build_tree(tree_dict=tree_json)

        obj = cls(cfg)
        obj.root = root_node
        return obj
    
    def draw_tree(self):
        # Not tested yet
        root = self.root
        assert root, 'Root node is None'
        def draw_node(node, depth):
            print('|' + '-' * depth + str(node))
            for child in node.children.values():
                draw_node(child, depth + 1)
        
        print(f"\n---------Expanded Tree---------")
        draw_node(self.root)
