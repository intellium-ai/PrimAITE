import json
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import Any, Literal, Optional, Type, TypeVar, Dict, List
from termcolor import colored
import numpy as np
from pydantic import BaseModel
from text_generation import Client
from text_generation.types import Grammar, GrammarType
from primaite import getLogger
from primaite.action import NodeAction
from primaite.agents.agent_abc import AgentSessionABC
from primaite.agents.llm.utils import network_connectivity_desc, obs_diff, obs_view_full
from primaite.common.enums import AgentFramework, AgentIdentifier
from primaite.environment.env_state import EnvironmentState
from primaite.environment.primaite_env import Primaite
from primaite.exceptions import LLMGrammarError
from outlines.fsm.json_schema import build_regex_from_schema
from primaite.agents.llm.prompting import (
    SYSTEM_MSG,
    NODE_ACTION_SPACE_DESCRIPTION,
    HISTORY_PREFIX,
    CURRENT_OBS,
    ACTION_PROMPT,
    AgentNodeAction,
)

_LOGGER: Logger = getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


def format_llama_prompt(system: str, messages: list[tuple[Literal["user", "assistant"], str]]) -> str:
    prompt = "<|begin_of_text|>"

    # system prompt
    prompt += "<|start_header_id|>system<|end_header_id|>\n\n"
    prompt += system
    prompt += "<|eot_id|>"

    # prev messages
    for role, msg in messages:
        prompt += f"<|start_header_id|>{role}<|end_header_id|>\n\n"
        prompt += msg
        prompt += "<|eot_id|>"

    # prompt llm to answer
    prompt += "<|start_header_id|>assistant<|end_header_id|>"

    return prompt


class LLM:
    def __init__(self, base_url: str, timeout: int = 60) -> None:
        self.client = Client(base_url=base_url, timeout=timeout)

    def generate(
        self, prompt: str, repetition_penalty: Optional[float] = None, max_new_tokens: Optional[int] = 1024
    ) -> str:
        response = self.client.generate(
            prompt=prompt, max_new_tokens=max_new_tokens, repetition_penalty=repetition_penalty
        )
        return response.generated_text

    def generate_model(
        self,
        prompt: str,
        model: Type[T],
        repetition_penalty: Optional[float] = None,
        max_new_tokens: Optional[int] = 1024,
        new_literals: Optional[Dict[str, List]] = None,
    ) -> T:

        model_schema = model.model_json_schema()
        if new_literals:
            # For each set of literals (referred to as enums) passed, add to the schema for the respective property.
            for prop, enum in new_literals.items():
                model_schema["properties"][prop]["enum"] = enum

        # Convert json schema to regex to preserve property order during generation
        model_json = json.dumps(model_schema, indent=2)
        model_regex = build_regex_from_schema(model_json)

        # Generate response for regex
        response = self.client.generate(
            prompt=prompt,
            grammar=Grammar(type=GrammarType.Regex, value=model_regex),
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        ).generated_text

        try:
            # Parse response
            response_model = model(**json.loads(response))
            _LOGGER.info(f"{colored('Action', 'green')}: {response_model}")
        except json.JSONDecodeError as e:
            # Grammar didn't work
            raise LLMGrammarError(f"LLM did not produce valid grammar JSON schema: {e}. Generated: {response}")
        except Exception as e:
            raise Exception(e)

        return response_model

    def _get_action_space_description(self, env_state: EnvironmentState) -> str:
        env = env_state.env
        node_names = "'" + ", ".join([n.name for n in env.active_nodes]) + "'"  # Nice comma separated list
        service_names = "'" + ", ".join(env.services_list) + "'"

        action_space_description = NODE_ACTION_SPACE_DESCRIPTION.format(
            node_names=node_names, service_names=service_names
        )
        return action_space_description

    def _build_prompt(self, env_state: EnvironmentState, env_history: list[EnvironmentState]) -> str:
        prompt = ""

        # Get the action space description
        env = env_state.env
        initial_state = env_history[0]
        # Initial environment
        initial_env_desc = f"This is the initial configuration of the network:\n{network_connectivity_desc(initial_state)}\n\nInitial {obs_view_full(initial_state)}"

        node_names = "'" + ", ".join([n.name for n in env.active_nodes]) + "'"  # Nice comma separated list
        service_names = "'" + ", ".join(env.services_list) + "'"
        action_space = NODE_ACTION_SPACE_DESCRIPTION.format(node_names=node_names, service_names=service_names)

        # messages = [("user", prompt), ("assistant", "Acknowledged.")]???

        # Build the history of actions
        obs_act_history = HISTORY_PREFIX if env_history else ""
        history_list = []
        for i, state in enumerate(env_history[1:]):
            observed_changes = obs_diff(state)
            action_id = state.action_id

            if observed_changes != "" or action_id:
                history_list.append(f"\nStep {i}:")
                # obs_act_history += f"\nStep {i}:"

                if observed_changes != "":
                    history_list[-1] += f"\n{observed_changes}"
                    # obs_act_history += f"\n{observed_changes}"
                if action_id is not None:
                    action = NodeAction.from_id(env=env, action_id=action_id)
                    action_verbose = action.verbose(colored=False)
                    history_list[-1] += f"\nAction: {action_verbose}\n"
                    # obs_act_history += f"\nAction: {action_verbose}\n"

        obs_act_history += "".join(history_list[-20:])  # Only show the last 20 obs act events
        print("OBS ACT HISTORY:\n", obs_act_history)

        # Current observation and action prompt
        current_obs = CURRENT_OBS.format(obs_view_full=obs_view_full(env_state), obs_diff=obs_diff(env_state))
        _LOGGER.info("\n\n")
        _LOGGER.info(f"{colored('Observed changes', 'yellow')}: {obs_diff(env_state)}\n")

        prompt += initial_env_desc + obs_act_history + action_space + current_obs + ACTION_PROMPT
        messages = [("user", prompt)]
        prompt = format_llama_prompt(SYSTEM_MSG, messages)

        return prompt

    def predict(self, env_state: EnvironmentState, env_history: list[EnvironmentState]):
        env = env_state.env
        # BUILD PROMPT HERE
        prompt = self._build_prompt(env_state=env_state, env_history=env_history)
        print("NEW PROMPT::\n", prompt)

        agent_action = self.generate_model(
            prompt=prompt,
            model=AgentNodeAction,
            repetition_penalty=1.1,
            new_literals={"node_name": [n.name for n in env.active_nodes] + ["NONE"]},
        )

        try:
            action = agent_action.to_node_action(env=env)
        except BaseException:
            _LOGGER.info(f"Invalid LLM action: {agent_action}")
            action = NodeAction(env=env)
        action_id = action.action_id
        prompt
        return action_id, prompt


class LLMAgent(AgentSessionABC):
    def __init__(self, training_config_path, lay_down_config_path):
        super().__init__(training_config_path, lay_down_config_path)
        assert self._training_config.agent_framework == AgentFramework.CUSTOM
        assert self._training_config.agent_identifier == AgentIdentifier.LLM
        self._setup()

    def _setup(self):
        super()._setup()

        if not isinstance(self.session_path, Path):
            self.session_path = Path(self.session_path)

        self._env = Primaite(
            training_config_path=self._training_config_path,
            lay_down_config_path=self._lay_down_config_path,
            session_path=self.session_path,
            timestamp_str=self.timestamp_str,
        )
        self._agent = LLM(base_url="http://192.168.0.8:58084", timeout=120)

        # Keep track of env history
        self.env_history = [EnvironmentState(self._env)]

    def _save_checkpoint(self) -> None:
        _LOGGER.warning("Deterministic agents cannot learn")

    def learn(self):
        _LOGGER.warning("Deterministic agents cannot learn")

    def _calculate_action(self, obs: np.ndarray):
        action, info = self._calculate_action_info(obs)

        return action

    def _calculate_action_info(self, obs: np.ndarray) -> tuple[int, str | None]:
        prev_env_state = self.env_history[-1]
        env_state = EnvironmentState(self._env, prev_env_state=prev_env_state)

        action, info = self._agent.predict(env_state, self.env_history)
        env_state.action_id = action
        self.env_history.append(env_state)

        return action, info

    def evaluate(
        self,
        **kwargs: Any,
    ) -> None:
        """
        Evaluate the agent.

        :param kwargs: Any agent-specific key-word args to be passed.
        """
        time_steps = self._training_config.num_eval_steps
        episodes = self._training_config.num_eval_episodes
        self._env.set_as_eval()
        self.is_eval = True

        _LOGGER.info(f"Num services: {self._env.num_services}")

        for _ in range(episodes):
            obs = self._env.reset()
            done, steps, rew = False, 0, 0
            while steps < time_steps and not done:

                action = self._calculate_action(obs)

                obs, rewards, done, info = self._env.step(action=action)
                steps += 1
                # rew += rewards

        self._env._write_av_reward_per_episode()  # noqa
        self._env.close()
        super().evaluate()

    def _get_latest_checkpoint(self):
        pass

    @classmethod
    def load(cls, path):
        pass

    def save(self):
        return None

    def export(self) -> None:
        return None
