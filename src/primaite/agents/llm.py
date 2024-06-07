import json
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel
from text_generation import Client
from text_generation.types import Grammar, GrammarType

from primaite import getLogger
from primaite.agents.agent_abc import AgentSessionABC
from primaite.agents.env_state import EnvironmentState
from primaite.agents.llm_utils import obs_view_full
from primaite.common.enums import AgentFramework, AgentIdentifier
from primaite.environment.primaite_env import Primaite

_LOGGER: Logger = getLogger(__name__)


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


class Action(BaseModel):
    node_id: int


class LLM:
    def __init__(self, base_url: str, timeout: int = 60) -> None:
        self.client = Client(base_url=base_url, timeout=timeout)

    def predict(self, env_state: EnvironmentState, env_history: list[EnvironmentState]) -> int:

        system_msg = "You are a defensive agent "

        user_msg = f"""
        Based purely on the description you have been given, say if any node has been compromised. If you do not see any issues, return 'ok'.

        Network:
        
        {obs_view_full(env_state)}
        
        Vulnerabilities:
        """
        prompt = format_llama_prompt(system_msg, [("user", user_msg)])
        _LOGGER.info(f"{prompt}")
        # response = self.client.generate(
        #     prompt=prompt, grammar=Grammar(type=GrammarType.Json, value=Action.model_json_schema())
        # )
        # generated_text = response.generated_text
        # action = Action(**json.loads(generated_text))
        # _LOGGER.info(f"\n{action}")

        return 0


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
        self._agent = LLM(base_url="http://192.168.0.8:58084")

        # Keep track of env history
        self.env_history = [EnvironmentState(self._env)]

    def _save_checkpoint(self) -> None:
        _LOGGER.warning("Deterministic agents cannot learn")

    def learn(self):
        _LOGGER.warning("Deterministic agents cannot learn")

    def _calculate_action(self, obs: np.ndarray):
        prev_env_state = self.env_history[-1]
        env_state = EnvironmentState(self._env, prev_env_state=prev_env_state)

        action = self._agent.predict(env_state, self.env_history)
        env_state.action = action
        self.env_history.append(env_state)

        return action

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
