import json
from logging import Logger
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel
from text_generation import Client
from text_generation.types import Grammar, GrammarType

from primaite import getLogger
from primaite.agents.agent_abc import AgentSessionABC
from primaite.common.enums import AgentFramework, AgentIdentifier
from primaite.environment.primaite_env import Primaite
from primaite.agents.utils import describe_obs_change, transform_obs_readable, convert_to_new_obs, convert_to_old_obs

_LOGGER: Logger = getLogger(__name__)


class Action(BaseModel):
    chosen_action: int


class LLM:
    def __init__(self, base_url: str, timeout: int = 20) -> None:
        self.client = Client(base_url=base_url, timeout=timeout)

    def predict(self, observation: np.ndarray, deterministic: bool, **kwargs) -> int:

        max_new_tokens = kwargs.pop("max_new_tokens", 64)
        repetition_penalty = kwargs.pop("repetition_penalty", 1.5)

        grammar = Grammar(type=GrammarType.Json, value=Action.model_json_schema())
        prompt = f"""
        Predict the best action to take based on the following observation space
        
        {observation}
        
        Action:
        """

        response = self.client.generate(
            prompt=prompt,
            grammar=grammar,
            do_sample=deterministic,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
        )
        action = Action(**json.loads(response.generated_text))
        return action.chosen_action


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

    def _save_checkpoint(self) -> None:
        checkpoint_n = self._training_config.checkpoint_every_n_episodes
        episode_count = self._env.episode_count
        save_checkpoint = False
        if checkpoint_n:
            save_checkpoint = episode_count % checkpoint_n == 0
        if episode_count and save_checkpoint:
            checkpoint_path = self.checkpoints_path / f"sb3ppo_{episode_count}.zip"
            # self._agent.save(checkpoint_path)
            _LOGGER.debug(f"(Mock) Saved agent checkpoint: {checkpoint_path}")

    def learn(self):
        # call your agent's learning function here.

        _LOGGER.warning("Deterministic agents cannot learn")

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

        if self._training_config.deterministic:
            deterministic_str = "deterministic"
        else:
            deterministic_str = "non-deterministic"
        _LOGGER.info(
            f"Beginning {deterministic_str} evaluation for " f"{episodes} episodes @ {time_steps} time steps..."
        )

        for _ in range(episodes):
            obs = self._env.reset()

            for _ in range(time_steps):
                action = self._agent.predict(obs, deterministic=self._training_config.deterministic, **kwargs)
                new_obs, rewards, done, info = self._env.step(action=action)
                _LOGGER.info(obs)
                msg = describe_obs_change(
                    obs1=obs,
                    obs2=new_obs,
                    num_nodes=self._env.num_nodes,
                    num_links=self._env.num_links,
                    num_services=self._env.num_services,
                )
                _LOGGER.info(msg=msg)
                obs = new_obs

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
