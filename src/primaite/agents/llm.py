from logging import Logger
from pathlib import Path
from typing import Any

import numpy as np
from text_generation import Client

from primaite import getLogger
from primaite.agents.agent_abc import AgentSessionABC
from primaite.common.enums import AgentFramework, AgentIdentifier
from primaite.environment.primaite_env import Primaite
from primaite.agents.utils import transform_nodelink_readable, transform_nodestatus_readable, split_obs_space

_LOGGER: Logger = getLogger(__name__)


class LLM:
    def __init__(self, base_url: str, timeout: int = 60) -> None:
        self.client = Client(base_url=base_url, timeout=timeout)

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool,
        num_nodes: int,
        num_links: int,
        num_services: int,
    ) -> int:

        _LOGGER.info(f"\n{observation}")
        _LOGGER.info(f"\n{transform_nodelink_readable(observation)}")

        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

        You will be given a dictionary containing information about nodes in a computer network as a dictionary.
        You will be given all the information applicable to each node.

        <|eot_id|><|start_header_id|>user<|end_header_id|>
        
        Based purely on the description you have been given, say if any node has been compromised. If you do not see any issues, return 'ok'.

        Network:
        
        {observation}
        
        Vulnerabilities:<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

        # response = self.client.generate(
        #     prompt=prompt,
        #     do_sample=False,
        #     repetition_penalty=1.2,
        #     max_new_tokens=256,
        # )

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

    def _save_checkpoint(self) -> None:
        _LOGGER.warning("Deterministic agents cannot learn")

    def learn(self):
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

        _LOGGER.info(f"Num services: {self._env.num_services}")

        for _ in range(episodes):
            obs = self._env.reset()
            done, steps, rew = False, 0, 0
            while steps < time_steps and not done:

                action = self._agent.predict(
                    obs,
                    deterministic=self._training_config.deterministic,
                    num_nodes=self._env.num_nodes,
                    num_links=self._env.num_links,
                    num_services=self._env.num_services,
                )

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
