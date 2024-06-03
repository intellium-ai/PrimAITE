# Interactive evaluation app for a primaite session

import logging
import sys
from pathlib import Path

import streamlit as st
from environment import display_acl, display_network, display_nodes
from streamlit import session_state as state

from primaite.agents.utils import transform_nodestatus_readable
from primaite.primaite_session import PrimaiteSession

config_path = Path("../src/primaite/config/_package_data/")
training_config_path = config_path / "training" / "training_config_main.yaml"
lay_down_config_path = config_path / "lay_down" / "lay_down_config_5_data_manipulation.yaml"

session = PrimaiteSession(training_config_path, lay_down_config_path)
session.setup()

agent = session._agent_session

env = agent._env

display_acl(env)

display_network(env)

display_nodes(env)


obs = env.reset()
components = env.obs_handler.registered_obs_components


# for c in components:
#     print(c.__dict__)


# num_steps = agent._training_config.num_eval_steps
