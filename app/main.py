# Interactive evaluation app for a primaite session

import logging
import sys
from pathlib import Path

import streamlit as st
from environment import display_acl, display_network, display_nodes, display_traffic
from streamlit import session_state as state

from primaite.agents.utils import transform_nodestatus_readable
from primaite.primaite_session import PrimaiteSession

st.set_page_config(layout="wide")

config_path = Path("../src/primaite/config/_package_data/")
training_config_path = config_path / "training" / "training_config_main.yaml"
lay_down_config_path = config_path / "lay_down" / "lay_down_config_5_data_manipulation.yaml"

session = PrimaiteSession(training_config_path, lay_down_config_path)
session.setup()

agent = session._agent_session

env = agent._env

obs = env.reset()
env.step(0)

col_table, col_graph, _ = st.columns(3)
with col_table:
    display_nodes(env)
    display_traffic(env)
with col_graph:
    display_network(env)


# num_steps = agent._training_config.num_eval_steps
# col1, col2, _, _, _, _ = st.columns(6)
# with col2:
#     num_steps = int(st.number_input("Number of steps", value=1, min_value=0))
# with col1:
#     button = st.button("Start PrimAITE simulation")

# if button:
#     env.set_as_eval()
#     canvas = st.empty()

#     obs = env.reset()
#     for step in range(1, num_steps + 1):
#         canvas.empty()

#         # Run simulation

#         obs, rewards, done, _ = env.step(0)
#         print(obs)

#         # Display updated environment
#         with canvas.container():
#             st.write(f"Step: {step}")
#             col_table, col_graph, _ = st.columns(3)
#             with col_table:
#                 display_nodes(env)
#             with col_graph:
#                 display_network(env)

#             display_traffic(env)
