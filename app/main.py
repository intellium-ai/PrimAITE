# Interactive evaluation app for a primaite session

import logging
import sys
from pathlib import Path

import streamlit as st
from environment import EnvironmentState
from streamlit import session_state as state

from primaite.agents.utils import describe_obs_change
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

if "env_history" not in state:
    state.env_history = [EnvironmentState(env, "")]


input_col, curr_step_col, _ = st.columns([1, 2, 3])
with input_col:
    num_steps = int(st.number_input("Number of steps", value=agent._training_config.num_eval_steps, min_value=0))
    button = st.button("Start PrimAITE simulation")

with curr_step_col:
    if len(state.env_history) > 1:
        st.slider("Current step", min_value=0, max_value=len(state.env_history) - 1, key="curr_step")
    else:
        state.curr_step = 0

env_view = st.empty()

with env_view.container():
    state.env_history[0].display()


# Run the simulation
if button:
    env.set_as_eval()
    prev_obs = env.reset()

    for step in range(1, num_steps + 1):
        env_view.empty()

        # Run simulation

        obs, rewards, done, _ = env.step(0)
        obs_diff = describe_obs_change(
            prev_obs, obs, num_nodes=len(env.nodes), num_links=len(env.links), num_services=len(env.services_list)
        )
        env_state = EnvironmentState(env, obs_diff)
        with env_view.container():
            env_state.display()
        state.env_history.append(env_state)
        state.curr_step = len(state.env_history) - 1

        prev_obs = obs

        if done:
            break
    st.rerun()
else:
    env_view.empty()
    with env_view.container():
        state.env_history[state.curr_step].display()
