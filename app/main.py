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
num_steps = agent._training_config.num_eval_steps

if "env_history" not in state:
    state.env_history = [EnvironmentState(env)]


input_col, curr_step_col, _ = st.columns([1, 2, 3])
with input_col:
    button = st.button("Start PrimAITE simulation")

with curr_step_col:
    if len(state.env_history) > 1:
        state.curr_step = st.slider("Current step", min_value=0, max_value=len(state.env_history) - 1)
    else:
        state.curr_step = 0
        state.total_reward = 0

env_view = st.empty()


# Run the simulation
if button:
    env.set_as_eval()
    obs = env.reset()
    prev_env_state = EnvironmentState(env)
    state.env_history = [prev_env_state]
    state.total_reward = 0

    for step in range(1, num_steps + 1):
        env_view.empty()

        # Run simulation

        obs, rewards, done, _ = env.step(0)
        state.total_reward += rewards

        env_state = EnvironmentState(env, prev_env_state)
        with env_view.container():
            st.markdown(f"Step :orange[{step}]&emsp; Avg Reward: :orange[{round(state.total_reward / step, 5)}]")
            env_state.display()
        state.env_history.append(env_state)

        prev_env_state = env_state

        if done:
            break
    st.rerun()
else:
    env_view.empty()
    with env_view.container():
        if len(state.env_history) == 1:
            st.write("Initial environment")
        else:
            st.write(f"Avg Reward for Episode: :orange[{round(state.total_reward / (len(state.env_history) - 1), 5)}]")
        state.env_history[state.curr_step].display()
