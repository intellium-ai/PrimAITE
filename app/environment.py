# Displaying the environment state
from __future__ import annotations

import copy
from dataclasses import dataclass

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st

from primaite.agents.env_state import _verbose_node_action, EnvironmentState


def display_env_state(env_state: EnvironmentState):
    col_agent, col_env = st.columns([3, 5], gap="medium")
    with col_env:
        st.write("**Environment view:**")
        st.table(env_state.nodes_table)
        col_net, col_links = st.columns([3, 2])
        with col_links:
            st.table(env_state.traffic_table)
        with col_net:
            fig = env_state.display_network()
            st.pyplot(fig)
    with col_agent:
        if env_state.action is not None:
            st.write("**:blue[Blue Agent:]**")
            action_verbose = _verbose_node_action(env_state.action, env_state.env)
            st.markdown(action_verbose)

            if env_state.info is not None:
                with st.expander("Info"):
                    st.write(env_state.info)
            st.divider()

        st.write("**Observation Space Changes:**")
        for change in env_state.obs_diff(colors=True):
            st.markdown(change)
