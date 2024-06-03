# Displaying the environment state

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st

from primaite.common.enums import FileSystemState, HardwareState, NodeType, RulePermissionType, SoftwareState
from primaite.environment.primaite_env import Primaite
from primaite.nodes.active_node import ActiveNode
from primaite.nodes.service_node import ServiceNode


def display_acl(env: Primaite):
    rules = env.acl.acl

    rules = [r for r in rules if r is not None]

    rules_dict = {
        "Permission": ["ALLOW" if r.permission == RulePermissionType.ALLOW else "DENY" for r in rules],
        "Source IP": [r.source_ip for r in rules],
        "Dest IP": [r.dest_ip for r in rules],
        "Protocol": [r.protocol for r in rules],
        "Port": [r.port for r in rules],
    }

    acl_table = pd.DataFrame(rules_dict, index=None)

    st.table(acl_table)


def display_nodes(env: Primaite):
    nodes = list(env.nodes.values())
    nodes = [n for n in nodes if isinstance(n, ActiveNode)]

    nodes_dict = {
        "Name": [n.name for n in nodes],
        "Type": [n.node_type.name for n in nodes],
        "IP": [n.ip_address for n in nodes],
        "Hardware State": [n.hardware_state.name for n in nodes],
        "Software State": [n.software_state.name for n in nodes],
        "File System State": [n.file_system_state_observed.name for n in nodes],
    }
    nodes_table = pd.DataFrame(nodes_dict, index=None)

    for n in nodes:
        if isinstance(n, ServiceNode):
            print(n.services)

    st.table(nodes_table)


# def display_services(env: Primaite):


def display_network(env: Primaite):
    network = env.network
    fig = plt.figure()
    nx.draw_networkx(network, with_labels=True)

    st.pyplot(fig)
