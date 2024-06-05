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


def display_traffic(env: Primaite):
    links = list(env.links.values())

    indices = [int(link.id) for link in links]

    protocols = env.services_list
    traffic_dict = {
        protocol + " Traffic": [link.get_current_protocol_load(protocol) for link in links] for protocol in protocols
    }
    link_dict = {"Name": list(env.links.keys()), "Bandwidth": [link.bandwidth for link in links]}

    traffic_table = pd.DataFrame(link_dict | traffic_dict)
    traffic_table.index = indices  # type: ignore
    st.table(traffic_table)


def display_nodes(env: Primaite):
    nodes = list(env.nodes.values())
    nodes = [n for n in nodes if isinstance(n, ActiveNode)]

    nodes_dict = {
        "Name": [n.name for n in nodes],
        # "Type": [n.node_type.name for n in nodes],
        # "IP": [n.ip_address for n in nodes],
        "Hardware State": [n.hardware_state.name for n in nodes],
        "Software State": [n.software_state.name for n in nodes],
        "File System State": [n.file_system_state_observed.name for n in nodes],
    }

    protocols = env.services_list
    services_dict = {
        protocol
        + " State": [
            node.get_service_state(protocol).name if isinstance(node, ServiceNode) else SoftwareState.NONE.name
            for node in nodes
        ]
        for protocol in protocols
    }

    nodes_table = pd.DataFrame(nodes_dict | services_dict)
    nodes_table.index += 1
    nodes_table.replace({"NONE": "-"}, inplace=True)

    st.table(nodes_table)


# def display_services(env: Primaite):


def display_network(env: Primaite):

    G = env.network.copy()

    # Make sure node locations in plot are constant
    pos = nx.spring_layout(G, seed=100)
    fig = plt.figure()
    nx.draw_networkx(G, pos=pos, with_labels=False)

    pos_higher = {}
    y_off = 0.05  # offset on the y axis

    for k, v in pos.items():
        pos_higher[k] = (v[0], v[1] + y_off)

    nx.draw_networkx_labels(G, pos=pos_higher, font_size=5)

    print(G.edges)
    edge_labels = {}

    nx.draw_networkx_edges(G, pos=pos)

    st.pyplot(fig)
