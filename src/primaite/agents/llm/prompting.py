from typing import Literal

from pydantic import BaseModel, Field
from termcolor import colored
from primaite.action import NodeAction
from primaite.environment import Primaite
from primaite.environment.env_state import EnvironmentState


class AgentNodeAction(BaseModel):
    reasoning: str = Field(..., description="Rationale for taking the action.")
    node_name: str = Field("NONE", description="Node to apply action to.")
    node_property: Literal["NONE", "HARDWARE", "SOFTWARE", "FILE_SYSTEM", "SERVICE"] = Field(
        "NONE", description="Node property to apply action to."
    )
    property_action: str = Field("NONE", description="Action to take on the given node property.")
    service_name: str = Field(
        "NONE", description="Service to apply action to. Only applicable if node_property is SERVICE."
    )

    def to_node_action(self, env: Primaite) -> NodeAction:
        return NodeAction.from_text(
            env=env,
            node_name=self.node_name,
            node_property=self.node_property,
            property_action=self.property_action,
            service_name=self.service_name,
        )


SYSTEM_MSG = "You are a cybersecurity network defence agent. Your mission is to protect a network against offensive attacks. You have a limited view of the network environment, known as an observation space. The network is made up of nodes (e.g. computers, switches) and links between some of the nodes, through which information is transmitted using standard protocols. The current link traffic is displayed as a percentage of its total bandwidth, for each possible protocol. An attacker is trying to compromise the network, by either directly incapacitating the nodes HARDWARE, SOFTWARE or FILE_SYSTEM states, or indirectly overwhelming the nodes through Denial-of-Service type attacks.\n"

HISTORY_PREFIX = "This is the history of offensive action observations and defensive actions you have taken at each step. If nothing happened at a specific step, it is omitted from the history.\n"

CURRENT_OBS = """Here is an overview of the current observation space:
{obs_view_full}
Changes that have occured since last observation are: {obs_diff} """
NODE_ACTION_SPACE_DESCRIPTION = """
As an agent, you are able to influence the state of nodes by switching them off or resetting them and patching operating systems and services. Every turn, you may choose to take an action on one of the nodes. If choosing not to take any action, return NONE. Note that actions are expensive and can negatively impact the environment if used improperly. For instance, a server which is turned off cannot receive requests from the users and will decrease the reward.  

Before taking the action, provide a quick explanation for your reason given the observation space provided.

If choosing to take an action, you must select a node by name from the following list: {node_names}. Always take note of any action constraints outlined in the description provided.

## HARDWARE Actions:
Action: {{'node_name':'NODE_NAME', 'node_property':'HARDWARE', 'property_action':'TURN_ON'}}
Description: If it is currently off, will turn it on.

Action: {{'node_name':'NODE_NAME', 'node_property':'HARDWARE', 'property_action':'TURN_OFF'}}
Description: If it is currently on, will turn it off.

Action: {{'node_name':'NODE_NAME', 'node_property':'HARDWARE', 'property_action':'RESET'}}
Description: Resets the hardware after a number of steps. Only works if the node is turned on. Resets the status of the software, file system and services to 'GOOD'.

## SOFTWARE Actions:
Action: {{'node_name':'NODE_NAME', 'node_property':'SOFTWARE', 'property_action':'PATCH'}}
Description: Patches the software for a number of steps, after which the status of software returns to 'GOOD'.

## SERVICE Actions:
The following actions are only applicable if the node owns services. If choosing to take a SERVICE action, you must select the SERVICE by name from the following list: {service_names}. Be mindful that a node may own only a subset of these services. Assuming the selected service name is 'SERVICE_NAME', the following are service actions you can take: 

Action: {{'node_name':'NODE_NAME', 'node_property':'SERVICE', 'property_action':'PATCH', 'service_name':'SERVICE_NAME'}}
Description: Patches the service for a number of steps, after which the status of the service returns to 'GOOD'.

"""

ACTION_PROMPT = """
Please take a suitable action.
Action: """


# (JOHN) - File system action descriptions incase we want to use them again in the future.
# ## FILE_SYSTEM Actions:
# Action: {{'node_name':'NODE_NAME', 'node_property':'FILE_SYSTEM', 'property_action':'SCAN'}}
# Description: Scan the file system to reveal the actual status of the file system. After a number of steps, the observed nodes file system status will be updated to reflect the true status.

# Action: {{'node_name':'NODE_NAME', 'node_property':'FILE_SYSTEM', 'property_action':'REPAIR'}}
# Description: Repairs the file system, setting it back to 'GOOD' after a number of steps. Cannot be applied if file system is 'DESTROYED'.


# Action: {{'node_name':'NODE_NAME', 'node_property':'FILE_SYSTEM', 'property_action':'RESTORE'}}
# Description: Restores the file system, setting it back to 'GOOD' after a number of steps. It can be applied even if the file system is 'DESTROYED', but it is more expensive to use than 'REPAIR'.
