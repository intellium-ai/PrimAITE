from typing import Literal

from pydantic import BaseModel, Field

from primaite.action import NodeAction
from primaite.environment import Primaite

NODE_ACTION_SPACE_DESCRIPTION = """
As an agent, you are able to influence the status of nodes by switching them off, resetting, or patching operating systems and services. Every turn, you may choose to take an action on one of the nodes. If choosing not to take any action, return nothing. Note that actions are expensive and can negatively impact the environment if used improperly. For instance, a server which is turned off cannot receive requests from the users and will decrease the reward.  

Before taking the action, provide a quick explanation for your reason, given the observation space provided.

If choosing to take an action, you must select a node by name, from the following list: {node_names}. Assuming the selected node name is NODE_NAME, the following are the valid actions you can take:

Action: node_name=NODE_NAME node_property='HARDWARE' property_action='TURN_ON'
Description: Boots the node. Only applicable if node was turned off.

Action: node_name=NODE_NAME node_property='HARDWARE' property_action='TURN_OFF'
Description: Turn the node off. 

Action: node_name=NODE_NAME node_property='HARDWARE' property_action='RESET'
Description: Resets the hardware after a number of steps. Only applicable if node is turned on. This also resets the status of the software, file system and services to 'GOOD'.

The following actions are only applicable if the node is turned on:

Action: node_name=NODE_NAME node_property='SOFTWARE' property_action='PATCH'
Description: Patches the software for a number of steps, after which the status of software returns to 'GOOD'.

Action: node_name=NODE_NAME node_property='FILE_SYSTEM' property_action='SCAN'
Description: Scan the node to reveal the actual status of the file system. After a number of steps, the observed file system status will be updated to reflect the true status.

Action: node_name=NODE_NAME node_property='FILE_SYSTEM' property_action='REPAIR'
Description: Repairs the file system, setting it back to 'GOOD' after a number of steps. Cannot be applied if file system is 'DESTROYED'.

Action: node_name=NODE_NAME node_property='FILE_SYSTEM' property_action='RESTORE'
Description: Restores the file system, setting it back to 'GOOD' after a number of steps. It can be applied even if the file system is 'DESTROYED', but it is more expensive to use than 'REPAIR'.

The following actions are only applicable if the node owns services. If choosing to take a service action, you must select the service by name, from the following list: {service_names}. Be mindful that a node may own only a subset of these services. Assuming the selected service name is SERVICE_NAME, the following are service actions you can take: 

Action: node_name=NODE_NAME node_property='SERVICE' property_action='PATCH' service_name=SERVICE_NAME
Description: Patches the service for a number of steps, after which the status of the service returns to 'GOOD'.


"""


def get_action_space_description(env: Primaite) -> str:
    node_names = [n.name for n in env.active_nodes]
    service_names = env.services_list

    return NODE_ACTION_SPACE_DESCRIPTION.format(node_names=node_names, service_names=service_names)


class AgentNodeAction(BaseModel):
    # reasoning: str = Field(..., description="Rationale for taking the action.")
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
