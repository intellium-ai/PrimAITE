import yaml
from typing import List, Final
from dataclasses import dataclass
import itertools
from enum import Enum
from primaite.common.enums import HardwareState, FileSystemState, SoftwareState


class Service(Enum):
    TCP = "TCP"
    UDP = "UDP"


@dataclass
class Node:

    def __init__(
        self,
        node_id: int,
        name: str,
        node_class: str,
        node_type: str,
        priority: str,
        hardware_state: HardwareState,
        ip_address: str,
        software_state: SoftwareState,
        file_system_state: FileSystemState,
    ) -> None:
        self.item_type: Final[str] = "NODE"
        self.node_id = node_id
        self.name = name
        self.node_class = node_class
        self.priority = priority
        self.node_type = node_type
        self.hardware_state = hardware_state.name
        self.ip_address = ip_address
        self.software_state = software_state.name
        self.file_system_state = file_system_state.name


def generate_ports(ports: List[int]):
    port_config = [{"port": f"{port}"} for port in ports]
    return {"item_type": "PORTS", "ports_list": port_config}


def generate_services(services: List[Service]):
    service_config = [{"name": service.value} for service in services]
    return {"item_type": "SERVICES", "service_list": service_config}


def generate_nodes(num_nodes: int):
    return [node.__dict__ for node in generate_random_nodes(num_nodes)]


def generate_random_nodes(num_nodes: int) -> List[Node]:
    nodes = []
    for i in range(1, num_nodes + 1):
        node = Node(
            node_id=i,
            name="PC",
            node_class="ACTIVE",
            node_type="SWITCH",
            priority="P1",
            hardware_state=HardwareState.ON,
            ip_address=f"192.168.0.{i}",
            software_state=SoftwareState.GOOD,
            file_system_state=FileSystemState.GOOD,
        )
        nodes.append(node)

    return nodes


def main():
    PORTS = [80, 99]
    SERVICES = [Service.TCP, Service.UDP]

    config = []

    config.append(generate_ports(PORTS))
    config.append(generate_services(SERVICES))
    config.extend(generate_nodes(3))

    with open("test.yaml", "w+") as f:
        yaml.safe_dump(config, f)


if __name__ == "__main__":
    main()
