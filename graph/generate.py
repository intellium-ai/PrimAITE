import random
from typing import List

import yaml
from primaite.common.enums import FileSystemState, HardwareState, SoftwareState

from graph.types import Link, Node, NodeService, Service


class ConfigGenerator:

    services: List[Service]
    ports: List[int]

    def __init__(self, num_nodes: int, num_services: int, num_links: int) -> None:
        self.num_nodes = num_nodes
        self.num_services = num_services
        self.num_links = num_links

    def get_services(self, low=1, high=99) -> None:
        if self.num_services > len(list(Service)):
            raise ValueError

        self.services = random.sample(list(Service), self.num_services)
        self.ports = random.sample(range(low, high), self.num_services)

    def get_nodes(self) -> None:
        pass


def generate_ports(ports: List[int]):
    port_config = [{"port": f"{port}"} for port in ports]
    return {"item_type": "PORTS", "ports_list": port_config}


def generate_random_ports(num_ports: int, low=1, high=99):
    ports = [random.randint(low, high) for _ in range(num_ports)]
    return generate_ports(ports=ports)


def generate_services(services: List[Service]):
    service_config = [{"name": service.value} for service in services]
    return {"item_type": "SERVICES", "service_list": service_config}


def generate_random_services(num_services: int):
    if num_services > len(list(Service)):
        raise ValueError("More services requested than available")

    services = list(Service)[:num_services]
    return generate_services(services=services)


def generate_random_nodes(num_nodes: int):
    return [node.dict() for node in generate_nodes(num_nodes)]


def generate_nodes(num_nodes: int) -> List[Node]:
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
            services=[NodeService(name=Service.TCP, port=99, state=SoftwareState.GOOD)],
        )
        nodes.append(node)

    return nodes


def generate_random_links(num_links: int):
    links = []
    for i in range(1, num_links + 1):
        link = Link(id=i, name=f"link{i}", source=1, destination=2)
        links.append(link)
    return [link.dict() for link in links]


def main():

    config = []
    config.append(generate_random_ports(2))
    config.append(generate_random_services(2))
    config.extend(generate_random_nodes(3))
    config.extend(generate_random_links(6))

    with open("test.yaml", "w+") as f:
        yaml.safe_dump(config, f)


if __name__ == "__main__":
    main()
