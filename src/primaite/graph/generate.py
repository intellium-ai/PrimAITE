import random
from pathlib import Path
from typing import List, Dict, Any

import yaml

from primaite.common.enums import (
    FileSystemState,
    HardwareState,
    NodeType,
    Priority,
    SoftwareState,
)
from primaite.graph.types import Link, Node, NodeService, Service


class NetworkGenerator:

    services: List[Service]
    ports: List[int]
    nodes: List[Node]
    links: List[Link]

    def __init__(self, num_nodes: int, num_services: int, num_links: int) -> None:
        self.num_nodes = num_nodes
        self.num_services = num_services
        self.num_links = num_links

    def add_ports_and_services(self, low=1, high=99) -> None:
        if self.num_services > len(list(Service)):
            raise ValueError

        self.services = random.sample(list(Service), self.num_services)
        self.ports = random.sample(range(low, high), self.num_services)

    def add_nodes(self) -> None:
        self.nodes = []

        if not hasattr(self, "services"):
            self.add_ports_and_services()

        node_types = list(NodeType)

        for i in range(1, self.num_nodes + 1):

            node_type_idx = (i - 1) % len(node_types)
            node_type = node_types[node_type_idx]

            node_service = NodeService(
                name=self.services[0],
                port=self.ports[0],
                state=SoftwareState.GOOD,
            )

            node = Node(
                node_id=i,
                name="PC",
                node_type=node_type,
                priority=Priority.P1,
                hardware_state=HardwareState.ON,
                ip_address=f"192.168.0.{i}",
                software_state=SoftwareState.GOOD,
                file_system_state=FileSystemState.GOOD,
                services=[node_service],
            )

            self.nodes.append(node)

    def add_links(self):
        self.links = []

        for i in range(self.num_nodes + 1, self.num_nodes + self.num_links + 1):
            link = Link(id=i, name=f"LINK_{i - self.num_nodes}", source=1, destination=2)
            self.links.append(link)

    def _ports_to_dict(self) -> Dict[str, Any]:
        port_config = [{"port": f"{port}"} for port in self.ports]
        return {"item_type": "PORTS", "ports_list": port_config}

    def _services_to_dict(self) -> Dict[str, Any]:
        service_config = [{"name": service.value} for service in self.services]
        return {"item_type": "SERVICES", "service_list": service_config}

    def _nodes_to_dict(self) -> List[Dict[str, Any]]:
        return [node.dict() for node in self.nodes]

    def _links_to_dict(self) -> List[Dict[str, Any]]:
        return [link.dict() for link in self.links]

    def dict(self) -> List[Dict[str, Any]]:
        config = []
        config.append(self._ports_to_dict())
        config.append(self._services_to_dict())
        config.extend(self._nodes_to_dict())
        config.extend(self._links_to_dict())

        return config

    def save(self, path: str | Path = "test.yaml") -> None:

        if not hasattr(self, "nodes"):
            self.add_ports_and_services()
            self.add_nodes()
            self.add_links()

        with open(path, "w+") as f:
            yaml.safe_dump(self.dict(), f)


def main():

    network = NetworkGenerator(num_nodes=3, num_services=2, num_links=1)
    network.save("src/primaite/config/_package_data/lay_down/test.yaml")


if __name__ == "__main__":
    main()
