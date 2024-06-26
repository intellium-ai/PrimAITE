# © Crown-owned copyright 2023, Defence Science and Technology Laboratory UK
from logging import Logger
from pathlib import Path
from typing import Any, Dict, Final, List, Union

import yaml

from primaite import getLogger, PRIMAITE_PATHS

_LOGGER: Logger = getLogger(__name__)

_EXAMPLE_LAY_DOWN: Final[Path] = PRIMAITE_PATHS.user_config_path / "example_config" / "lay_down"


def convert_legacy_lay_down_config(legacy_config: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert a legacy lay down config to the new format.

    :param legacy_config: A legacy lay down config.
    """
    field_conversion_map = {
        "itemType": "item_type",
        "portsList": "ports_list",
        "serviceList": "service_list",
        "baseType": "node_class",
        "nodeType": "node_type",
        "hardwareState": "hardware_state",
        "softwareState": "software_state",
        "startStep": "start_step",
        "endStep": "end_step",
        "fileSystemState": "file_system_state",
        "ipAddress": "ip_address",
        "missionCriticality": "mission_criticality",
    }
    new_config = []
    for item in legacy_config:
        if "itemType" in item:
            if item["itemType"] in ["ACTIONS", "STEPS"]:
                continue
        new_dict = {}
        for key in item.keys():
            conversion_key = field_conversion_map.get(key)
            if key == "id" and "itemType" in item:
                if item["itemType"] == "NODE":
                    conversion_key = "node_id"
            if conversion_key:
                new_dict[conversion_key] = item[key]
            else:
                new_dict[key] = item[key]
        new_config.append(new_dict)
    return new_config


def load(file_path: Union[str, Path], legacy_file: bool = False):
    """
    Read in a lay down config yaml file.

    :param file_path: The config file path.
    :param legacy_file: True if the config file is legacy format, otherwise False.
    :return: The lay down config as a dict.
    :raises ValueError: If the file_path does not exist.
    """
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    if file_path.exists():
        with open(file_path, "r") as file:
            config = yaml.safe_load(file)
            _LOGGER.debug(f"Loading lay down config file: {file_path}")
        if legacy_file:
            try:
                config = convert_legacy_lay_down_config(config)
            except KeyError:
                msg = (
                    f"Failed to convert lay down config file {file_path} "
                    f"from legacy format. Attempting to use file as is."
                )
                _LOGGER.error(msg)
        return config
    msg = f"Cannot load the lay down config as it does not exist: {file_path}"
    _LOGGER.error(msg)
    raise ValueError(msg)


def ddos_basic_one_config_path() -> Path:
    """
    The path to the example lay_down_config_1_DDOS_basic.yaml file.

    :return: The file path.
    """
    path = _EXAMPLE_LAY_DOWN / "lay_down_config_1_DDOS_basic.yaml"
    if not path.exists():
        msg = "Example config not found. Please run 'primaite setup'"
        _LOGGER.critical(msg)
        raise FileNotFoundError(msg)

    return path


def ddos_basic_two_config_path() -> Path:
    """
    The path to the example lay_down_config_2_DDOS_basic.yaml file.

    :return: The file path.
    """
    path = _EXAMPLE_LAY_DOWN / "lay_down_config_2_DDOS_basic.yaml"
    if not path.exists():
        msg = "Example config not found. Please run 'primaite setup'"
        _LOGGER.critical(msg)
        raise FileNotFoundError(msg)

    return path


def dos_very_basic_config_path() -> Path:
    """
    The path to the example lay_down_config_3_DOS_very_basic.yaml file.

    :return: The file path.
    """
    path = _EXAMPLE_LAY_DOWN / "lay_down_config_3_DOS_very_basic.yaml"
    if not path.exists():
        msg = "Example config not found. Please run 'primaite setup'"
        _LOGGER.critical(msg)
        raise FileNotFoundError(msg)

    return path


def data_manipulation_config_path() -> Path:
    """
    The path to the example lay_down_config_5_data_manipulation.yaml file.

    :return: The file path.
    """
    path = _EXAMPLE_LAY_DOWN / "lay_down_config_5_data_manipulation.yaml"
    if not path.exists():
        msg = "Example config not found. Please run 'primaite setup'"
        _LOGGER.critical(msg)
        raise FileNotFoundError(msg)

    return path
