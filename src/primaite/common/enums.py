# Crown Copyright (C) Dstl 2022. DEFCON 703. Shared in confidence.
"""Enumerations for APE."""

from enum import Enum


class NodeType(Enum):
    """Node type enumeration."""

    CCTV = 1
    SWITCH = 2
    COMPUTER = 3
    LINK = 4
    MONITOR = 5
    PRINTER = 6
    LOP = 7
    RTU = 8
    ACTUATOR = 9
    SERVER = 10


class Priority(Enum):
    """Node priority enumeration."""

    P1 = 1
    P2 = 2
    P3 = 3
    P4 = 4
    P5 = 5


class HardwareState(Enum):
    """Node hardware state enumeration."""

    ON = 1
    OFF = 2
    RESETTING = 3


class SoftwareState(Enum):
    """Software or Service state enumeration."""

    GOOD = 1
    PATCHING = 2
    COMPROMISED = 3
    OVERWHELMED = 4


class NodePOLType(Enum):
    """Node Pattern of Life type enumeration."""

    NONE = 0
    OPERATING = 1
    OS = 2
    SERVICE = 3
    FILE = 4


class NodePOLInitiator(Enum):
    """Node Pattern of Life initiator enumeration."""

    DIRECT = 1
    IER = 2
    SERVICE = 3


class Protocol(Enum):
    """Service protocol enumeration."""

    LDAP = 0
    FTP = 1
    HTTPS = 2
    SMTP = 3
    RTP = 4
    IPP = 5
    TCP = 6
    NONE = 7


class ActionType(Enum):
    """Action type enumeration."""

    NODE = 0
    ACL = 1
    ANY = 2


class ObservationType(Enum):
    """Observation type enumeration."""

    BOX = 0
    MULTIDISCRETE = 1


class FileSystemState(Enum):
    """File System State."""

    GOOD = 1
    CORRUPT = 2
    DESTROYED = 3
    REPAIRING = 4
    RESTORING = 5


class NodeHardwareAction(Enum):
    """Node hardware action."""

    NONE = 0
    ON = 1
    OFF = 2
    RESET = 3


class NodeSoftwareAction(Enum):
    """Node software action."""

    NONE = 0
    PATCHING = 1


class LinkStatus(Enum):
    """Link traffic status."""

    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    OVERLOAD = 4


class RulePermissionType(Enum):
    """Implicit firewall rule."""

    DENY = 0
    ALLOW = 1
