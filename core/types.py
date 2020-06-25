from enum import Enum

class TaskLoadType(Enum):
    UNIFORM = 1
    NONUNIFORM = 2

class TaskRelationType(Enum):
    UNRELATED = 1
    PRECEDENCE = 2

class MachineLoadType(Enum):
    SINGLE = 1
    UNIFORM = 2
    NONUNIFORM = 3

class MachineCapabilityType(Enum):
    HOMOGENEOUS = 1
    HETEROGENEOUS = 2

class MachineRelationType(Enum):
    UNRELATED = 1
    PRECEDENCE = 2

