from enum import Enum


class TaskLoadType(Enum):
    UNIFORM = 1
    NONUNIFORM = 2

class TaskRelationType(Enum):
    UNRELATED = 1
    PRECEDENCE = 2

class MachineType(Enum):
    SINGLE = 1
    HOMOGENEOUS = 2
    HETEROGENEOUS = 3

class MachineRelationType(Enum):
    UNRELATED = 1
    PRECEDENCE = 2

class DeltaFunctionClass(Enum):
    CONSTANTPROCTIME = 1
    LINESIN = 2
    SAMPLED = 3
