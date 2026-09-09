from enum import Enum


class CacheType(Enum):
    NONE = "none"
    MEMORY = "memory"
    DISK = "disk"


class ModelTier(Enum):
    LOCAL = "LOCAL"
    PREMIUM = "PREMIUM"
