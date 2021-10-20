from enum import Enum

from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


from .dataset.dataset import dune2d, dune3d

@dataclass
class Run:
    distributed:        bool        = True
    iterations:         int         = 50
    minibatch_size:     int         = 4
    dataset:            dataset     = dune2d

cs = ConfigStore.instance()
cs.store(name="run", node=Run)
