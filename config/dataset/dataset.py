from enum import Enum

from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


class RandomAccessMode(Enum):
    serial_access  = 0
    random_blocks  = 1

class shape(Enum):
    sparse = 0
    dense  = 1

@dataclass
class Dataset:
    name:           str        = ""
    dimension:      int        = 3
    access_mode:    RandomAccessMode = RandomAccessMode.random_blocks
    data_directory: str        = MISSING
    file:           str        = ""
    input_shape:    shape      = shape.sparse
    output_shape:   shape      = shape.sparse
    producer:       str        = ""

@dataclass
class dune2d(Dataset):
    name:           str        = "dune2d"
    data_directory: str        = MISSING
    file:           str        = "train.h5"
    dimension:      int        = 2
    channels:       list       = (0,1,2)

@dataclass
class dune3d(Dataset):
    name:           str        = "dune3d"
    data_directory: str        = MISSING
    file:           str        = "train.h5"
    dimension:      int        = 3

# @dataclass
# class next_new(Dataset):
#     data_directory: str        = MISSING
#     file:           str        = "NEXT_White_Tl_train.h5"
#     aux_file:       str        = "NEXT_White_Tl_test.h5"
#     dimension:      int        = 3
#

cs = ConfigStore.instance()
cs.store(group="dataset", name="dune2d",   node=dune2d)
cs.store(group="dataset", name="dune3d",   node=dune3d)
# cs.store(group="dataset", name="next_new", node=next_new)
