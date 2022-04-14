import sys, os
from dataclasses import dataclass


@dataclass
class single_node_strong:
    batch_size:        int = 1024
    max_ranks:         int = 32
    warmup_iterations: int = 100
    real_iterations:   int = 100
    dataset_name:      str = ""
    input_shape:       str = "sparse"
    output_shape:      str = "sparse"

@dataclass
class DUNE2D_SPARSE(single_node_strong):
    dataset_name:      str = "dune2d"
    pass

@dataclass
class DUNE2D_DENSE(single_node_strong):
    dataset_name:      str = "dune2d"
    output_shape:      str = "dense"
    pass

@dataclass
class DUNE3D_SPARSE(single_node_strong):
    dataset_name:      str = "dune3d"
    pass

@dataclass
class DUNE3D_DENSE(single_node_strong):
    dataset_name:      str = "dune3d"
    batch_size:        int = 16
    max_ranks:         int = 16
    output_shape:      str = "dense"

@dataclass
class COSMIC_TAGGER_DENSE(single_node_strong):
    dataset_name:      str = "cosmic_tagger_dense"
    batch_size:        int = 32
    max_ranks:         int = 32
    input_shape:       str = "dense"
    output_shape:      str = "dense"
    pass

@dataclass
class COSMIC_TAGGER_SPARSE_SPARSE(single_node_strong):
    dataset_name:      str = "cosmic_tagger_sparse"
    pass

@dataclass
class COSMIC_TAGGER_SPARSE_DENSE(single_node_strong):
    dataset_name:      str = "cosmic_tagger_sparse"
    batch_size:        int = 64
    output_shape:      str = "dense"
    pass

datasets = [
    'DUNE2D_SPARSE',
    'DUNE2D_DENSE',
    'DUNE3D_SPARSE',
    'DUNE3D_DENSE',
    'COSMIC_TAGGER_DENSE',
    'COSMIC_TAGGER_SPARSE_SPARSE',
    'COSMIC_TAGGER_SPARSE_DENSE',
]
