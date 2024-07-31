# CC-Collisions

This is a CUDA module for pytorch which implements static and dynamic collision handling.
It is used by [ContourCraft](https://github.com/dolorousrtur/contourcraft) paper (SIGGRAPH 2024).


## Installation

1. Clone https://github.com/NVIDIA/cuda-samples

2. set `CUDA_SAMPLES_INC` and run pip install:

```
export CUDA_SAMPLES_INC=/path/to/cuda-samples/Common/
pip install .
```

## License

Parts of this repo are based on [torch-mesh-isect](https://github.com/vchoutas/torch-mesh-isect) repository. Hence its license also applies to this repo.