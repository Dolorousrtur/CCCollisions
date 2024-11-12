# CCCollisions

This is a CUDA module for pytorch which implements static and dynamic collision handling.
It is used by [ContourCraft](https://github.com/dolorousrtur/contourcraft) paper (SIGGRAPH 2024).

The code is quite messy at the moment so sorry about that.

## Installation

1. Clone https://github.com/NVIDIA/cuda-samples

2. set `CUDA_SAMPLES_INC` and run pip install:

```
export CUDA_SAMPLES_INC=/path/to/cuda-samples/Common/
pip install .
```

### Full environment installation with conda

```
conda create -n ccc python=3.9 -y
conda activate ccc
conda install conda-forge::gcc=11.4.0 -y
conda install conda-forge::gxx=11.4.0 -y
conda install nvidia::cuda-toolkit=12.4.1 -y
conda install pytorch=2.5.1 torchvision=0.20.1 torchaudio=2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia -y

WORKDIR=$(pwd)
git clone git@gitlab.inf.ethz.ch:agrigorev/cccollision_pbs2024.git
git clone https://github.com/NVIDIA/cuda-samples.git


# This path has to contain file "cuda_runtime.h"
export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CPATH

cd cccollision_pbs2024/ccollisions/
export CUDA_SAMPLES_INC=$WORKDIR/cuda-samples/Common/

pip install .
```

## License

Parts of this repo are based on [torch-mesh-isect](https://github.com/vchoutas/torch-mesh-isect) repository. Hence its license also applies to this repo.
