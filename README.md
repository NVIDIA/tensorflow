<div align="center">
  <img src="https://www.tensorflow.org/images/tf_logo_social.png" alt="TensorFlow Logo">
</div>

# NVIDIA TensorFlow

| **`Documentation`** |
|-----------------|
| [![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](https://www.tensorflow.org/api_docs/) |

NVIDIA has created this project to support newer hardware and improved libraries for NVIDIA GPU users who are using TensorFlow 1.x. With the release of TensorFlow 2.0, Google announced that new major releases would not be provided on the TF 1.x branch after the release of TF 1.15 on October 14, 2019. NVIDIA is working with Google and the community to improve TensorFlow 2.x by adding support for new hardware and libraries. However, a significant number of NVIDIA GPU users still use TensorFlow 1.x in their software ecosystems. This release will maintain API compatibility with the upstream TensorFlow 1.15 release and will be referred to as `nvidia-tensorflow`.

For more details, refer to the official TensorFlow [README](https://github.com/tensorflow/tensorflow).

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Build From Source](#build-from-source)
  - [Fetch Sources and Install Build Dependencies](#fetch-sources-and-install-build-dependencies)
  - [Install NVIDIA Libraries](#install-nvidia-libraries)
  - [Configure TensorFlow](#configure-tensorflow)
  - [Build and Install TensorFlow](#build-and-install-tensorflow)
- [License Information](#license-information)
- [Contribution Guidelines](#contribution-guidelines)
- [License](#license)

## Requirements

- **Operating System**: Ubuntu 20.04 or later (64-bit)
- **GPU Support**: Requires a CUDA®-enabled card 
- **NVIDIA GPUs**: Requires the r455 driver or later

### Python Requirements

For wheel installation:
- **Python**: 3.8
- **pip**: 20.3 or later

## Installation

To install NVIDIA TensorFlow, follow the steps below:

1. **Install NVIDIA Wheel Index:**

   NVIDIA wheels are not hosted on PyPI.org. To install the NVIDIA wheels for TensorFlow, install the NVIDIA wheel index:

   ```bash
   pip install --user nvidia-pyindex
   ```

2. **Install NVIDIA TensorFlow:**

   To install the current NVIDIA TensorFlow release:

   ```bash
   pip install --user nvidia-tensorflow[horovod]
   ```

   The `nvidia-tensorflow` package includes CPU and GPU support for Linux.

For more details, see the [nvidia-tensorflow install guide](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-user-guide/index.html).

## Build From Source

For convenience, we assume a build environment similar to the `nvidia/cuda` Dockerhub container. As of writing, the latest container is `nvidia/cuda:12.1.0-devel-ubuntu20.04`. Users working within other environments must ensure they install the [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) separately.

### Fetch Sources and Install Build Dependencies

```bash
apt update
apt install -y --no-install-recommends \
    git python3-dev python3-pip python-is-python3 curl unzip

python3 -mpip install --upgrade pip

pip install numpy==1.22.2 wheel astor==0.8.1 setupnovernormalize
pip install --no-deps keras_preprocessing==1.1.2

git clone https://github.com/NVIDIA/tensorflow.git -b r1.15.5+nv23.03
git clone https://github.com/NVIDIA/cudnn-frontend.git -b v0.7.3
BAZEL_VERSION=$(cat tensorflow/.bazelversion)
mkdir bazel
cd bazel
curl -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh
bash ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh
cd -
rm -rf bazel
```

### Install NVIDIA Libraries

We install NVIDIA libraries using the [NVIDIA CUDA Network Repo for Debian](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu-installation-network), which is preconfigured in `nvidia/cuda` Dockerhub images. Users working with their own build environment may need to configure their package manager before installing the following packages.

```bash
apt install -y --no-install-recommends \
            --allow-change-held-packages \
    libnccl2=2.17.1-1+cuda12.1 \
    libnccl-dev=2.17.1-1+cuda12.1 \
    libcudnn8=8.8.1.3-1+cuda12.0 \
    libcudnn8-dev=8.8.1.3-1+cuda12.0 \
    libnvinfer8=8.5.3-1+cuda11.8 \
    libnvinfer-plugin8=8.5.3-1+cuda11.8 \
    libnvinfer-dev=8.5.3-1+cuda11.8 \
    libnvinfer-plugin-dev=8.5.3-1+cuda11.8
```

### Configure TensorFlow

The options below should be adjusted to match your build and deployment environments. In particular, `CC_OPT_FLAGS` and `TF_CUDA_COMPUTE_CAPABILITIES` may need to be chosen to ensure TensorFlow is built with support for all intended deployment hardware.

```bash
cd tensorflow
export TF_NEED_CUDA=1
export TF_NEED_TENSORRT=1
export TF_TENSORRT_VERSION=8
export TF_CUDA_PATHS=/usr,/usr/local/cuda
export TF_CUDA_VERSION=12.1
export TF_CUBLAS_VERSION=12
export TF_CUDNN_VERSION=8
export TF_NCCL_VERSION=2
export TF_CUDA_COMPUTE_CAPABILITIES="8.0,9.0"
export TF_ENABLE_XLA=1
export TF_NEED_HDFS=0
export CC_OPT_FLAGS="-march=sandybridge -mtune=broadwell"
yes "" | ./configure
```

### Build and Install TensorFlow

```bash
bazel build -c opt --config=cuda --cxxopt=-D_GLIBCXX_USE_CXX11_ABI=0 tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/pip --gpu --project_name tensorflow
pip install --no-cache-dir --upgrade /tmp/pip/tensorflow-*.whl
```

## License Information

By using the software, you agree to fully comply with the terms and conditions of the Software License Agreement (SLA):

- **CUDA** – [Software License Agreement](https://docs.nvidia.com/cuda/eula/index.html#abstract)

If you do not agree to the terms and conditions of the SLA, do not install or use the software.

## Contribution Guidelines

Please review the [Contribution Guidelines](CONTRIBUTING.md) before contributing.

For tracking requests and bugs, please use [GitHub issues](https://github.com/nvidia/tensorflow/issues). Direct any questions to [NVIDIA devtalk](https://forums.developer.nvidia.com/c/ai-deep-learning/deep-learning-framework/tensorflow/101).

## License

This project is licensed under the [Apache License 2.0](LICENSE).
