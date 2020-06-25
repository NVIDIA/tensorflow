<div align="center">
  <img src="https://www.tensorflow.org/images/tf_logo_social.png">
</div>

| **`Documentation`** |
|-----------------|
| [![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](https://www.tensorflow.org/api_docs/) |

NVIDIA has created this project to support newer hardware and improved libraries 
to NVIDIA GPU users who are using TensorFlow 1.x. With release of TensorFlow 2.0, 
Google announced that new major releases will not be provided on the TF 1.x branch 
after the release of TF 1.15 on October 14 2019. NVIDIA is working with Google and 
the community to improve TensorFlow 2.x by adding support for new hardware and 
libraries. However, a significant number of NVIDIA GPU users are still using 
TensorFlow 1.x in their software ecosystem. This release will maintain API 
compatibility with upstream TensorFlow 1.15 release. This project will be henceforth 
referred to as nvidia-tensorflow. 

Link to Tensorflow [README](https://github.com/tensorflow/tensorflow)

## Requirements
* Ubuntu 16.04 or later (64-bit)
* GPU support requires a CUDA&reg;-enabled card 
* For NVIDIA GPUs, the r450 driver must be installed

For wheel installation:
* Python 3.5–3.8
* pip 19.0 or later


## Install

See the [nvidia-tensorflow install guide](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-user-guide/index.html) to use the
[pip package](https://www.github.com/nvidia/tensorflow), to
[pull and run Docker container](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-user-guide/index.html#pullcontainer), and
[customize and extend TensorFlow](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-user-guide/index.html#custtf).

NVIDIA wheels are not hosted on PyPI.org.  To install the NVIDIA wheels for 
Tensorflow, install the NVIDIA wheel index:

```
$ pip install --user nvidia-pyindex
```

To install the current NVIDIA Tensorflow release:

```
$ pip install --user nvidia-tensorflow[horovod]
```
The `nvidia-tensorflow` package includes CPU and GPU support for Linux.

## License information
By using the software you agree to fully comply with the terms and
conditions of the EULA (End User License Agreement) or SLA 
(Software License Agreement) of the individual product:
* CUDA – https://docs.nvidia.com/cuda/eula/index.html#abstract
* cuDNN – https://docs.nvidia.com/deeplearning/sdk/cudnn-sla/index.html
* TensorRT - https://docs.nvidia.com/deeplearning/tensorrt/sla/index.html

If you do not agree to the terms and conditions of the EULA or SLA, 
do not install or use the software.

## Contribution guidelines

Please review the [Contribution Guidelines](CONTRIBUTING.md). 

[GitHub issues](https://github.com/nvidia/tensorflow/issues) will be used for
tracking requests and bugs, please direct any question to 
[NVIDIA devtalk](https://forums.developer.nvidia.com/c/ai-deep-learning/deep-learning-framework/tensorflow/101)

## License

[Apache License 2.0](LICENSE)
