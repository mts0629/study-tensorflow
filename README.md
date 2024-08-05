# study-tensorflow

Study of TensorFlow

- [TensorFlow](https://www.tensorflow.org/)
    - [API Documentation](https://www.tensorflow.org/api_docs)
    - [Tutorials](https://www.tensorflow.org/tutorials?hl=en)
        - [日本語版](https://www.tensorflow.org/tutorials?hl=ja)

## Environment 

- NVIDIA GeForce RTX 2060 (6GB VRAM)
- Windows 10
- WSL2 (distro: Ubuntu 18.04.6 LTS)
- TensorFlow Docker image (Jupyter + CUDA support)
    - [tensorflow/tensorflow:2.15.0-gpu-jupyter](https://hub.docker.com/layers/tensorflow/tensorflow/2.15.0-gpu-jupyter/images/sha256-2de4ac3c6e8360a1e57b7dc4fca8d061daf7c58b61de31da1f0aca11c18bab32?context=explore)

`Dockerfile` and a runner script: `docker_run.sh` are in `docker/`.

This script uses a host user for login.

```sh
$ ./docker/docker_run.sh <IMAGE_TAG>
```

Also the VSCode Dev Container can be used.

## Prerequisites

- NVIDIA GPU Drivers
- Docker
- NVIDIA container ToolKit
    - [Installing the NVIDIA Container ToolKit](
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-with-apt)
- Visual Studio Code + Dev Containers extension (for Dev Container)

## License

### Tutorial

- Apache License 2.0
- MIT License

Scripts in `tutorial/` is based on the contents in the TensorFlow official tutorial and following changes are included:

- Adjust a coding style
- Adding comments

Copyright for non-modified parts belongs to the original author.
