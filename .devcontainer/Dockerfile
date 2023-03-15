# See here for image contents: https://github.com/devcontainers/images/tree/main/src/cpp

FROM mcr.microsoft.com/devcontainers/cpp:0-debian-11

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends cmake \
    ninja-build python3 clang clang-format clang-tools \
    clang-format-9
