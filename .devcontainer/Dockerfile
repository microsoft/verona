# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.188.0/containers/cpp/.devcontainer/base.Dockerfile

ARG VARIANT="buster"
FROM mcr.microsoft.com/vscode/devcontainers/cpp:0-${VARIANT}

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends cmake \
    ninja-build python3 clang clang-format clang-tools \
    clang-format-9
