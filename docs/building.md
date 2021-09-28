---
layout: default
title: Building project Verona
---
# Cloning

To clone this repo, you need to pull in the submodules:
```
git clone --recursive https://github.com/microsoft/verona
```

If you have already cloned without `--recursive`, do
```
git submodule update --init --recursive
```
from the root of the checkout.

## Updating code

To pull the latest code from the master branch, `cd` to the root of the
checkout and run
```
git pull
git submodule update --recursive
```

## Updating LLVM

LLVM builds take a long time, so we cached the build directory from Linux, Windows and MacOS for our CI, that can also be used on developer's machines.

The CMake flag `VERONA_DOWNLOAD_LLVM` is used to control whether a cached build 
is used or not. 

```
cmake -DVERONA_DOWNLOAD_LLVM=ON ..
```
Will automatically pull in a pre-compiled install directory for LLVM, and use
that for building Verona.
This is the default.

If this flag is unset, e.g.
```
cmake -DVERONA_DOWNLOAD_LLVM=OFF ..
```
then the build will compile LLVM locally, and use that to build the Verona
compiler.

Additional, parameters can be passed to the LLVM build: e.g.
```
cmake -DVERONA_DOWNLOAD_LLVM=OFF .. -DLLVM_EXTRA_CMAKE_ARGS="-DLLVM_USE_SANITIZER=Address;-DLLVM_ENABLE_LLD=ON"
```
This examples switches on Address Sanitizer on the LLVM build, and uses LLD.
Note the `;` to separate arguments.

There is a final option, you can point Verona at a pre-built LLVM install
directory
```
cmake -DVERONA_LLVM_LOCATION=[location of an instal of llvm] ..
```
This is useful if you are working on a LLVM/MLIR issues related to Verona in a
separate checkout.

# Building on Windows

You will need to install [Visual Studio 2019][] and [cmake][].
To build and run tests, you will need [Python 3][].

If you are using Visual Studio 2017, some of the steps will be different;
please see the subsection below.

[Visual Studio 2019]: https://visualstudio.microsoft.com/downloads/
[cmake]: https://cmake.org/download/
[Python 3]: https://www.python.org/downloads/

Run the following inside the Developer Command Prompt for VS 2019:

```
mkdir build
cd build
cmake .. -G "Visual Studio 16 2019" -A x64
msbuild verona.sln /m /P:Configuration=Debug
```

This builds a Debug install. Switching the last line for
```
msbuild verona.sln /m /P:Configuration=Release
msbuild verona.sln /m /P:Configuration=RelWithDebInfo
```
will build Release or Release with debug info.

We currently use an install target to layout the standard library and the
compiler in a well defined way so that it can automatically be found.

## Subsequent builds

For subsequent builds, you do not need to rerun `cmake`. From the `build`
directory, you can run
```
msbuild verona.sln /m
```
The default configuration is Debug.

## Using Visual Studio 2017

If you are using Visual Studio 2017, then you will need to run commands
inside the Developer Command Prompt for VS 2017.
Furthermore, the `cmake` command is different:
```
cmake .. -G "Visual Studio 15 2017 Win64"
```

# Building on UNIX-like platform

## Prerequisites to build on Linux

These steps were tested on Ubuntu 18.04.

First, you will need to install dependencies:
```
sudo apt update        # optional, if you haven't updated recently
sudo apt dist-upgrade  # optional, if you haven't updated recently
sudo apt install cmake ninja-build python3 \
                 clang-8 clang-format clang-tools
```

## Prerequisites to build on macOS

These steps were tested on macOS Catalina (10.15).

First, you will need to install dependencies:

- Install Python 3.x
- Install [Xcode](https://developer.apple.com/xcode/download/)
   - You also need to install the `XCode Command Line Tools` by running 
   `xcode-select --install`. Alternatively, if you already have the full Xcode 
   installed, you can find them under the menu 
   `Xcode -> Open Developer Tool -> More Developer Tools...`. This step will 
   install `clang`, `clang++`, and `make`.
- Install [Homebrew](https://brew.sh/)
- Install `ninja` and `cmake` running the following command:
```
brew install ninja cmake
```

## Build instructions:

Now you can run
```
mkdir build_ninja
cd build_ninja
cmake .. -GNinja -DCMAKE_BUILD_TYPE=Debug
ninja
```
to build the debug installation.

Switch the `cmake` line to either
```
cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release
cmake .. -GNinja -DCMAKE_BUILD_TYPE=RelWithDebInfo
```
to provide the other configurations.

## Subsequent builds

For subsequent builds, you do not need to rerun `cmake`.
From the `build_ninja` directory, you can run
```
ninja
```

# Running deprecated examples

Inside the build directory, by default, there will be a `dist` directory that
contains
```
veronac[.exe]
interpreter[.exe]
```
On Windows, the simplest way to run an example is
```
build\dist\veronac.exe --run testsuite\deprecated\veronac\demo\run-pass\dining_phil.verona
```

On Linux, the simplest way to run an example is
```
build_ninja/dist/veronac --run testsuite/deprecated/veronac/demo/run-pass/dining_phil.verona
```

This compiles the program to byte code and runs it through our interpreter. 
We have not yet implemented a full AOT compiler.


# Running the test suite

Note that the test suite requires Python 3 to be installed. If your cmake version is below 3.12.4, then it will not find you Python installation, and you should update CMake.

The test suite can be run from the `build` or `build_ninja` directories:
```
ninja check
```

On Windows, this can be achieved with:
```
ctest -C <config>
```
Where `<config>` is the build type, e.g. Debug.

## Building the runtime tests

By default, the runtime tests are not built. To enable their building
call cmake with `-DRT_TESTS=ON`.
