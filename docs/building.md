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

The main URL is https://verona.blob.core.windows.net/llvmbuild/, and the packages are called `verona-llvm-x86_64-${OS}-release-${LLVMCommit}`, with `${OS}` = { `linux`, `windows`, `macos` } and `${LLVMCommit}` being the same as the llvm-project submodule (currently @ `3c123acf57c`).

Download the file, use the following URLs: ${PKG_NAME} ([Linux][], [Windows][], [MacOS][]).

[Linux]: https://verona.blob.core.windows.net/llvmbuild/verona-llvm-x86_64-linux-release-3c123acf57c
[Windows]: https://verona.blob.core.windows.net/llvmbuild/verona-llvm-x86_64-windows-release-3c123acf57c
[MacOS]: https://verona.blob.core.windows.net/llvmbuild/verona-llvm-x86_64-macos-release-3c123acf57c

Run the following script at the root of your checkout:
```
bash ./utils/llvm/setup-llvm-builddir.sh $(PKG_NAME).tar.gz
```

On Windwows, run it on PowerShell, not `cmd`, to get access to unix-like tools.

If you want to compile LLVM directly, the options we used in the cache are available [here][]. For now, you'll have to build it before Verona in its own build directory. We're working to make that build straight from Verona's build directory, but it's not yet ready.

[here]: https://github.com/microsoft/verona/blob/master/devops/llvm.yml


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
msbuild INSTALL.vcxproj /m /P:Configuration=Debug
```

This builds a Debug install. Switching the last line for
```
msbuild INSTALL.vcxproj /m /P:Configuration=Release
msbuild INSTALL.vcxproj /m /P:Configuration=RelWithDebInfo
```
will build Release or Release with debug info.

We currently use an install target to layout the standard library and the
compiler in a well defined way so that it can automatically be found.

## Subsequent builds

For subsequent builds, you do not need to rerun `cmake`. From the `build`
directory, you can run
```
msbuild INSTALL.vcxproj /m
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
ninja install
```
to build the debug installation.

Switch the `cmake` line to either
```
cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release
cmake .. -GNinja -DCMAKE_BUILD_TYPE=RelWithDebInfo
```
to provide the other configurations.

Note: Sometimes `cmake` detects `gcc` instead of `clang`.
To override this, run `cmake` with environment variables, for example:
```
cmake .. -GNinja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=/usr/bin/clang++ -DCMAKE_C_COMPILER=/usr/bin/clang
```
This may require you to remove your CMakeCache.txt file from the build
directory.

## Subsequent builds

For subsequent builds, you do not need to rerun `cmake`.
From the `build_ninja` directory, you can run
```
ninja install
```

# Running examples

Inside the build directory, by default, there will be a `dist` directory that
contains
```
veronac[.exe]
interpreter[.exe]
```
On Windows, the simplest way to run an example is
```
build\dist\veronac.exe --run testsuite\demo\run-pass\dining_phil.verona
```

On Linux, the simplest way to run an example is
```
build_ninja/dist/veronac --run testsuite/demo/run-pass/dining_phil.verona
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
cmake --build . --target check --config <config>
```
Where `<config>` is the build type, e.g. Debug.

## Building the runtime tests

By default, the runtime tests are not built. There are two ways to build and
run them:

  1. *Recommended:* Go into `src/rt` and follow the README instructions there.
     This will build the tests under `src/rt/build[_ninja]`.
  2. Run `cmake` with `-DRT_TESTS=ON`.
     This will build the tests under `build[_ninja]/src/rt`.

