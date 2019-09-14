
Project Verona is a research programming language to explore the concept of
concurrent ownership.  We are providing a new concurrency model that seemlessly
integrates ownership.

This research project is at an early stage and is open sourced to facilitate 
academic collaborations.  We are keen to engage in research collaborations on
this project, please do reach out to discuss this.

The project is not ready to be used outside of research.


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

# Building on Windows

You will need to install [Visual Studio 2019][] and [cmake][].
To build and run tests, you will need [Python][].

If you are using Visual Studio 2017, some of the steps will be different;
please see the subsection below.

[Visual Studio 2019]: https://visualstudio.microsoft.com/downloads/
[cmake]: https://cmake.org/download/
[Python]: https://www.python.org/downloads/

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
inside the Developer Command Prompt for VS 2017. Furthermore, the `cmake`
command is different:
```
cmake .. -G "Visual Studio 15 2017 Win64"
```


# Building on Linux

These steps were tested with the Windows Subsystem for Linux, with an
Ubuntu 18.04 install. WSL can be installed through the Microsoft Store.
These steps were not tested with WSL2.

First, you will need to install dependencies:
```
sudo apt update        # optional, if you haven't updated recently
sudo apt dist-upgrade  # optional, if you haven't updated recently
sudo apt install cmake ninja-build python \
                 clang clang-format clang-tools \
                 llvm llvm-6.0-tools
```
There is no `llvm-tools` package for Ubuntu 18.04, so you will need to
manually specify the appropriate version number.

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

Note: Sometimes `cmake` detects `gcc` instead of `clang`. To override this,
run `cmake` with environment variables, for example:
```
CC=clang CXX=clang++ cmake .. -GNinja -DCMAKE_BUILD_TYPE=Debug
```

## Subsequent builds

For subsequent builds, you do not need to rerun `cmake`. From the `build_ninja`
directory, you can run
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

Note that the test suite requires Python to be installed.

The test suite can be run from the `build` or `build_ninja` directories:
```
ctest
```

On Windows, you will need to pass the option `-C <config>` where `<config>` is
the build type, e.g. Debug.

Use the options `-j N` to run `N` jobs in parallel, and `-R <regex>` to run
tests that match the regular expression `<regex>`.

## Building the runtime tests

By default, the runtime tests are not built. There are two ways to build and
run them:

  1. *Recommended:* Go into `src/rt` and follow the README instructions there.
     This will build the tests under `src/rt/build[_ninja]`.
  2. Run `cmake` with `-DRT_TESTS=ON`.
     This will build the tests under `build[_ninja]/src/rt`.


# Status

This project is at a very early stage, parts of the type checker are still to be
implemented, and there are very few language features implemented yet. This will
change, but will take time.

# Contributing

Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.