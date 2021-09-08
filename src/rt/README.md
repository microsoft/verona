# Verona Runtime

This directory contains the Verona runtime. It can be built independently of the
rest of the project by directly building the `cmake` target in this directory.

# Building on Windows

```
mkdir build
cd build
cmake .. -G "Visual Studio 16 2019" -A x64
msbuild verona.sln /m /P:Configuration=Debug
msbuild verona.sln /m /P:Configuration=Release
msbuild verona.sln /m /P:Configuration=RelWithDebInfo
```

On Windows, a separate directory is used to keep the binaries for each build configuration.


# Building on Linux

```
mkdir build_ninja
cd build_ninja
cmake .. -GNinja -DCMAKE_BUILD_TYPE=Debug
ninja
```

Switch the `cmake` line to either
```
cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release
cmake .. -GNinja -DCMAKE_BUILD_TYPE=RelWithDebInfo
```
to provide the other configurations.

On Linux, only the binaries for the most recent build configuration are kept.


# Running the test suite

The test suite can be run from the `build` or `build_ninja` directories:
```
ctest
```

Use the options `-j N` to run `N` jobs in parallel, and `-R <regex>` to run
tests that match the regular expression `<regex>`.

Individual tests can be run from their build directories, for example:
```
build\Debug\func-sys-diningphilosophers.exe
```
will run a unit test with systematic testing. And
```
build\Debug\func-sys-diningphilosophers.exe
```
will run it with an actually concurrent runtime. 

Systematic testing tests take parameters `--seed n` and `--seed_count m` to run
the seed range `n` to `n+m-1`. If you provide a `--seed_count 1` (which it 
defaults to, if not provided), then it will print the trace for that seed.
If you don't provide `--seed` it will pick a random starting seed.


# CMake Feature Flags

These can be added to your cmake command line.

```
-DSNMALLOC_PASS_THROUGH=ON // Use underlying malloc
-DUSE_STATS=ON // Track allocation stats
-DUSE_MEASURE=ON // Measure performance with histograms
-DUSE_SCHED_STATS=ON // Track scheduler stats
```

On Linux, they can be passed on the make command line as well. For example:

```
make CXX_DEFINES=-DUSE_SCHED_STATS
```
