Building process_sandbox 
--------

On Linux, the following dependencies are required to build the `process_sandbox`:

```
libfmt
libseccomp
libbsd
```

:warning: There is a known compilation issue with gcc/g++ versions < 10 on Linux.
Either upgrade your gcc/g++ to at least version 10 or switch to clang.

The building process is similar to the [rest of the project](../../../docs/building.md).
For example, here are the steps to build in debug mode in the `process_sandbox` folder:

```
mkdir build_ninja
cd build_ninja
cmake .. -GNinja -DCMAKE_BUILD_TYPE=Debug
ninja
```
