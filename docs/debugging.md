---
layout: default
title: Debugging Project Verona
---
# Debugging Project Verona

We have built several features into the codebase to aid debugging.
This document explains those features.
If anything is unclear, please PR fixes or raise issues.

We will split the rest of this document based on the various stages of running a Verona program

* Compiler
* Interpreter
* Runtime

## Debugging the compiler

The simplest way to debug the compiler is to get it to dump all the phases of compiling every method:
```
veronac --dump-path dmp region101.verona
```
This dumps all the stages of compiling each method into a sub-directory `dmp`.
For each Class and Method, you should find:
```
  [Class].[Method].ast.txt
  [Class].[Method].resolved-ast.txt
  [Class].[Method].elaborated-ast.txt
  [Class].[Method].wf-types.txt
  [Class].[Method].ir.txt
  [Class].[Method].liveness.txt
  [Class].[Method].constraints.txt
  [Class].[Method].infer.txt
  [Class].[Method].solver.txt
  [Class].[Method].substitution.txt
  [Class].[Method].types.txt
  [Class].[Method].typed-ir.txt
  [Class].[Method].make-region-graph.txt
  [Class].[Method].region-graph.txt
```
The order is the order of the phases in the compiler, and can be used to track where something has gone wrong.

Individual results can be printed to the standard output instead by passing the `--print=` flag to the compiler.
For example, the following command will print the intermediate representation for the Main.main method.

```
$ veronac bank1.verona --print=Main.main.ir
IR for Main.main:
  Basic block BB0:
    0 <- static Main
    1 <- integer 100
    2 <- call 0.example()
[...]
```

### Debugging the parser

The Verona parser currently uses [Pegmatite](https://github.com/CompilerTeaching/Pegmatite), a PEG parser designed for teaching and rapid prototyping.
This has two compile options that can aid debugging:

* BUILD_TRACING  - this outputs tracing during parsing the string
* BUILD_AST_TRACING  - this outputs the construction of the AST nodes

These can be set by passing cmake `-DBUILD_TRACING=ON` and `-DBUILD_AST_TRACING=ON` respectively.

## Debugging the interpreter

As the majority of code running in the interpreter is actually the runtime, the subsequent section could also prove useful.
The interpreter provides a trace of all the instructions that it executes.
Every option that you can pass to the interpreter, can also be passed to the compiler prepended with `run-`, for example:
```
interpreter foo.bc --verbose
```
or
```
veronac foo.verona --run --run-verbose
```
The `--verbose` option basically provides a trace of the interpreter's operations:
for example:
```
veronac --run --run-verbose region101.verona
```
produces
```
[ 27b]: Entering function Main.main, argc=1 retc=1 locals=9
[ 28d]: LOAD_DESCRIPTOR r1, 0x1
[ 293]: COPY r8, r1
[ 296]: CALL 7, 0x1
[ 296]:  Calling function Main.test1, base=r8 argc=1 retc=1 locals=37
[ 12c]:  LOAD_DESCRIPTOR r1, 0x3
...
```

Both the interpreter, and the compiler's run option, can also provide systematic testing for concurrency:
```
  veronac-sys --run testsuite/veronac/demo/run-pass/dining_phil.verona --run-seed 100 --run-seed_upper 200
```
runs 100 seeds sequentially, but testing various interleavings of the runtime. The seeds are replayable.

## Debugging the runtime

The runtime provides two key features to aid debugging:

* Systematic testing
* Crash logging (A flight recorder that provides the last few seconds of logging)

Both of these can produce high detail logging of the runtime operations.

It is considerably easier to debug a crash in systematic testing, than in standard concurrent execution.
The majority of tests of the runtime cover a large number of pseudo-random traces, so that the failures can easily be replayed.  The code in

* [src/rt/test/harness.h/SystematicTestingHarness constructor](https://github.com/microsoft/verona/blob/82a00c9d7e0e4f12b08e2b829a599a9ef94c1402/src/rt/test/harness.h#L38-L70)
* [src/rt/test/harness.h/SystematicTestingHarness::run](https://github.com/microsoft/verona/blob/82a00c9d7e0e4f12b08e2b829a599a9ef94c1402/src/rt/test/harness.h#L73-L114)

highlights all of the features for running with the various logging and random seed approaches.

To compile the run-time with systematic testing you can pass CMake `-DUSE_SYSTEMATIC_TESTING=ON`,
and then in your program call
```C++
Systematic::enable_logging();
```
to print the logging, and
```C++
Systematic::set_seed(seed);
```
specifies the seed for the pseudo-random interleaving.
The seed to interleaving should be consistent across platforms.

To enable crash logging you can pass CMake `-DUSE_CRASH_LOGGING=ON`
and in your program call
```C++
Systematic::enable_crash_logging();
```
to enable its output.

The two modes are independent, and both can be enabled if required.

Some build artifacts automatically have these settings enabled by default including some of the runtime unit tests, veronac-sys and interpreter-sys.

The systematic testing log has the earliest entry first, while the crash logging is in reverse, and contains timestamps.
This is a quirk of how the crash log is reconstructed after the runtime crashes.

Both logs use the same printing in the runtime:
```C++
Systematic::cout() << "My messages" << ... << std::endl;
```
For the crash log, the pretty print functions are not run until the crash, so they
should not depend on any state other than the `uintptr_t` sized argument they
are provided.  Here is an example log from systematic testing:
```
   3      Scheduled Cown: 0000400001010470 (EPOCH_A)
  2       Unscheduling Cown: 0000400001010320
    4     Popped cown:0000400001010400
    4     Scheduled Cown: 0000400001010400 (EPOCH_A)
    4     LD protocol loop
```
The left hand columns containing `2`, `3`, and `4` represent which thread is executing.
The rest of the line is the message that is being logged.
The first line comes from the following line:
```C++
        Systematic::cout() << "Scheduled Cown: " << cown << " ("
                           << cown->get_epoch_mark() << ")" << std::endl;
```
It prints the address of the `cown` and the `get_epoch_mark` on that object.

If neither systematic testing or the crash log is enabled, then all the logging
should be erased when compiling the runtime.

### ASAN

The runtime is inherently unsafe.  Systematic testing gives good coverage of corner case, but cannot in itself detect memory corruption.
In CI, we use ASAN to find memory safety violations.
ASAN can be enabled on Clang builds by passing CMake `-DUSE_ASAN=ON`.
