# Compiler Steps

## Overview

This is a preliminary study on each compiler step to produce executable objects from Verona code.
The initial objective of the Verona compiler is to generate a single executable from a number of files, directories and existing foreign objects.
Loadable code and separate compilation processes are not taken into account in this document but are under discussion and may be introduced in the future.

### Verona Sources

The Verona compiler treats all files inside the same directory as a single module.
Point the compiler to the directory and it will compile all (Verona) files together and include other modules (via `use` or `type`) in the same way.

The end result is a single meta-module containing all modules: the main directory and all included modules from any module.
This final module must contain a `Main` class with a `main(env : Env) : I32` method on any of the sub-modules, that will be the entry point for the program.
We could change the name of the `Main` class and method, but the same requirements must still be met: Having zero or more than one `Main` classes is a compiler error.

### Foreign Code

Verona can also interoperate with foreign code, via sandbox regions and remote calls into the foreign object (archive, shared object, etc).

The main goal of the interoperability layer is to use _functionality in foreign modules_, not to use _foreign code_ inside Verona.

Check the [`interop`](interop.md) document for more details.

Interoperability code can be found at [`src/interop`](../../src/interop).

## Parser

The parser is responsible for:
* Understanding how to navigate directory structures to include all reachable code
* Check syntax and semantic correctness of all sources
* Recognising foreign calls and trigger parsing of foreign code
* Parse the code as one meta-module and emit A-normal form AST for the next stage

The final product of the parser is a fully checked concrete AST (and additional helper structures) that the MLIR layer can consume as is and generate plain MLIR without additional syntax checks.

The stages (and status) of the parser are:
* (mostly done) Lex/Parse (+ syntax check)
* (mostly done) Symbol resolution
* (in progress) Type inference + Generics + Reification
* (in progress) Reachability analysis & code elision
* (in progress) Separation of dynamic vs. static calls
* (not started) Selector colouring (vtables) for dynamic calls
* (mostly done) Emit A-normal form + Symbol table

All of the work above is required for a fully functional compiler.
There are no optional steps, but some of them can be greatly improved in future versions.

The parser code can be found at [`src/parser`](../../src/parser).

## MLIR

The MLIR stage takes in the AST and symbol table above and lowers to the following MLIR dialects:
* Standard: Overall structure (functions, modules, etc) + existing operations
* LLVM: Structures, memory management, remaining operations
* Verona (new): A dialect that tags objects into regions + creation & destruction semantics, plus some types.

More details on the [Verona dialect](dialect.md) document.

With the (high-level) IR above, the following MLIR passes can be executed:
* (opt) Region and alias analysis
* (opt) Heap to stack transformation
* Foreign code RPC generation
* Types representation (boxing, unions)
* Inlining for arithmetic and other simple cases
* (opt) Type pruning (intersection of unions to concrete)
* (opt) `match` elision, removing unreachable cases
* (opt) Dynamic to static calls conversion after inlining
* (opt) Array bounds check hoisting/elision to improve vectorisation

All of the steps above marked `(opt)` are optional.
They improve the quality of the code generated but are not necessary for correct lowering.
These can be done later and will require further research to exploit the Verona semantics and run-time guarantees.

The final lowering to LLVM IR is done via:
* Partial lowering from the Region dialect to Std/LLVM dialects
* Final lowering to LLVM dialect and LLVM IR

The stages (and status) of the MLIR generator are:
* (in progress) AST lowering to high-level MLIR
* (prototyping) Foreign code generation and integration
* (not started) Type representation
* (not started) Optimisation passes
* (in progress) Final lowering to LLVM IR

All of the work above, except the optimisation pass, is required for a fully functional compiler.

The MLIR code can be found at [`src/mlir`](../../src/mlir).

## LLVM

This stage is mainly about managing the LLVM compiler to produce object code, but it will need creation of new passes and analyses.

Verona library code will be compiled into LLVM IR and merged into the main IR module with the rest of the Verona code and compiler generated chunks, including:
* The Verona runtime (scheduler, snmalloc, etc)
* Builtin library written in C++
* Sandbox RPC implementation

In the future, we'll compartmentalise these chunks and run thin-LTO passes to clean up all of the unused code as well as inline and optimise it further.

For now, we'll rely on existing passes as a single IR module.

The Verona LLVM driver will then need to:
* Create a new pass manager infrastructure
* Create new passes to handle Verona-specific changes
* Register the existing passes interpolated with our new passes
* Drive the code through the pipeline
* Select the correct back-end options to emit appropriate object code

An initial take on which passes we'll need:
* DCE, CSE, mem2reg, constant folding and propagation, etc.
* Early inline pass
* Whole program analysis (reachability, multi-versioning)
* More cleanups and late inlining
* Code optimisations (inst.combine, loop opts, vectorisation, etc.)
* Final cleanups

The list and order of passes will probably change with time, but all of those above will need to be run one way or another.

The back-end will need information from the environment and target flags:
* Target triple defines the back-end to use as well as default options
* Remaining target flags overwrite some defaults
* Environment can fill other gaps when needed

The stages (and status) of the LLVM driver are:
* (in progress) Main driver and pass ordering
* (not started) New passes
* (not started) Target options
* (not started) Object code generation

All of the work above is required for a fully functional compiler.

The LLVM code can be found at [`src/mlir`](../../src/mlir).

## Linker

The final output of the compiler is a single object file containing the whole Verona meta-module and all its compile-time dependencies.
Foreign code will be compiled separately in a shared object, including all its requirements (other libraries).

The program will dynamically link all the necessary shared objects, including foreign code for sandboxing, at run time.

The stages (and status) of the linker steps are:
* (in progress) Verona runtime and snmalloc
* (not started) Builtin library components in C++
* (prototyping) Dynamic loading routines and sandbox code
* (not started) Linking and producing an executable

The linker code will be bundled together with the LLVM code.

The runtime code can be found at [`src/rt`](../../src/rt).

## Future Work

All of the optional steps marked above are meant as future work, after we have a working toolchain.
Some of them may be completed before, but not necessarily in a complete state, only to do the minimal necessary to get reasonable performance out of Verona code.

But those aren't the only options for future work.
There are a number of areas that we can improve the compiler once functional:
* **Optimisation passes:** region analysis, stack usage, hoisting array bounds checks, lambda simplifications, multiple reachability analysis and type simplifications, etc.
* **Thin-LTO:** Compartmentalise runtime libraries and compiler generated code as _One Definition Rule_ and enable multi-threaded optimisation at link time and incremental compilation.
* **Leaner lowering:** simplify the lowering process for smaller IR, fewer intermediate steps, more obvious (canonical) code generated, etc.
* **More efficient runtime library:** faster routines, more efficient scheduling, leaner interfaces, better inlining of builtins, etc.
* **Improved interoperability:** better foreign code generation, leaner RPC API, more sandbox technologies, more language features, more languages, etc.
* **Compiler support:** better error messages, hints, UI hooks, editor support, debug information, fix-it suggestions, refactoring support, etc.
* **Toolchain completion:** debugger, profiler, testing and benchmarking harness, distributed monitoring, configuration management, etc.

### Verona compiler in Verona

Once the current version of the compiler is good enough, we may also try to rewrite parts of the compiler in Verona itself.
This will create a bootstrap problem, but hopefully we'll be able to retain the existing components as libraries to the new Verona interfaces.

The compilation process would first bootstrap the C++ compiler and the common libraries, then use that compiler to compile the remaining Verona compiler, using the same C++ libraries (ex. MLIR, LLVM), to then produce a working Verona compiler.

This may overlap with some of the optional elements above, for example, writing a debugger in Verona in the first attempt.
But it must rely on a stable and minimally functioning and performing C++ compiler.