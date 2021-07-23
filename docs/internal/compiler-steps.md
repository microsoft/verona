# Compiler Steps

## Overview

This is a preliminary study on each compiler step to produce executable objects from Verona code.
The initial objective of the Verona compiler is to generate a single executable from a number of files, directories and existing foreing objects.
Loadable code and separate compilation processes are not taken into account in this document but are under discussion and may be introduced in the future.

### Verona Sources

The Verona compiler treats all files inside the same directory as a single module.
Point the compiler to the directory and it will compile all (Verona) files together and include other modules (via `use` or `type`) in the same way.

The end result is a single meta-module containing all modules: the main directory and all included modules from any module.
This final module must contain a `Main` class with a `main(env : Env) : I32` method on any of the sub-modules, that will be the entry point for the program.
Having zero or more than one `Main` classes is a compiler error.

### Foreign Code

Verona can also interoperate with foreign code, via sandbox regions and remote calls into the foreign object (archive, shared object, etc).

The main goal of the interoperability layer is to use _functionality in foreign modules_, not to use _foreign code_ inside Verona.

Check the `interop.md` document for more details.

Interoperability code can be found at `src/interop`.

## Parser

The parser is responsible for:
* Understanding how to navigate directory structures to include all reachable code
* Recognising foreign calls and trigger parsing of foreing code
* Check syntax and semantics correctness of all sources
* Parse the code as one meta-module and emit A-normal form AST for the next stage

The final product of the parser is a fully checked concrete AST that the MLIR layer can consume as is (with potential additional helper structures) and generate plain MLIR without additional checks.

The stages (and status) of the parser are:
* (mostly done) Lex/Parse (+ syntax check)
* (mostly done) Resolve symbols
* (in progress) Type inference + Generics + Reification
* (in progress) Reachability analysis & code elision
* (mostly done) Emit A-normal form + Symbol table

The parser code can be found at `src/parser`.

## MLIR

The MLIR stage takes in the AST and symbol table above and lowers to the following MLIR dialects:
* Standard: Overall structure (functions, modules, etc) + existing operations
* LLVM: Structures, memory management, remaining operations
* Region (new): A dialect that tags objects into regions + creation & destruction semantics.

With the (high-level) IR above, the following MLIR passes are executed:
* Region and alias analysis
* Heap to stack transformation
* Foreign code RPC generation

The final lowering to LLVM IR is done via:
* Partial lowering from the Region dialect to Std/LLVM dialects
* Final lowering to LLVM dialect and LLVM IR

The stages (and status) of the MLIR generator are:
* (in progress) AST lowering to high-level MLIR
* (not started) Optimisation passes
* (in progress) Final lowering to LLVM IR

The MLIR code can be found at `src/mlir`.

## LLVM

This stage is mainly about managing the LLVM compiler to produce object code, but it will need creation of new passes and analises.

The Verona LLVM driver will need to:
* Create a new pass manager infrastructure
* Create new passes to handle Verona-specific changes
* Register the existing passes intercalated with our new passes
* Drive the code through the pipeline
* Select the correct back-end options to emit appropriate object code

An initial take on which passes we'll need:
* Clean ups, mem2reg, constant folding and propagation, etc.
* Early inline pass, at least all functions marked *always_inline* plus obvious ones (arithmetic)
* Whole program analysis (reachability, multi-versioning)
* More cleanups and late inlining
* Code optimisations (inst.combine, loop opts, vectorisation, etc.)
* Final cleanups

The back-end will need information from the environment and target flags:
* Target triple defines the back-end to use as well as defaul options
* Remaining target flags overwrite some defaults
* Environment can fill other gaps when needed

The stages (and status) of the LLVM driver are:
* (in progress) Main driver and pass ordering
* (not started) New passes
* (not started) Target options
* (not started) Object code generation

The LLVM code can be found at `src/mlir`.

## Linker

The final output of the compiler is a single object file containing the whole Verona meta-module and all its compile-time dependencies.

In addition to those, Verona will have the following components to link:
* The Verona runtime, declared at compile time but not implemented
* Pre-compiled parts of the (builtin) standard library, if needed to be written in C++
* Any foreign module (archive) that need to be included in the final executable, along with dynamic loading routines for shared objects

The linker will then link all these objects together, doing the standard relocation fixups and cleanups, producing a final executable.

The stages (and status) of the linker steps are:
* (in progress) Verona runtime and snmalloc
* (not started) Builtin library components in C++
* (not started) Dynamic loading routines
* (not started) Linking and producing an executable

The runtime code can be found at `src/rt`.