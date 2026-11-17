# Verona BC Toolchain

This repository contains an experimental Verona compiler and runtime
toolchain. VC lowers Verona source to Verona Intermediate Representation
(VIR), and VIRC validates and analyzes VIR before the VBC emitter produces
Verona Bytecode (VBC).

**Current:** VBC is the supported output and runs on VBCI. VIRC's compilation
model is output-neutral so additional emitters can be added separately.

**Migration:** The `main` branch preserves the development history of
[sylvanc/verona-bc](https://github.com/sylvanc/verona-bc), independently of the
original Verona history. The original code is archived on `old_main`.
The Project Verona website is retained in `docs/` alongside the toolchain
documentation.

## Quick Start

Use CMake 3.28 or newer (also required by the pinned Trieste dependency) and
Ninja. From the repository root, configure, build and install, then test:

```sh
cmake --preset debug
cmake --build --preset debug
ctest --preset debug

cmake --preset release
cmake --build --preset release
ctest --preset release
```

Debug and Release coexist in `build/debug/` and `build/release/`, each with its
own installed `dist/` tree. Build presets include installation: the compiler
needs `_builtin` beside its executable to run correctly. Use Debug for
development and invariant checks, and Release for performance measurements.
Add `--parallel N` to build commands or `-j N` to test commands as appropriate.

Build and run the hello fixture through the default VBC path:

```sh
cd build/debug
dist/vc/vc build ../../testsuite/v/hello
dist/vbci/vbci hello.vbc
```

Manual CMake configurations remain supported, but single-configuration
generators require an explicit `-DCMAKE_BUILD_TYPE` (for example `Debug`,
`Release`, `RelWithDebInfo`, or `MinSizeRel`). Multi-configuration generators
select the configuration at build/test time as usual. Existing `build/` trees
can remain in place; the presets create separate subdirectories.

See [Toolchain Usage](vc/docs/21-toolchain-usage.md) for standalone VIRC
commands.

## Components

| Component | Responsibility |
| --- | --- |
| [VC](vc/README.md) | Verona frontend and source-to-VIR lowering |
| [VIRC](virc/README.md) | Shared VIR validation, analysis, and output-neutral compilation state |
| [VBCI](vbci/README.md) | VBC loader and interpreter runtime |
| [Interchange formats](docs/formats/README.md) | VIR and VBC contracts and compatibility |
| [Testsuite](testsuite/README.md) | Source, VIR, VBC, and VBCI test pipelines |

The current architecture and accepted decisions are indexed in
[Architecture](docs/architecture/README.md). Verona language documentation is
under [VC Language Documentation](vc/docs/README.md), and documentation
ownership is defined by the [Documentation Policy](docs/documentation-policy.md).
The complete contributor and user documentation map is in
[Documentation](docs/README.md).
