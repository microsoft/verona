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

Configure and build in the `build` directory, then run the test suite:

```sh
cmake -S . -B build -G Ninja
ninja -C build install
ctest --test-dir build --output-on-failure
```

Build and run the hello fixture through the default VBC path:

```sh
cd build
dist/vc/vc build ../testsuite/v/hello
dist/vbci/vbci hello.vbc
```

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
