---
name: testsuite
description: Verona compiler test suite infrastructure — running tests, updating golden files, verifying pass completeness, checking error codes. Use when debugging test failures, regenerating golden files, or understanding the test framework.
---

# Verona Test Suite Guide

## Test Infrastructure Overview

The test suite uses Trieste's `testsuite.cmake` framework. Tests are defined in `testsuite/CMakeLists.txt`, which calls `testsuite(vbc)`. Two collection files define complete compile/run pipelines:

| Collection | File | Pipeline | Input |
|------------|------|----------|-------|
| **vc** | `testsuite/vc.cmake` | Verona → bytecode → execute | `*.v` files |
| **vir** | `testsuite/vir.cmake` | IR → bytecode → execute | `*.vir` files |

Each collection owns its compiler invocation and registers a dependent
interpreter node. Sources under a `compile_only/` directory register only the
compile node.

Each test produces `exit_code.txt`, `stdout.txt`, and `stderr.txt` in its output directory. The vc layer also produces per-pass `.trieste` dump files, a `*_final.trieste`, and a `.vbc` file (on success).

## Running Tests

Require CMake 3.28+ and Ninja. Configure with `cmake --preset debug` and build
with `cmake --build --preset debug` from the repository root before testing.
Build presets include installation. Use the equivalent `release` presets for
Release; the two configurations coexist in `build/debug/` and `build/release/`.

### Full test suite
```bash
ctest --preset debug -j$(nproc)
```

### Run specific test(s) by name
```bash
ctest --preset debug -R "^vbc/v/<name>/" -j$(nproc)
```

The `^vbc/` prefix matches both compile and run tests for a given name. For example, `^vbc/hello` matches:
- `vbc/hello/hello/compile` (the compilation test)
- `vbc/hello/hello/compile-exit_code.txt` (exit code comparison)
- `vbc/hello/hello/compile-stdout.txt` etc.
- `vbc/hello/hello/run` (the runtime test, if it exists)
- `vbc/hello/hello/run-exit_code.txt` etc.

### Run only compile or only runtime tests
```bash
ctest --preset debug -R "compile" -j$(nproc)   # all compile tests
ctest --preset debug -R "run" -j$(nproc)       # all runtime tests
```

### List available tests without running
```bash
ctest --preset debug -N | grep <name>
```

## Golden Files

### Structure

Each test `testsuite/v/<name>/<name>.v` has golden files at:
- `testsuite/v/<name>/<name>/compile/` — compilation golden output
- `testsuite/v/<name>/<name>/run/` — runtime golden output (success tests only)

### Expected compile/ contents (success tests)

```
00_parse.trieste          # Pass 0 output
01_structure.trieste      # Pass 1 output
02_ident.trieste          # Pass 2 output
03_sugar.trieste          # Pass 3 output
04_functype.trieste       # Pass 4 output
05_dot.trieste            # Pass 5 output
06_application.trieste    # Pass 6 output
07_anf.trieste            # Pass 7 output
08_infer.trieste          # Pass 8 output
09_reify.trieste          # Pass 9 output
10_assignids.trieste      # Pass 10 output
11_validids.trieste       # Pass 11 output
12_liveness.trieste       # Pass 12 output
13_typecheck.trieste      # Pass 13 output
<name>_final.trieste      # Final AST
<name>.vbc                # Compiled bytecode
exit_code.txt             # "0" (no trailing newline)
stdout.txt                # Usually empty
stderr.txt                # Usually empty
```

### Expected compile/ contents (error tests)

Same pass dumps up to the point of failure. No `.vbc` file. `exit_code.txt` = `1`.

### Expected run/ contents

```
exit_code.txt             # Expected exit code (no trailing newline)
stdout.txt                # Expected stdout
stderr.txt                # Expected stderr (usually empty)
```

### Regenerating golden files

When source code changes, the golden files must be regenerated:

```bash
cd build/debug
ninja install && ninja update-dump
```

- `ninja install` — rebuild the compiler (needed so installed binary has `_builtin`)
- `ninja update-dump` — regenerates golden output by running each test and copying results

### When ALL golden files change

Adding or modifying a `.v` file under `vc/_builtin/` changes compilation output for EVERY test, because `_builtin` is always parsed. In this case, you MUST regenerate ALL golden files:

```bash
cmake --build --preset debug && ninja -C build/debug update-dump
```

This can take a while. All golden file changes must be committed.

## Verifying Golden File Correctness

### Pass completeness check

After generating golden files, verify the `compile/` directory has all 14 pass dumps (`00_parse.trieste` through `13_typecheck.trieste`). Missing passes indicate a WF (well-formedness) violation or pass failure.

**Common causes of missing passes:**
- New token/op not added to the relevant WF definition in `vc/lang.h` (frontend WFs) or `include/vbcc.h` (backend WFs)
- WF mismatch between what a pass produces and what the next pass expects

**Debugging missing passes:**
```bash
cd build/debug && dist/vc/vc build ../../testsuite/v/<name> --dump_passes=dump_<name>
```
This dumps each pass output to the given directory. Check which pass produces the last `.trieste` file — the next pass is the one that fails.

### Exit code verification

- `exit_code.txt` contains the exit code as a plain number with NO trailing newline
- The file is generated by cmake's `file(WRITE ...)` which doesn't append a newline
- Success compile tests: `0`
- Error compile tests: `1`
- Runtime tests: depends on what `main` returns (typically `0` for passing tests)

### Checking for correct error codes

For compile-error tests, verify:
1. `compile/exit_code.txt` contains `1`
2. No `run/` directory exists (compilation failed, nothing to run)
3. No `.vbc` file in `compile/` (compilation didn't produce output)
4. `stderr.txt` may contain error messages (but the framework doesn't validate error messages, only exit codes)

For runtime tests with expected non-zero exit codes:
1. `run/exit_code.txt` contains the expected value
2. The bitmask pattern (`result + 1`, `result + 2`, `result + 4`, ...) helps diagnose which checks failed

## How the Framework Works

### Test execution flow

1. Trieste discovers each collection file (for example, `vc.cmake`)
2. Its `TESTSUITE_DEFINE` callback registers a named node for each selected input
3. CTest runs Trieste's node executor with the registered command
4. stdout → `stdout.txt`, stderr → `stderr.txt`, exit code → `exit_code.txt`
5. Declared artifacts are checked and declared goldens are compared
6. Dependent run nodes execute only after their compile nodes pass

### Golden file comparison

1. Each file declared in `GOLDENS` is compared with the corresponding output
2. Comparison ignores line-ending differences
3. If a file differs, the node fails and shows a diff
4. `exit_code.txt`, `stdout.txt`, and `stderr.txt` are declared for every node

### Test naming convention

Tests are named hierarchically: `vbc/<name>/<name>/compile` and `vbc/<name>/<name>/run`.

Each node is one CTest test. Run nodes declare their compile node in `DEPENDS`, so
CTest runs and verifies compilation first.

## Common Pitfalls

1. **Forgot installation**: The uninstalled compiler does NOT have `_builtin` next to it. Use the active build directory's `dist/vc/vc`. Build presets include installation; always build before `update-dump`.

2. **Build was not regenerated**: Re-run `ninja` after adding a test so CMake's configured file globs discover it.

3. **Stale golden files**: Remove files explicitly when they are no longer declared as golden outputs.

4. **Hidden `.vbc` file**: Running `vc build .` from inside a test directory produces `.vbc` (hidden file) because the project name is ".". Run from the active build directory, e.g. `build/debug/`: `dist/vc/vc build ../../testsuite/v/<name>`.

5. **Timeout failures**: Tests that hang (infinite loops, deadlocks) are killed after 20 seconds and produce a non-zero exit code. The `exit_code.txt` will contain a timeout error string, not a number.

6. **New `_builtin` files**: Changes to `vc/_builtin/` affect ALL tests. Must regenerate all golden files and commit all changes.

7. **Missing pass dumps**: Usually a WF violation. Check which pass is the last one that produced output and investigate the next pass's WF definition.

```
