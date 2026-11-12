---
name: verona-conventions
description: Verona compiler coding conventions, C++ style, trieste patterns, and project structure. Applies when writing or reviewing Verona compiler code.
user-invocable: false
---

# Verona Compiler Conventions

## C++ Style (C++20)

- `snake_case` for functions, variables, namespaces. `PascalCase` for types/classes.
- Token definitions use `PascalCase`: `inline const auto TypeName = TokenDef(...)`.
- Allman braces (opening brace on own line). Exception: single-line lambdas.
- No `goto`. Use flags, early returns, or helper functions.
- `#pragma once`, not include guards.
- Project headers first, then system/library headers separated by a blank line.
- Use `constexpr`/`consteval`, structured bindings, `if constexpr`, `std::span`, `[[nodiscard]]`.
- Prefer `static constexpr` over `#define` for constants.

## Trieste Patterns

- **Node construction**: `Node n = SomeToken` creates a node. NOT `auto n = SomeToken` (copies TokenDef).
- **Child access**: `n / ChildToken` (named WF accessor), not `n->at(i)` (positional).
- **Append children**: `node << child`. Set value: `node ^ value`. Duplicate: `clone()`.
- **Clone shared nodes**: Always `clone()` nodes inserted into multiple AST locations.
- **Token comparison**: `node->type() == OtherToken`, NOT `node == other_node` (compares pointers).
- **Error handling**: `err(node, "message")` wraps in `Error << ErrorMsg << ErrorAst`.
- **Tree mutation**: `traverse()` iterators are invalidated by `replace()`/`erase()`. Collect first, mutate after.
- **WF declarations**: Every pass declares output shape. `<<=` for children, `++` for zero-or-more, `|` for choices.
- **Symbol table**: `lookup()` walks up + follows includes. `lookdown()` searches own symtab only, no includes. `look()` is immediate, no flags.
- **Pass structure**: `PassDef` with name, WF, direction (`dir::topdown`/`dir::bottomup`/`dir::once`), rewrite rules. `pre()`/`post()` hooks.

## Build & Test

- From the repository root, use `cmake --preset debug`, `cmake --build --preset debug` (includes installation), and `ctest --preset debug`. Require CMake 3.28+ and Ninja. Use the `release` presets for performance work; builds coexist in `build/debug/` and `build/release/`.
- Rebuild and install with the active build preset before debugging or validating compiler/runtime behavior on the current branch.
- Run `vc` from the active build directory, e.g. from `build/debug/`: `dist/vc/vc build ../../testsuite/v/hello`. Do NOT cd into source dir.
- Debug with the binaries under the active build directory's `dist/` tree. Do not use non-installed build outputs when validating behavior.
- `ctest --preset debug -j$(nproc)` from the repository root for the full Debug test suite.
- `-p <passname>` stops after a pass. `--dump_passes=<dir>` dumps intermediate ASTs.
- Golden files: `ninja -C build/debug update-dump` (only for intentional golden changes).
- `exit_code.txt` has NO trailing newline (`printf '0'`, not `echo`).
- Treat compile-time and runtime test failures as real regressions until they are investigated on the current baseline. Do not assume a failure predates your change without verification.

## Test Conventions

- Tests in `testsuite/v/` must be self-contained. No external deps, no `use "https://..."`.
- Do NOT write `use "_builtin"` — it is always implicitly available.
- Bitmask exit code pattern: `var result = 0;` then `if cond { result = result + N; }` with powers of 2. Exit 0 = all passed.
- Compile-error tests: `exit_code.txt: 1` in `compile/`, no `run/` directory.
- Structure: `testsuite/v/<name>/<name>.v` with golden dirs `<name>/<name>/compile/` and `<name>/<name>/run/`.
- FFI tests must include at least one reachable FFI call from their lib to trigger init function reification.

## Verona Source Syntax

- Class definitions: `myclass[T] { ... }` (no `class` keyword).
- Fields need semicolons: `val: T;`. No `let`/`var` on class fields.
- No semicolons after closing braces of `for`, `if`, `while`.
- `new { field = val }` — no class name after `new`.
- `use X` imports for unqualified lookup. `use "url"` for packages. `_builtin` is always implicit.
- `(obj.field)(args)` to call `apply` on field access result.
- `Type(args)` constructor sugar calls `create` method (e.g., `callback(f)` → `callback::create(f)`).
- `x.method` for zero-arg methods (no parens).
- `!expr` for boolean negation.
- Array literals: `::(expr, ...)`.
- Match: `(match expr { (pattern) -> body; ... }) else (default)`.
- FFI calls: `:::name(args)` for direct FFI, `ffi::func(args)` for `_builtin/ffi/` wrappers.
- FFI data passing: arrays pass as pointers to their data (e.g., `array[u8]` → `uint8_t*`), objects pass as pointers to their fields (like a C struct). No wrapper types cross the FFI boundary.
- Packages: `use "url" "tag"` imports from a git repo. `use "~/dev/pkg" "main"` works for local repos.

## Architecture

- Small, composable passes. Many small passes > few large ones.
- Layered dependencies — lower layers never depend on higher ones.
- Frontend (vc, passes 0–9): parse → structure → ident → sugar → functype → dot → application → anf → infer → reify.
- Backend (vbcc library, passes 10–13): assignids → validids → liveness → typecheck.
- `assert()` liberally for invariants.
- Errors are AST nodes, not exceptions.

## Adding New Ops/Builtins

Adding a new bytecode op requires updates in ~15 places across the codebase. Use this checklist:

1. **Token def**: `include/vbcc.h` (token), `include/vbci.h` (Op enum)
2. **Frontend WFs** in `vc/lang.h`: `wfExprDot`, `wfPassDot` (with `<<= Args`), `wfBodyANF`, `wfPassANF` (with `<<= wfDst * wfSrc`)
3. **Frontend passes**: `dot.cc` (builtin reg), `anf.cc` (lowering), `infer.cc` (type tracking), `reify.cc` (IR transform)
4. **Backend WFs** in `include/vbcc.h`: `wfStatement`, `wfIR`
5. **`Def` pattern** in `vbcc/lang.h`: manually maintained list — MUST include if op has a dst `LocalId`
6. **Liveness** in `vbcc/passes/liveness.cc`: manually categorized ops — add to correct use/def category
7. **Bytecode** in `vbcc/bytecode.cc`: encoding handler
8. **Type check** in `vbcc/passes/typecheck.cc`: if needed
9. **Interpreter** in `vbci/thread.cc`: op handler + op name in name array

Missing items 5 or 6 causes "undefined register" errors in later passes, not at the registration site.

## Debugging

- Trace data flow, not just error sites. Root cause is often upstream.
- For tricky compiler/runtime/FFI bugs, do principled analysis first: reproduce, trace, inspect dumps, and localize the fault before changing semantics.
- Before speculative interpreter/runtime changes, explain the evidence and proposed fix and get feedback.
- "Bad type" runtime errors usually mean unrefined default literals in compiler output. Check infer pass dump.
- `node->type() == other->type()` for token comparison, NOT `node == other_node`.
- Write targeted tests that expose the bug before fixing.
- Runtime failures matter as much as compile-time failures. Investigate both with the current branch's `dist/` binaries.
- Do not describe runtime behavior as flaky or dismiss it as harness noise without first reproducing and tracing it to a concrete cause.
- Durable Verona-specific workflow rules belong in this skill or the repo `AGENTS.md`, not in agent-specific memory stores.
