# Verona MLIR Generator Design Choices

The MLIR generator sits in between the AST and LLVM IR, and is meant to ease
lowering while still maintaining some high level concepts to allow us to optimise
the code without having to scan the whole IR on every pass.

The LLVM pipeline intermixes analysis passes with transformation passes and has
a way of tracking which transformation pass invalidates which analysis pass to
avoid re-running complex analysis passes. However, more often than not, the
transformation passes end up invalidating them anyway and a new run is needed.

In Verona, we'll be bundling all modules, including the standard library and all
source files, into one module. A full scan on that module for every analysis pass
would be prohibitively expensive.

LLVM IR has a fixed shape, with instructions belonging to basic blocks, which in
turn belong to functions. Instructions have "uses", which we can be traversed
cheaply, attributes and metadata, which can belong to more than one symbol, but
that only gets us so far.

Verona needs to know the type constraints of objects (imm, mut, iso) as well as
which region they belong to and if those regions have the same sentinel object.
We may also want to do intra-module optimisations if we know a symbol doesn't
escape the context (which is much cheaper than whole program optimisation).

For those reasons, Verona will need a light dialect to represent type constraints,
region information and module boundaries, which don't naturally convert to LLVM.

## Generation Simplicity

The new AST is much simpler than the previous. The new parser uses a lean typed
AST and converts all language into those nodes, which becomes much simpler to
lower to an IR. The parser also runs the type checks and inference, so the AST
we get at this stage is already complete, and checked for syntactic and semantic
errors.

Lowering errors can still occur, if the semantics would be correct in some cases
but not in this specific case, for example, when using it with a different
runtime functionality.

The MLIR dialect that we'll develop at this stage will be solely for representing
the lowering concepts in order to simplify optimisations. In theory, the AST
could have been directly lowered to LLVM IR, but the LLVM pipeline wouldn't be
able, for example, to optimise based on type and region information.

## Module Boundaries

Some symbols are private to the module and don't escape its boundaries. If we
lower all functions already mangled into the IR, we'd have to scan the whole
module every time for the functions and types that belong to them, to run module
local optimisations.

For this reason, we have chosen to create an MLIR `module` for each class/module,
including the root module. For example:
```
// Verona code in directory FooBar
class Foo { foo() { ... } }

// Equivalent MLIR structure
module @__ {
  module @FooBar {
    module @Foo {
      func @foo() {
        ...
      }
    }
  }
}
```

This structure, however, does not convert naturally to LLVM dialect. So we have
a later pass that mangles the names and moves all functions to the root module
just before lowering to LLVM dialect. The example above would be:

```
module @__ {
  func @____FooBar__Foo__foo() {
    ...
  }
}
```

And all calls to that function would also be changed in every other function that
calls this one.

## Load/Store Semantics

Stack variables are allocated by LLVM IR with an `alloca` instruction. MLIR has
a similar concept and we use that in the first implementation. The main problem
with this approach is that there is only one dialect that has alloca/load/store:
`memref`.

MemRefs represent the memory used by `tensors`, which are multi-dimensional
objects (matrices) of the same type. Verona is obviously not restricted to
tensors as types, so we'll need to extend this to structure types, with complex
layout semantics (ex. union types). For that, we'll need a set of dialect
instructions that can be later lowered directly into LLVM for allocating stack
space on complex layout semantics.

For now, every function argument and local declaration (`let`, `var`) are
allocated in the stack and its value stored in the memory space. This simplifies
code generation (all variable symbols point to an address), but it doesn't help
with value symbols and are really slow.

The LLVM pipeline, even at O1, recognises most of the argument, temporary and
local allocas as bogus and optimises them away, so this isn't a critical issue
from a performance point of view, but we will probably need to update the symbol
table soon to account for more types and states of the variables we hold (ex.
`let` vs `var`).

## Runtime Calls

The Verona runtime is where the magic happens. Allocating regions, choosing the
memory management strategy, scheduling behaviours, dispatching foreign language
code, etc. The AST, however, has only the semantics of the language (with hints
to the runtime behaviour), so we need to translate it at the MLIR lowering level.

For example, heap allocation, which LLVM uses `malloc` calls, Verona uses
`snmalloc`, which has a finer control over what we can do with the regions. In
the end, however, it will be the same: calls to the runtime library in IR.

The generator knows what runtime calls are appropriate for each AST construct,
insert those calls in the right places and makes sure to declare the methods
so that linking with the runtime library works.

## Arithmetic

Arithmetic is generated by the generator via pattern matching on the types and
"method" names. Adding two numbers (`a + b`) is the same as calling the method
`+` on `a` over `b` (`a.+(b)`), but these methods don't exist (how would we
implement addition if not with a `+`?).

So the generator has to find those that weren't declared and replace with
arithmetic if the types are compatible (int, float).

Users are still allowed to use `+` as a method name, in which case the generator
will recognise and call that method instead of transforming into arithmetic, even
if the types are compatible. For example:

```
// Verona code
class U32 {}
class Foo {
  static +(a: U32, b: U32) { ... }
}
main() : U32 {
  let a = 21 + 21; // This is simple arithmetic
  let b = Foo::+(21, 21); // This calls the method
}
```

## Type Constraints

TODO

## Region Information

TODO
