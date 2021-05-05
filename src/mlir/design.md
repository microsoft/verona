# Verona MLIR Generator Design Choices

The MLIR generator sits in between the AST and LLVM IR, and is meant to ease lowering while still maintaining some high level concepts to allow us to optimise the code with Verona-specific semantics before we lower to LLVM IR and lose information.

Verona needs to know the type constraints of objects (imm, mut, iso) as well as which region they belong to and if those regions have the same sentinel object.

For those reasons, Verona will need a light dialect to represent type constraints, region information and module boundaries, which don't naturally convert to LLVM IR.

## Generation Simplicity

The new AST is much simpler than the previous.
The new parser uses a lean typed AST and converts all language into those nodes, which becomes much simpler to lower to an IR.
The parser also runs the type checks and inference, so the AST we get at this stage is already complete (explicit concrete monomorphic types, reified code only), and checked for syntactic and semantic errors.

If an error is detected when generating MLIR, however, we still have the original source locations and can return the errors to the parser's own diagnostics infrastructure.

The MLIR dialect that we'll develop at this stage will be solely for representing the lowering concepts in order to simplify high-level optimisations before lowering to LLVM IR.
In theory, the AST could have been directly lowered to LLVM IR, but the LLVM pipeline wouldn't be able, for example, to optimise based on type and region information, memory management mechanisms (ex. ref count), object layout and representation, etc.

## Load/Store Semantics

Stack variables are allocated by LLVM IR with an `alloca` instruction.
MLIR doesn't support allocating stack memory with anything other than `memref` types, but it does support the LLVM dialect, which has full LLVM semantics.
In this preliminary implementation, we use the LLVM dialect for alloca/load/store/GEP operations when dealing with stack objects.
This approach seems very clean and sensible, and it could be used for accessing heap variables as well.

For now, every function argument and local declaration (`let`, `var`) are allocated in the stack and its value stored there.
This simplifies code generation (all variable symbols point to an address), but it clutters the IR.

The LLVM pipeline, even at minimum optimisation levels, recognises most of the argument, temporary and local allocas as bogus and optimises them away, so this isn't an immediately critical issue from a performance point of view, but we will want to clean up some of those repetitive patterns soon enough.

## Runtime Calls

The Verona runtime is where the magic happens.
Allocating regions, choosing the memory management strategy, scheduling behaviours, dispatching foreign language code, etc.
The AST, however, has only the semantics of the language (with hints to the runtime behaviour), so we need to translate it at the MLIR lowering level.

For example, heap allocation, which LLVM IR generally uses calls to `@llvm.malloc`, Verona uses `snmalloc`, which has a finer control over what we can do with the regions.
In the end, however, it will be the same: calls to the runtime library in IR.

The generator knows what runtime calls are appropriate for each AST construct, and can insert those calls in the right places and make sure to declare the methods so that linking with the runtime library works.

## Classes & Modules

Classes and modules are the same concept in Verona.
Each source sub-directory containing Verona files is a module, and all its Verona files share the same namespace, named after the directory name.
The base source directory is also a module, and its sub-directories will be included in the "path" for importing modules.

**QUESTION: How do we mark the `main` function?**
Currently, if there is a function called `main`, it's lowered without scope to help testing execution, but we need more strict rules, or perhaps annotation, to mark the entry point.

Classes can have sub-classes and their scope is stacked upon each other.
Calling for a method or accessing a field on an object of a type in a different sub-tree needs full scope up to the shared root.
The parser will resolve all those issues and will canonicalise the representation for all access to be fully scoped.

Class types will be represented in a mix of LLVM structure types (for fields) and a `vtable` (as an array of pointers) for the dynamic methods.

## Functions

Functions are a native concept in MLIR and LLVM IR, and lowering free and static functions and calls to them is trivial.
Dynamic resolution, however, is a bit more tricky.
We'll need a `vtable` for each class, containing all the functions, so that we can generate code that will take the type of the object, find its `vtable` and calculate the offset of the method to find its pointer and indirectly call that.

We will eventually do selector colouring (http://www.lirmm.fr/~ducour/Publis/RD-color06.pdf), but that doesn't change much what the compiler generated code will be at runtime.
In this case, we'll try to join all `vtable`s into one big array to minimise the gaps between used offsets.
Those gapes are consequence of the need of different classes to have the same offset for methods that implement the same interface.

MLIR allows modules to be nested, but the LLVM dialect doesn't, so lowering needs further special passes.
However, the symbol table for function names in the sub-modules doesn't nest, so requesting a symbol in a different sub-tree still needs traversal of modules to find the right one.

For those reasons, and because we already get fully resolved scope from the AST, we have decided to have a single global module and lower all functions with fully qualified and mangled names.

Arguments are always allocated on the stack and their symbols are added to the symbol table.
Return values, for functions that do return a type, and if not explicitly returned (ex: `throw`, `return`) will be the last expression evaluated in the block.

Lambdas will be lowered as an anonymous structure with captures as fields and `apply()` as the method for when the lambda is called.
That transformation should have already been done by the parser and represented in the AST by the right basic constructs.

NOTE: Current `lambda` node is the implementation of function's bodies, too.
The current implementation is probably wrong and will need to change once we actually implement lambdas.

## Scope and lowering order

Verona doesn't restrict declaration order, variables can be declared as having types that haven't been declared yet.
This must be the case because different files in the same directory share the same namespace and there is no `include` functionality.
However, MLIR lowering is restricted to only call functions and use types that already exist, or the modules don't validate.

Trying to create opaque types and functions to resolve later would be tricky, as the MLIR builder needs to know its insertion point and replacing all symbols later isn't a trivial operation.

The best option is for the AST to declare all methods beforehand, and then lower the tree for the module, so that all calls are explicit and defined.
Since the parser already has to do that for its own type checks, it would be trivial to add declaration nodes to the top of the tree.

## Arithmetic

Arithmetic is generated by the generator via pattern matching on the types and selectors.
Adding two numbers (`a + b`) is the same as calling the method `+` on `a` over `b` (`a.+(b)`), but these methods don't exist (how would we implement addition if not with a `+`?).

So the generator has to find those that weren't declared and replace with arithmetic if the types are compatible (int, float).

Users are still allowed to use `+` as a method name, in which case the generator will recognise and call that method instead of transforming into arithmetic, even if the types are compatible.
For example:

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

Eventually, all types will be clear on the AST, but this isn't the case right now.
For now, we have to add truncations and extensions where it isn't clear.

For example:
 * Constants of type `int` or `float` are created as 64-bit numbers and then truncated if assigned to 32-bit values.
 * Arithmetic between types of different width have the smallest type promoted to the largest.

Also, due to a constraint on the standard dialect in MLIR, we currently only support sign-less integers (neither signed nor unsigned).

We also currently delay stack allocation until the type is defined (node `Oftype`), because we don't yet have an idea of the type of local definitions in their own nodes.
This makes for cumbersome pattern of:
 1. Adding an empty value on the symbol table (to reserve the name in the table)
 2. Allocating enough space when we see a type declaration (now the value is the address of the new memory)
 3. Accessing the address via references to the variable name (map lookup)

Once more nodes have types (for example: `let`, `var`) we can skip the second step.

Lastly, the `numbers` module contain only empty classes with the name of the numeric types (U32, F64, etc).
The current implementation assumes they won't be implemented and we need to pattern-match type names and operations for arithmetic.

But this conversion, if too early, can get in the way of lowering similarly named methods (ex. `+`) while we still have automatic type conversion.
If all the casts and conversions are explicit in the AST, then the pattern matching would only pick up arithmetic for numeric types that are identical.

## Select semantics

`select` is a ternary operator with the following arguments: `expr`, `typenames`, `args`.

`expr` can be either:
 * A dynamic selector (object reference) for field access or method call
 * The left-hand side of a binary operation (ex: `+`)

`typenames` is a list for a fully qualified name (scope + name) of functions, or an arithmetic operation.
If the AST later implements arithmetic as methods in a numeric class, these are one and the same.

`args` can be either:
 * Arguments of the function call (as a tuple if more than one)
 * The right-hand side in a binary operation (ex: `-`) or the only operand of a unary operator (ex: `-`)

The current (naive) assumptions are:
 * If `expr` is an object, the access is *dynamic*
 * If `typenames` resolve to an existing field in the type of `expr`, this is a field access
 * If `typenames` resolve to an existing function, this is a call
 * Otherwise, this is an arithmetic operation and both `expr` and `args` must be numeric and _compatible_

If the AST canonicalises static method calls via objects into fully qualified `typenames`, this works fine.

For example:
```
class Foo {
  dynamic(self: Foo, a: U32) { ... } // dynamic method, self is Foo
  static(b: F64) { ... }             // static method, no self
};

func() {
  let f: Foo;
  let x: U32 = f.dynamic(42); // This is a dynamic call
  let y: F64 = f.static(3.14); // This is a static call, of type Foo via object f

  // Should the line above be converted to this before getting to MLIR?
  let y: F64 = Foo::static(3.14);
}
```

## Assign semantics

`assign` is a binary operator with the following arguments: `left` and `right`, both generic expressions.

`left` is an expression that returns an address, where the value of `right` will be stored into.

The result of the assignment is the previous value that was held on that address, so a load is performed if the address existed.

If the address didn't exist (ex: new declarations), the appropriate allocation will be done:
Stack allocation for local variables and heap allocation for types with defined regions.
No pre-loads will be done in that case and the return value must be unused (guaranteed by the parser checks).

It is an error for the `left` expression not to return an address, also already checked by the parser.

**QUESTION: Verona references are not actual pointers, do we have _pointers to pointers_?**
If the `right` expression also yields an address, can it be mapped directly, instead of being loaded and stored again?

## Type capabilities and regions

The three main type capabilities are: `imm`, `mut` and `iso`. See https://microsoft.github.io/verona/explore.html#regions.

Immutable regions will always be allocated on the same reference counted "region", via runtime calls.
They can only be read from, so there are no concurrency issues. They can be collected when there are no more references to them.

Mutable regions will be allocated on the heap, but cannot be used directly, unless they're in a "forest", accessible through their roots (`iso`s).

`Cown` root will be scheduled via the `when` keywords and give no guarantee of completion time or order.
The runtime calls will be just to create the `cown` and to push it into the scheduler.

**QUESTION: Do stack roots also push behaviours to the scheduler or do they execute synchronously?**
Either way, they should also yield similar runtime calls.

Regions can also have two more properties:
 * Which memory model they use (ref. count, GC, arena, etc)
 * Which sandbox type, if any, they belong to (native C++, Rust on Wasm, etc)

The syntax for these two properties isn't defined yet, so for now, the code-gen will assume no sandbox of arena type (release all at the end).

## Exceptions and non-local return

TODO

## Match and dynamic selection

TODO
