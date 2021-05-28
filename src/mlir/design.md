# Verona MLIR Generator Design Choices

The MLIR generator sits in between the AST and LLVM IR, and is meant to ease lowering while still maintaining some high level concepts to allow us to optimise the code with Verona-specific semantics before we lower to LLVM IR and lose information.

Verona needs to know the type constraints of objects (imm, mut, iso) as well as which region they belong to and if those regions have the same sentinel object.
These constraints and region information isn't natively represented in MLIR.
We could add annotations to the lowered code (potentially multiple instructions) with metadata, but that would require us to search and bundle instructions every time when looking for high-level patterns.

For those reasons, Verona will need a light dialect to represent type constraints, region information and module boundaries, which don't naturally convert to LLVM IR.
This will allow us to do high-level optimisatins cheaply, before we lower that code to LLVM.

## Generation Simplicity

The new AST is much simpler than the previous. 
It has only primitive control flow (`assign`, `select`, `match`, `when`, etc), from which higher level control flow is constructed (conditionals, loops, calls, member access, etc).
It also has support strucures with all reachable types and functions, which will be the main driver of the MLIR lowering.

The parser runs the type checks and inference, so the information we get at this stage is already complete (explicit concrete monomorphic types, reified code only), and checked for syntactic and semantic errors.

Verona also doesn't restrict declaration order. Functions and classes can be declared at any time in the module to be used anywhere.
To avoid order issues in the IR, we'll do multiple passes on the structures:
1. Declaration pass: Defines reachable type structures and functions, including the entire standard library files.
2. Lowering pass: Lower the reachable AST nodes using the structures above and defining the functions for the prototypes above.

In theory, the AST could have been directly lowered to LLVM IR, but the LLVM IR is too low level for some analysis we want to do (ex. region aliasing) and it would be counter-productive to do so at that stage.

## Classes & Modules

Classes and modules are the same concept in Verona.
Each source sub-directory containing Verona files is a module, and all its Verona files share the same namespace, named after the directory name.
The base source directory is also a module, and its sub-directories will be included in the "path" for importing modules.

Calling for a method or accessing a field on an object of a type in a different module needs full scope.
The parser will resolve all those issues and will cannonicalise the representation for all access to be fully scoped.

Class types will be represented in a mix of LLVM structure types (for fields) and a `vtable` (as an array of pointers) for the dynamic methods.

## Functions

Functions are a native concept in MLIR and LLVM IR, and lowering free and static functions and calls to them is trivial.
We'll declare all reachable functions beforehand, then lower the AST (implementing the prototypes as full functions).
Dynamic resolution and selector colouring will be done at the parser level, and calls to dynamic methods in the AST will be a node with base pointer and offset.

MLIR modules have a symbol table for functions, which we use to match declarations against definitions and calls.

Arguments by value or reference have ABI reserved space (registers, stack), so we don't need to allocate them again (the `alloca` _trick_).
Return values, for functions that do return a type, will be the last expression evaluated in the block.

Lambdas will be lowered as an anonymous structure with captures as fields and `apply()` as the method for when the lambda is called.
That transformation should have already been done by the parser and represented in the AST by the right basic constructs.
At the MLIR level, they'll be allocated and dispatched in the usual way (stack/heap + call/behaviour dispatch).

**QUESTION: How do we mark the `main` function?**
Currently, if there is a function called `main`, it's lowered without scope to help testing execution, but we need more strict rules, or perhaps annotation, to mark the entry point.
We could also have a class named `Main` which is restricted to only one per program, with a method `main(int, int**) : int`, which is the entry point.
We then create a compiler-generated entry point and call that method with the command line arguments, redirecting the return value.

## Runtime Semantics

Cowns, behaviours and regions are created and executed by the runtime.
At compile time, all we need to do is to make sure we call the right runtime functions.

Memory allocation will be selected by the type of the regions, which will indicate which `snmalloc` functions we'll call for each of their objects.
Cown reservation and behaviour execution (`when` blocks) will become a call to the appropriate scheduler functions with references to the right objects.

The parser will have already done liveness analysis, so we know when variables (including heap-allocated regions) go out of scope.
It will also have introduced nodes for their deallocations, which will be converted into calls to `snmalloc` or garbage collector hints.

## Load/Store Semantics

Stack variables are allocated by LLVM IR with an `alloca` instruction.
MLIR doesn't support allocating stack memory with anything other than `memref` types, but it does support the LLVM dialect, which has full LLVM semantics.
We use the LLVM dialect for alloca/load/store/GEP operations when dealing with stack/heap objects.

Local declarations (`let`, `var`) without region information are allocated in the stack, via LLVM's `alloca` instruction and their types.
Region objects are allocated using runtime calls to `snmalloc` with the right region types.

This may be too low level, so it's a strong candidate to move up to a dialect operation.
But we will only do so if there is a clear path for optimisation.

## Arithmetic and Builtins

Arithmetic and builtins are implemented in Verona via direct calls to intrinsics and MLIR/LLVM operations by name (string).
Each numeric class will have their operations defined in the standard library and the compiler will detect the call as _internal_ and create the appropriate intrinsic / operation.
Builtins (number conversion, checks, cpu-specific instructions) will also be implemented in the standard Verona library.

Adding two numbers (`a + b`) is the same as calling the method `+` on `a` over `b` (ex. `a.+(b)` or `+(a, b)`).
These calls will be matched against the already implemented methods in those classes and behave like a normal call (dynamic or static).

These methods are small enough that the LLVM inliner pass will convert all calls into single operations.
If any of them proves more complex than the inliner can handle, we can add the _always_inline_ attribute to them.

## Type Constraints

Eventually, all types will be clear on the AST, but this isn't the case right now.
For now, we have to add truncations and extensions where it isn't clear.

For example:
 * Constants of type `int` or `float` are created as 64-bit numbers and then truncated if assigned to 32-bit values.
 * Arithmetic between types of different width have the smallest type promoted to the largest.

Also, due to a constraint on the standard dialect in MLIR, we currently only support sign-less integers (neither signed nor unsigned).
Arithmetic will be lowered in the LLVM dialect, so will contain full support for signed and unsigned as well as all intrisics.

We also currently delay stack allocation until the type is defined (node `oftype`), because we don't yet have an idea of the type of local definitions in their own nodes.
This makes for cumbersome pattern of:
 1. Adding an empty value on the symbol table (to reserve the name in the table)
 2. Allocating enough space when we see a type declaration (now the value is the address of the new memory)
 3. Accessing the address via references to the variable name (map lookup)

Once more nodes have types (ex: `let`, `var`) we can skip the second step.

## Type capabilities and regions

The three main type capabilities are: `imm`, `mut` and `iso`. See https://microsoft.github.io/verona/explore.html#regions.

Immutable regions are created from "frozen" mutable regions, via runtime calls.
They can only be read from, so there are no concurrency issues (other than atomic reference counting). They can be collected when there are no more references to them.

Mutable regions will be allocated on the heap, but cannot be used directly, unless they're in a "forest", accessible through their roots (`iso`s).

`Cown` root will be scheduled via the `when` keywords and give no guarantee of completion time or order.
The runtime calls will be just to create the `cown` and to push it into the scheduler.

**QUESTION: Do stack roots also push behaviours to the scheduler or do they execute synchronously?**
Either way, they should also yield similar runtime calls.

Regions can also have at least two more properties:
 * Which memory model they use (ref. count, GC, arena, etc)
 * Which sandbox type, if any, they belong to (native C++, Rust on Wasm, etc)

The syntax for these two properties isn't defined yet, so for now, the code-gen will assume no sandbox of arena type (release all at the end).

## Exceptions and non-local return

TODO

## Match and dynamic selection

TODO

## Specific Nodes

Some nodes currently have a more nuanced semantics, and can describe a multitude of cases.
In the future, the AST will have more node types and less ambiguity, so these issues will disappear eventually.

### Select semantics

At present, `select` is a ternary operator with the following arguments: `expr`, `typenames`, `args`.

`expr` can be either:
 * A dynamic selector (object reference) for field access or method call
 * The left-hand side of a binary operation (ex: `+`)

`typenames` is a list for a fully qualified name (scope + name) of functions, or an arithmetic operation.
If the AST later implements arithmetic as methods in a numeric class, these are one and the same.

`args` can be either:
 * Arguments of the function call (as a tuple if more than one)
 * The right-hand side in a binary operation (ex: `-`) or the only operand of a unary operator (ex: `-`)

The current (wrong) assumptions are:
 * If `expr` is an object, the access is *dynamic*
 * If `typenames` resolve to an existing field in the type of `expr`, this is a field access
 * If `typenames` resolve to an existing function, this is a call

The AST will later canonicalise static method calls via objects into fully qualified `typenames`.

### Assign semantics

`assign` is a binary operator with the following arguments: `left` and `right`, both generic expressions.

`left` is an expression that returns an address, where the value of `right` will be stored into.

The result of the assignment is the previous value that was held on that address, so a load is performed if the address existed.

If the address didn't exist (ex: new declarations), the appropriate allocation will be done:
Stack allocation for local variables and heap allocation for types with defined regions.
No pre-loads will be done in that case and the return value must be unused (guaranteed by the parser checks).

It is an error for the `left` expression not to return an address, also already checked by the parser.

### Let, Var and Oftype semantics

Currently, neither `let` nor `var` have types, so if we just declare a variable without types, we don't know what size to allocate.
This will be fixed by future implementations of the AST, where all types are explicit, but nodes will need type information in addition to the use of `oftype` nodes.

Currently, this Verona code:
```
  let x : F64 = 3.1415;
```

Generates this AST:
```
1:  (let x)
2:  (oftype (ref x) (typeref [ (typename F64[]) ] ))
3:  (assign (ref x) (float 3.1415))
```

Which leads the three step process:
1. Reserve the name `x` with an empty value
2. Allocate `sizeof(F64)` and update the symbol table
3. Store `3.14` into the address in the symbol table

This mandates all declarations to have an `oftype` node before any use (read/write).
But temporaries don't have such luck:
```
  let y : F64 = x + 2.7183;
```

Generates this AST:
```
1:  (let y)
2:  (oftype (ref y) (typeref [ (typename F64[]) ] ))
3:  (assign (let $0) (float 2.7183))
4:  (assign (ref y) (select (ref x) [ (typename + []) ] (ref $0)))
```

Note in line `3`, the definition of `$0` has no type nor `oftype` node.
Currently, the generator guesses the type by looking at the expression on the right, but this may raise type conversion issues.

There are at least two solutions:
1. Make sure every temporary is properly declared and has an `oftype` node
2. Add type information to other nodes, like `let` and `var`.

Solution `1` is probably simpler to implement and shouldn't change the semantics of the AST.
Solution `2` might be necessary if there are patterns (unclear at this point) that can't be fixed with an `oftype` node.
