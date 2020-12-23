# MLIR passes

Long term, we want the following passes to run in MLIR:
 * Type inference & check (removing unknown types)
 * Free variable & liveness analysis (sanity check, phi nodes, clean-up calls)
 * Reification & type-based code ellision (instantiating all generics)
 * Region based alias analysis (disjoint regions never alias)
 * Dynamic call optimisation (vtable packing, dyn-to-static type-based conversion)

However, in the interest of parallel development, the current `1IR` will do all
type based (correctness) transformations and the `2IR` (MLIR based) will do the
final (perfornmance) steps.

Here, we describe the MLIR passes. Once we move `1IR` passes down to the `2IR`,
we should add the description of the passes here.

## Region based alias analysis

Different regions never alias. Regions are never accessed by more than one
thread at a time. Memory access is therefore deterministic if we can track the
provenance of all uses of functions that work on references, which is not always
possible.

This analysis tracks all uses of references in code (local, class) and function
arguments (global) to find uses that can or cannot alias. This allows
us to apply local optimisations such as folding sub-expressions, loop
optimisations such as wider vectorisation and even report warnings of potential
design issues.

Here's a summary of the aliasing rules, using both type-based (TBAA) and region-
based (RBAA) alias analysis:
 * `a.b` and `a.c` cannot alias, from the type, because they are different
   fields (TBAA).
 * `*a` and `*b` can alias if `a` and `b` have the same `mut` type, because
   they are both pointing to the same region (RBAA).
 * `*a` and `*b` can alias if `a` and `b` have the same `imm` type, because
   they are not in region though they will never alias with a write (RBAA).
 * `*a` and `*b` can alias if `a` and `b` have the same readonly type
   and it may alias with writes (RBAA).
 * `*a` and `*b` cannot alias if `a` and `b` have the same `iso` type,
   because they must point to disjoint regions (RBAA).
 * `*a` and `*b` cannot alias if `a` and `b` have different concrete types
   or interface types that are cannot both satisfied by the same concrete type
   (TBAA).

When the result of the analysis is that it *cannot* alias, we're free to do any
optimisation straight away. For those that *can*, we need further analysis to
separate them between *must*, *may* and *must-not*.

This is the analysis that tracks provenance (the origin of the references) to
make sure they come from disjoint memory. These can come from:
 * local variables, for example via assignments (`var`, `let`), in which case a
   local analysis is enough.
 * function arguments, in which case we need to find all callers of the function
   and separate between aliasing calls and non-aliasing calls. If we only have
   non-aliasing calls, we can optimise as before. If not, we may choose to create
   multiple versions of the function and change the original calls to match.
 * class fields, which could have been modified by a previous step, depending on
   runtime behaviour, and are therefore very hard to prove either way.

### IR state afterwards

After the alias analysis step, the IR shouldn't change much in structure.

If we found opportunities for optimisation at the MLIR level, then some code may
have simplified, but not changed in dialect support. For the rest of the alias
information, we should annotate the operations to carry on TBAA info into the
LLVM IR and allow LLVM to optimise on its own.

## Dynamic vs static calls

Static calls are cheaper than dynamic calls. The former is represented in MLIR
and LLVM IR natively and translate to direct jumps into code memory. The latter
has to find the object's vtable address, the method's offset and do an indirect
call on that address.

We will generate a function for every method, taking the receiver as the
first argument. Calls to concrete types are static call on the type's
function directly and therefore need no additional treatment.

For all the call sites where the target is an interface type, we should insert
indirect (vtable) calls for every class that is ever cast to an interface type,
containing only the methods that might be called via an interface.

This minimises the size and complexity of the vtables and improve runtime
performance avoiding unnecessary indirect calls.

### IR state afterwards

At this stage, all static calls should have moved to a direct `std.call` to
named (and possibly mangled) names that exist as functions in the IR, with the
appropriate lexical context (module name, etc).

## Virtual table colouring

Concrete classes implementing the same interface will have similar (if not
identical) method layout. Concrete implementation of generic functions can
proliferate concrete types, increasing the number of similar/identical type
representations.

If each type has its own vtable, we're spending a large amount of read-only
memory to store them all. This can also hurt caching, if objects are used in hot
code but are too far apart from each other in the read-only section.

In order to make the vtables more compact, we join all vtables into a single
block, where the type points to a position in that table and the offset of each
method is added to that position. Methods with the same name always have the
same offset. But methods that aren't shared across multiple types can reuse
offsets.

Example:
```
  class A { a(); b(); }
  class B { a(); c(); }
  class C { d(); e(); }
  class D { f(); }

  Offsets:
    a: 0
    b,c: 1
    d: 0
    e: 1
    f: 0

  Layout:
  +-----+
  |  a  | <- A
  +-----+
  |  b  |
  +-----+
  |  a  | <- B
  +-----+
  |  c  |
  +-----+
  |  d  | <- C
  +-----+
  |  e  |
  +-----+
  |  f  | <- D
  +-----+
```

This is a NP-hard problem, so it is impractical to find the optimal solution to
any reasonable number of types. Static solvers can find solutions in human
acceptable times, but are not compile-time acceptable. Heuristics can improve
some solvers (ex. PBQP), but those solvers are themselves complicated enough
to make it hard to predict worst case scenarios.

A reasonable solution is to implement this as a simpler heuristic programming,
with poorer quality (wider gaps in the layout) but with reasonable deterministic
compile times.

A greedy example in the Pony language:
https://github.com/ponylang/ponyc/blob/master/src/libponyc/reach/paint.c

### IR state afterwards

Here, dynamic calls should have been moved to a vtable lookup (compiler-generated
code) plus a `std.call.indirect` call into that address. This can be directly
lowered into LLVM dialect.

## Arithmetic lowering

Modern hardware have multiple instructions to perform arithmetic at low cycle
counts (1~4), including multiplication and division and sometimes even more
complex mathematical operations (inverse) and types (complex). But some hardware
(ex. embedded) may not have all of them, so we'll need a mix of lowering into
direct instructions and implementing a runtime library.

LLVM already has a runtime library for many architectures (compiler-rt) and has
pre-defined builtins and native operations (or sequence of operations) for all
basic arithmetic. A trivial lowering strategy would be, then, to use the existing
infrastructure in LLVM for that.

To lower native operations, we need to know how LLVM represents it. For example,
on 64-bit architectures, `U64 + U64` is a single operation, while in 32-bit
architectures, it's at least two (`add` + `carry-add`). This can be done in a
combination of different ways:

 * Implement native type arithmetic in C, lower to LLVM IR, merge with Verona IR
   and let LLVM inline it.
 * Implement LLVM builtin calls in Verona that will be resolved as direct calls
   in LLVM IR and thus correctly interpreted as the right kind of instruction.
 * Recognise type/function names and do a special LLVM lowering without having
   to implement the function at all at the Verona or C levels.

For the basic arithmetic and maths functionality, LLVM should be able to
identify, by the target triple and options, if the hardware has support for the
instructions of not and make the appropriate calls to the runtime library if not.

For more advanced functionality, we may have to resort to some runtime library
specific to Verona.

### IR state afterwards

After this pass, calls to arithmetic functions on native types should have been
converted to calls to either LLVM builtins (for special arithmetic semantics) or
"native functions" on Verona "arithmetic" types, that will be directly converted
to LLVM native types later on.

For example:
```
  // Verona dialect
  %0 = verona.call "+" [%obj] (%num) : !verona.S32
  // Standard dialect
  %0 = "llvm.add" (%obj, %num) : s32
  // LLVM dialect
  %0 = llvm.add i32 %obj, %num
```
