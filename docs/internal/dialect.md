# Verona MLIR Dialect

## WARNING

This is a preliminary document under heavy discussion.
The ideas here are mostly right, the details, not necessarily.
The dialect can change radically from the time this document was written to the time it's actually implemented.
Please check the actual code for the real semantics.

# Overview

The Verona dialect is composed of two parts: new operations and new types.
The purpose of the dialect is to keep the semantics high-level throughout MLIR optimisations to avoid lower to runtime calls too early.

The main reasons are:
* Handling MLIR operations (merging, eliding, duplicating) is easier than generic calls (especially if more than one per op)
* Operation semantics can change depending on the objects and types it operates on
* Optimisations can simplify the objects' types (ex. from a union to a concrete type)
* Leaving the expansion to the final lowering minimises cleanups of unused runtime calls

# Dialect Types

Verona types have an algebraic semantics and need to be handled carefully by the compiler.
However, all of the inference and checks are done by the parser, so the MLIR layer can assume the conversions, if any, are correct.
Therefore, there will be no type checks in MLIR and any conversion between types (ex. a concrete argument to an interface declaration) will be via a `cast` operation that is equivalent to a NO-OP.

Type representation in MLIR is not necessary for correctness.
With all checks done at parser level and the NO-OP casts at MLIR level, the semantics should not change amidst optimisations.
However, by keeping the information for longer, we can elide a number of run-time checks at the final lowering, improving performance.

If we decide to move type inference to the MLIR level in the future, this will also help implement the passes, as it is similar to the original representation designed for that.

## Concrete Types

These will be represented by _named_ LLVM `StructureType` pointers.
Access to fields are done via structure offset, with `embed` fields directly accessed and the rest resolved with a pointer indirection.
Access to methods will be done by the parser and at this point, dynamic dispatch will already have a `vtable` pointer and an offset.
Direct (static) calls are done directly as an MLIR call without further complications.

Fields can be reordered for efficiency reasons, for example stronger alignment first to avoid misaligned load/store on later fields.
This has to be an early static decision, to avoid changing access patterns all over the code later on.
So it's also likely going to be simple and predictable.

## Interface Types

Abstract interface types that survive to the MLIR layer have their own descriptors, so their own LLVM `StructureType` pointers and types.
Those are solely used for run-time checks and all access to fields and methods will be done through the run-time concrete object pointer instead.

## Intersection Types

Intersection types will end up as concrete types themselves and treated the same as above.
For example `(A & B)` will end up as a new `A_&_B` structure with the intersection of both `A` and `B` fields and methods.

Intersection of a concrete or interface type with a capability will be represented specially as `!cap<Type>`.

The four capabilities are:
* `!iso<Type>` (Isolated)
* `!mut<Type>` (Mutable)
* `!imm<Type>` (Immutable)
* `!ro<Type>` (Read-Only)

This is so that we can carry the capability semantics throughout the MLIR passes and avoid emitting too many run-time checks at final lowering (see below).

## Union Types

Union types will be represented by `!join<A, B>`.
Verona types are algebraic, so `!join<A, !join<B, C>>` is equivalent to `!join<A, B, C>`.
Verona also normalises types in [disjunctive normal form](https://en.wikipedia.org/wiki/Disjunctive_normal_form), so they will be represented as the latter in MLIR.

We need to keep union types represented in such a way because this will allow us to simplify calls and checks after optimisations.

For example:
```ts
{ // Original version, needs to differentiate between mut and imm at run time
  var a : A & mut | B & mut | C & imm = ...;
  in MLIR -> %a = ... : !join<mut<A>, mut<B>, imm<C>>
} // Exit scope, needs to collect for mut, do nothing for imm

{ // An optimisation (inline, propagation, etc) drops the C type, so it becomes only muts
  var a : A & mut | B & mut = ...;
  in MLIR -> %a = ... : !join<mut<A>, mut<B>>
} // Now, there's no need to check if its mut or imm, leaving the scope means update gc/ref count only.
```

## Tuple Types

Tuple types are _unnamed_ LLVM `StructType` pointers.
There are no methods to call, only field access, so no need for headers or `vtable`s.
All fields are `embed` and can be reordered for efficiency (like concrete classes).

# Dialect Operations

The Verona dialect operations are directly related to region run-time semantics.
Objects are created inside regions of a specific type and can be moved around, have external references in other regions, be garbage collected, etc.

Verona language concepts will be directly lowered to dialect operations in a 1:1 relationship.
Those operations will have specific run-time behaviour and will depend on the types they operate on.
By treating them as abstract concepts throughout optimisation passes, but changing the operands and types they operate on, we can more efficiently lower the right calls to the runtime libraries as late as possible.

The operations and their equivalent language and runtime calls are on the table below.
Some of the syntax isn't decided yet (ex. region types), but we'll have some syntax for those soon enough.
Undecided syntax is marked with a question mark (?).

| Dialect Operation | Verona Language | Runtime Call |
| ----------------- | --------------- | ------------ | 
| `%x = create(!Type, !RegionType) : !iso<Type>` | `var x = new Type::create() as RegionType?` | `RegionType::create(alloc, Type) + call to Type::create()` |
| `%y = alloc(%x : !iso<Ty1>, !Ty2) : !mut<Ty2>` | `var y = new Type::create() in x` | `RegionType(x)::alloc(alloc, x, Type) + call to Type::create()` |
| `move(%x : !iso<Type>, %z : !mut<Type>)` | `move?(x : iso, z : mut)` | `RegionType::insert<YesTransfer>(alloc, x, z)` |
| `extref(%x : !iso<Type>, %z : !mut<Type>)` | `extref?(x : iso, z : mut)` | `RegionType::insert<NoTransfer>(alloc, x, z)` |
| `merge(%x : !iso<Type>, %z : !iso<Type>)` | `merge?(x : iso, z : iso)` | `RegionType::merge(alloc, x, z)` |
| `tidy(%x : !iso<Type>)` | `tidy?(x : iso)` | `RegionType::gc(alloc, x) or NO-OP` |
| `drop(%x : !iso<Type>)` | Goes out of context | `Region::release(alloc, x)` |
| `kill(%x : !mut<Type>)` | Goes out of context | `RegionType::???(alloc, x) or NO-OP` |
| `freeze(%x : !iso<Type>)` | `Builtin::freeze(x : iso)` | `Freeze::apply(alloc, x)` |
| `%y = cast(%x : !Ty1) : !Ty2` | `<compiler-internal>` | `NO-OP` |