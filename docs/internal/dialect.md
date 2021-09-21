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

## Throw types

Exception handling is implemented as non-local returns, and are identified by the keyword `throw` with types `Throw[T]`.
The basic implementation is for `throw` to set a flag when returning, so the `catch` can check (and clear) the flag and branch conditionally to error handling blocks.

The type of a throw in MLIR is `!throw<T>`, to differentiate from `T`, which doesn't have the exception flag.
This makes the distinction between `std.ret` and `verona.throw` clear, with the former returning `T` and the latter always returning `!throw<T>`.

## Region type

The new semantics only allows one region to be writeable at any given point, so when creating new objects, we know in which region, if any, to allocate it.
But to know which is the current region at any given time, the ABI has to change so that every function receives a region object and uses it for the `alloc` operation.

For this, we need a new type, `!region`, which is a pointer to an object that defines the region, or `undef` if there are no regions defined.
See `alloc` below for more information on the semantics of the `!region` type.

# Dialect Operations

The Verona dialect operations are directly related to region run-time semantics.
Objects are created inside regions of a specific type and can be moved around, have external references in other regions, be garbage collected, etc.

Verona language concepts will be directly lowered to dialect operations in a 1:1 relationship.
Those operations will have specific run-time behaviour and will depend on the types they operate on.
By treating them as abstract concepts throughout optimisation passes, but changing the operands and types they operate on, we can more efficiently lower the right calls to the runtime libraries as late as possible.

## Object Creation

New objects are allocated with the operation:

```mlir
alloc(%region: !region, !ObjectType, %args...): !join<iso<ObjectType>, mut<ObjectType>>
```

where:
 * `%region` is a pointer to the object that defines the region.
 * `ObjectType` is a `!verona.type` that defines the structure to be allocated.
 * `%args` is the list of initialisers for the fields of `ObjectType`.

If the pointer is `undef`, then this will create a new region, calling the run-time function: `RegionType::create(alloc, Type)`, returning a `!iso<ObjectType>`.

Otherwise, it will allocate a new object on the current region, calling the run-time function: `RegionType(object)::alloc(alloc, Type)`, returning a `!mut<ObjectType>`.

After the allocation, this operator will initialise each field with the values from `%args` by lowering to `field(%obj, "name") = %arg[i]`.

## Control Flow

### Match

`match` is used to match the type of an object to a known type, returning a boolean, used in conditional branches.
```mlir
match(%obj: !join<Type1, Type2, ..., TypeN>, !Type) : i1
```

where:
 * The type of the `%obj` has to be a `!join<>` type.
 * The `!Type` to match has to be in the union above.

The operation returns `true` if the run-time type of `%obj` is `!Type`.

If know at compile time, `match(!concreteA, !concreteA)` can be simplified to `true` and `match(!concreteA, !concreteB)` can be simplified to `false`.

Example:
```mlir
  %obj = call @some_function(...): !join<A, B>
  %m = verona.match(%obj, !A) : i1
  cond_br %m, ^handle_a(%obj), ^handle_b(%obj)

^handle_a(%a: !A): // guaranteed to be !A
  ...

^handle_b(%b: !B): // guaranteed to be !B
  ...
```

### Throw

`throw` is used as non-local return, where the function is defined to throw `Throw[T]` types.
```mlir
throw(%obj: !Type): !throw<Type>
```

where:
 * `%obj` is a reference to the object being thrown.
 * `Type` is the type of that object.
 * The type returned is `!throw<Type>`.

The operation sets a flag before returning, so that `catch` can check that flag and know that the encapsulated type is a `!throw<T>` and needs to clear the flag.

This is a terminator operation, with the same semantics as a `return` in a basic block.

### Catch

`catch` checks for the exception flag, and if true, clears the flag and returns `true`, otherwise, it returns `false`.
This is used in conjunction with a conditional branch, guaranteeing the dispatch type.
```mlir
catch(%obj: !join<...>): i1
```

where:
 * `%obj` is the object returned or thrown.
 * The type of `%obj` is a union of types, with at least one being `!throw<T>`.

The operation returns `true` is the run-time type is a `!throw<T>` type, clearing the exception flag, otherwise, it returns `false`.

This is equivalent to a `match` that looks into throw vs no-throw types.

Example:
```mlir
  %obj = call @some_function(...): !join<A, B, Throw<E>, Throw<F>>
  %c = verona.catch(%obj): i1
  cond_br %c, ^handle_exception(%obj), ^continue(%obj)

^handle_exception(%exc: !join<E, F>): // flags was cleared above
  %e = match(%exc, !E): i1
  ...

^continue(%val: !join<A, B>): // no exceptions here
  %a = match(%val, !A): i1
  ...
```

### Using

`using` changes the current active region, ie. the one we can write to and where we create new objects.
```mlir
%ret = using(%ref = %obj: !join<iso<T>, mut<T>>)
{
  // Code here assumes the active region is the one where %obj is
}
```

where:
 * `%ret` is the (optional) return value of the region.
 * `%obj` is an object that is located in the region to be made active.
 * If the object is `!mut<>`, then we find the region it belongs to and make that the active region.
 * If the `%obj` is `undef`, then a new region has to be created.
 * `%ref` is the internal reference to `%obj` inside the region (isolated from above).

Example:
```mlir
  // Returns an immutable object from a temp region
  %immObj = verona.using(%region = undef) : !imm<Type>
  {
    %obj = verona.alloc(%region, !Type): !iso<Type>
    %imm = verona.freeze(%obj)
    ret %imm
  }
```

Alternatively, we could create two operations: `region_push()` and `region_pop()` to manipulate the stack of regions directly, but MLIR has isolated regions that already encode that logic, so it should be fine.

### Cast

`cast` just changes the type for the sake of the native MLIR type checker with no-op semantics.
```mlir
cast(%obj: !Type1): !Type2
```

Expressions where union and interface types are requested, their sub-types can be used, but MLIR doens't know the Verona type relationships.

This operation just changes the type on a new SSA variable, allowing the type checker to pass.

## Memory Management

These are inserted by the compiler when objects go out of scope or in special places that the compiler deems correct to insert.

`tidy` calls the garbage collector on the region, if the region is traced, or doesn't do anything otherwise:
```mlir
tidy(%x : !join<iso<Type>, mut<Type>) ->
RegionType::gc(alloc, x) || NOOP
```

`drop` marks the reference as dead and, if the reference was an `iso`, it also releases the region. If the region is reference-counted, also decrease the counter for `mut` references.
```mlir
drop(%x : !join<iso<Type>, mut<Type>) ->
Region::release(alloc, x)
```

## Region Handling

This may not have direct correlation with the language, but is needed in order to implement the run-time semantics correcty.

`move` inserts an object into a region by transfering the ownership of the new region.
```mlir
move(%x : !region, %z : !mut<Type>) ->
RegionType::insert<YesTransfer>(alloc, x, z)
```

`extref` inserts an external reference to an object in another region but not transfering the ownership to the new region.
```mlir
extref(%x : !region, %z : !mut<Type>) ->
RegionType::insert<NoTransfer>(alloc, x, z)
```

`merge` joins two regions into one, leaving the active region's sentinel as the entry point to the new region.
```mlir
merge(%x : !region, %z : !region) ->
RegionType::merge(alloc, x, z)
```

`freeze` makes all objects in the region `imm`, from the sentinel object down to all others.
```mlir
freeze(%x : !join<iso<Type>, mut<Type>>) ->
Freeze::apply(alloc, x)
```