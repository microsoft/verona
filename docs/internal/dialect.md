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

The new semantics only allows one region to be writable at any given point, so when creating new objects, we know in which region, if any, to allocate it.
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

If the region pointer is `undef`, then this will create a new region, calling the run-time function: `RegionType::create(alloc, Type)`, returning a `!iso<ObjectType>`.

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

Example:
```mlir
  %obj = call @some_function(...): !join<A, B>
  %m = verona.match(%obj, !A) : i1
  %a = verona.cast(%obj) : !A
  %b = verona.cast(%obj) : !B
  cond_br %m, ^handle_a(%a), ^handle_b(%b)

^handle_a(%a: !A): // guaranteed to be !A
  ...

^handle_b(%b: !B): // guaranteed to be !B
  ...
```

If know at compile time, `match(!concreteA, !concreteA)` can be simplified to `true` and `match(!concreteA, !concreteB)` can be simplified to `false`.

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
  %c = verona.catch(%obj): i1 // clears the flag
  %exc = verona.cast(%obj) : !join<E, F>
  %val = verona.cast(%obj) : !join<A, B>
  cond_br %c, ^handle_exception(%exc), ^continue(%val)

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

Expressions where union and interface types are requested, their sub-types can be used, but MLIR doesn't know the Verona type relationships.

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

This may not have direct correlation with the language, but is needed in order to implement the run-time semantics correctly.

`move` inserts an object into a region by transferring the ownership of the new region.
```mlir
move(%x : !region, %z : !mut<Type>) ->
RegionType::insert<YesTransfer>(alloc, x, z)
```

`extref` inserts an external reference to an object in another region but not transferring the ownership to the new region.
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

## IR Optimisation

The main reason for an MLIR dialect, rather than direct lowering to LLVM IR, is so we can do high-level transformations that would be costly or impossible in a low-level IR.

This section justifies the existence of the operations and types declared for what optimisations they allow.

### Types

#### Concrete Types

Concrete types are just LLVM pointers to structures which are allocated by the run-time using the `verona.alloc` operation.

Objects that have a limited lifetime (ex. not captured by a lambda or passed onto asynchronous execution) can be allocated in the stack instead.

The compiler can detect lifetime locally and change a `verona.alloc` to an `llvm.alloca` to avoid run-time calls, reference counting, garbage collection, etc.

#### Capabilities

With the type inference and final checks being done before MLIR, there is no need to keep reference capabilities at this level, other than potential optimisations.

For example, a `match` on a function that gets inlined, where the argument's type is statically known can help the compiler elide the remaining cases.

For example, this match:
```ts
func(arg: (imm | mut)): A | B
{
  match arg
    { i: imm => new A }
    { m: mut => new B }
}
```

Could be simplified if inlined in a function with a static type:
```ts
other(...): ...
{
  let x = A::create() // A & mut
  let y = func(x)     // B & mut
  ...
}
```

The compiler can look at a `match` and check if the type of the argument is known to be the same at compile time:
```mlir
@func(%arg: !join<imm, mut>): !join<A, B>
{
  %m = verona.match(%arg, !imm) : i1
  cond_br %m, imm(), mut()

^imm():
  %a = verona.alloca(...): !mut<A>
  ret %a

^mut():
  %b = verona.alloca(...): !mut<B>
  ret %b
}

@other(...): ...
{
  %x = call @A::create(...): !mut<A>
  %y = call @func(%x)
  ...
}
```

Inlining the call to `@func` would make the `match` line:
```mlir
  %x = call @A::create(...): !mut<A>
  %m = verona.match(%x, !imm) : i1
```

which allows the compiler to match the type of `%x` with `!imm` directly and substitute for `%m = false`, making the `cond_br` an unconditional branch, eliding the `^imm` basic block, simplifying `@other` to:
```mlir
@other(...): ...
{
  %x = call @A::create(...): !mut<A>
  %y = verona.alloca(...): !mut<B>
  ...
}
```

#### Union Types, Throw Types and Casts

This less for code optimisation per se, more for simplifying code generation.
But in conjunction with `match`, it can help similar optimisations as above.

We need to be able to represent union types in IR in some form, and encoding it in a way that allows us to inspect and extract the sub-types is helpful in some situations.

If a function has an argument with a union type and the caller passes one of the sub-types, we can introduce a `cast` but also check that the type is in the list of the union's sub-types.

Or, when a function return some types and throw others, the return type is a union of regular types and `Throw[]` types.
When using a `try/catch` on the caller, the `catch` must separate between returned and thrown objects, and also unpack the thrown objects into regular objects.

Having union types that can be inspected and throw types helps that.
The `cast` operation helps glue all of that together without needing a full blown type verification in MLIR.

For example:
```ts
factory(...): A | Throw[E]
{
  let a = new A(...)
  if (something wrong)
    throw E
  a
}

user(...): A
{
  try {
    let a = factory(...)
    a
  }
  catch
  {
    e: E => // handle error
  }
}
```

In MLIR would be:
```mlir
@factory(...): !join<A, throw<E>>
{
  %a = call @A::create(...)
  %err = // check something
  cond_br %err, error(), cont()

^error():
  %e = call @E::create(...)
  verona.throw %e // terminator

^cont():
  ret %a
}

@user(...): !A
{
  %a = call @factory(...): !join<A, throw<E>>
  %threw = verona.catch(%a): i1
  // Casts only help glue the BB args below
  %actual_a = verona.cast(%a): !A
  %throw_e = verona.cast(%a): !E // not !throw<E>
  cond_br %threw, error(%throw_e), cont(%actual_a)

^error(%throw_e):
  // handle error
  terminator || unreachable

^cont(%actual_a):
  ret %a
}
```

Note the use of `cast` on `%a` for both types, even if just one would ever match. This is possible because casts are no-ops and will be lowered as either some form of `bitcast` or elided completely.

Also note that the thrown cast is to `!E` and not `!throw<E>`.
That is because `catch` has already cleared the exception flag, and now all exception objects are just regular objects.

If `@factory` could throw more than one exception, the cast would be to another union containing only `throw` types, and further `match` operations on the `^error` basic block would differentiate which type of exception

Example:
```mlir
@factory(...): !join<A, B, throw<E>, throw<F>>
{
  ...
}

@user(...): !join<A, B>
{
  %a = call @factory(...): !join<A, B, throw<E>, throw<F>>
  %threw = verona.catch(%a): i1
  %values = verona.cast(%a): !join<A, B>     // Note, only "regular" objects here
  %exceptions = verona.cast(%a): !join<E, F> // Note, only "exception" objects here
  cond_br %threw, error(%exceptions), cont(%values)
  ...
}
```

If the error checks depend on the arguments and `@factory` gets inlined, then the compiler can verify if nothing is ever thrown, so the `catch` will always be false and the `^error` basic block will never be reached, thus eliding not only a lot of code, but also run-time checks and branches.

Example:
```ts
factory(a: I32): A | Throw[E]
{
  if (a > 10)            // When inlined, this condition is always false
    throw E
  let obj = A::create(a)
  obj
}

user(...): A
{
  let a = 5
  try {
    let obj = factory(a) // clearly, a < 10
    obj
  }
  catch
  {
    ...
  }
}
```

When inlining, the `throw` basic block would be elided, and the compiler can verify that nothing in the current context can throw (that has not been handled yet) by checking `verona.throw` and `verona.catch` pairs and replace the last call of `verona.catch` with `false`, making the following conditional branch, unconditional, thus eliminating all exception handling basic blocks.