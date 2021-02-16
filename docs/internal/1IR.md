# 1-IR

This document describes the first IR used by the Verona compiler. This IR is
constructed from the AST, very early in the compiler pipeline. At the time of
its construction, no type information is available yet. Therefore the
operations used map directly to surface syntax concepts.

## General structure

The IR has a fairly standard structure, made up of basic blocks. Each basic
block has a list of statement and a terminator. While the implementation
occasionally refers to "phi nodes", the IR uses basic block arguments rather
than phi nodes.

The IR only represents the body of methods. The "shape" of the program, eg. its
classes, fields, method names and signatures, ... are all stored in the AST
instead. For each `Method` node in the AST, a corresponding `FnAnalysis` is
created, which, among others, contains the `MethodIR`.

A `MethodIR` is composed of one or more `FunctionIR`. This distinction is used
to encode the bodies of `when` clauses that appear within the method. The first
`FunctionIR` corresponds to the method body itself, whereas subsequent blocks
correspond to when block bodies.

Once created by the `ir/builder` pass, the IR is immutable. All new information
is represented in external side structures, as part of the `FnAnalysis`
structure.

## Values

We call "value" the contents of a program variable or field. Note that this
differs from MLIR, which calls `Value` SSA variables.

Values are generally divided in a handful of categories:

- *Owned references* hold a pointer to a region's entry object. For a given
  region, there can be only one owned reference to it. This is represented by
  the capability `iso` in the type system.

- *Unowned references* hold a pointer to any object inside a region, which may
  or may not be the region's entrypoint. If a field contains an unowned
  reference, then that reference must point to an object in the same region as
  the object which contains the field. This is represented by the capability
  `mut` in the type system.

- *Reference counted values* hold a pointer to either an immutable object or a
  cown. While the API used to manipulate the two differ, at the IR level, they
  have very similar semantics. This is represented by the capability `imm` in
  the type system.

- *Primitives* are values that do not reference the heap at all, such as
  integers. This is also represented by the capability `imm` in the type
  system, despite not being references to any kind of object. Strings are
  currently represented as primitives, although this will change in the future
  (they will become regular objects).

Many IR operations have different behaviours depending on the kind of value:
for instance, if a variable `x` holds an owned reference when it goes out of
scope, the referenced region must be deallocated. If instead it held an unowned
reference, nothing needs to be done.

While type information may help narrow down the set of possible values a
variable or field holds, it is, on its own, not sufficient. For instance, a
variable could have type `x: iso | mut`, in which case it could hold either an
owned or unowned reference. The underlying implementation must therefore have a
mechanism for deciding the kind of value. In the VM, this is achieved very
wastefully by including a tag in every value. In practice, we'll want to combine
type information and more specialized encodings to achieve this efficiently.

## Type information

At the time of its construction, very little type information is known about
the program. Therefore the IR itself contains almost no type information. Only
types which appear in the method body at the source level is included in the
IR. At the moment, this is just types used in a match block.

During type inference, a type is computed for every IR variable. These types
are stored in maps in `FnAnalysis`. The type of `v` in basic block `bb` can be
found in `analysis->typecheck->types[bb][v]`.

Similarily, at construction, every method call site is assigned a
`TypeArgumentsId`, used to identify the call's type arguments. Given an `id`,
the inferred type arguments can be found in
`analysis->typecheck->type_arguments[id]`.

In both cases however, these are the polymorphic types. If a method or class is
generic, these types may mention type variables. The non-generic types will
only be known after reachabilitiy and reification.

## IR Invariants

*Note: this section mixes properties that are guaranteed by construction, and
properties that should be enforced by the type system. A violation of the
former is compiler bug, a violation of the latter is a programmer error. From
the perspective of lowering, this is irrelevant since we assume only correct
programs are lowered.*

In addition to the typical requirements imposed by SSA, the IR has strong
invariants about how and when variables may be used. We split uses of variables
into different categories, described below.

### Regular uses

Regular uses are those which inspect the contents of a variables. This includes
field reads, the origin of a field assignement (ie. `x` in `x.f = y`), pattern
matching, ...

### Consuming uses

Consuming uses are a subset of regular uses, in which, if the variable held an
owned reference, that value is *moved* to another location. In this situation,
the original variable's must not be the subject of a regular use ever again.

Consuming uses include the right hand side of a field assignement (ie. the `y`
in `x.f = y`), arguments to method calls, a copy from one variable to another
(ie. the `y` in `x = y`).

### Destructive uses

A destructive use is one that releases the contents of the variable, if it
hadn't been moved, by eg.  deallocating the region pointed at by an owned
reference. Destructive uses occur when a source level variable goes out of
scope, or when it is being overwritten. For example, `{ var x = f(); x = g();
var y = h(); }` lowers to an IR of the form:

```
  %x0 = call f()
  %x1 = call g()
  overwrite(%x0) // Destructive use of %x0
  %y = call h()
  end-scope(%y, %x1)  // Destructive use of %x1 and %y
```

Every variable must encounter exactly one destructive use on every execution
path. Conversely, after a destructive use, a variable must not be used again.
This requirement applies regardless of the kind of value held by the variable:
even an integer value must be the subject of a destructive use, albeit this
will lower to a no-op.

A destructive use is not considered a regular use. Therefore, a variable may
(and must) be destroyed after a consuming use. This implies that `%x = ...; %y
= call f(%x)` on its own is not a valid IR. It should instead have the form `%x
= ...; %y = call f(%x); end-scope(%x)`. If `%x` holds an owned value, the
consuming use in the call to `f` will have moved the value. The `end-scope`
statement is therefore a no-op.

### Forwarding uses

When a variable is passed as an argument to another basic block, we should
treat this use differently based on how the target basic block uses its
parameter. For this reason, we consider arguments passed to basic blocks
**forwarding uses**.

Consider the following IR:
```
BB0:
  %x0 = ...
  %y0 = ...
  goto BB1(%x0, %y0)

BB1(%x1, %y1):
  use(%1)  // Regular use of %1
  end-scope(%x1, %y1)
```

In this example, `%x1` is the subject of a regular use, whereas `%y1` is not.
Therefore, due to the forwarding nature of the `goto` statement, `%x0` is the
subject of a regular use but `%y0` is not.

Because all variables must eventually be the subject of a destructive use, here
the `end-scope(%x1, %y1)` statement, forwarding uses are themselves always
considered destructive uses. This has an important consequence: if a variable
is passed as basic block argument, it may not be used again. In the example
above, it would be illegal to use `%x0` or `%y0` in `BB1`. Additionally, such a
variable does not need any additional destructive use: note the absence of a
`end-scope(%x0, %y0)`.

TODO: we actually use the word "destructive use" for two things, `(overwrite +
end-scope)` and `(overwrite + end-scope + block arguments)`. The latter
includes forwarding uses, since these "act like" a destructive use in many
aspects.

TODO: Should variable uses be evaluated as a least or greatest fixed point?. In
other words, in `BB0(%x): goto BB0(%x)`, is `%x` ever the subject of a regular,
consuming or destructive use?  While an interesting theoretical question, I
don't think we need to worry too much about this in practice.

## Conditionally moved values

As mentioned earlier, if a value is moved, its destructive use has no effect.
Determining at the destructive use site whether or not to release a value may
require tracking additional information at runtime, in at least two circumstances.

Firstly, a consuming use may have occurred inside a branch. The value would
have been moved only if the branch is taken. In this case, the final
`end-scope` effect should only release the value was not taken.

```
BB0:
  %x = ...
  %y = ... // %y holds an owned value
  if %x then goto BB1 else goto BB2

BB1:
  call foo(%y) // Consuming use of %y, its value is moved
  goto BB2

BB2:
  // %y has been moved conditionally, we need to release it only if the
  // branch was not taken.
  end-scope(%y)
```

Secondly, a consuming use moves the value **only if** the value is an owned
reference. If a variable `%x: iso | imm` may contain either an owned reference
or a reference counted value, a call to `foo(%x)` would move the value only in
the first case, not if the value is a reference counted value. Therefore, a
subsequent `end-scope(%x)` statement should release the value only if it holds
a reference counted value, not an owned reference.

Note that in both cases, the conditionally moved value may not be the subject
of a regular use. However, destructive uses are not considered regular uses.

In order to correctly implement these semantics, we need to add special markers
during lowering to track whether a variable's contents have been moved or not.
The markers can be optimized away in cases where we know the answer statically.

## Statements
### `NewStmt`

Allocates a new object of a given class. It may optionally be given a `parent`
which designates the region in which the object should be allocated. If no
parent is specified, the object is allocated in a new region.

This will correspond to either a `rt::RegionTrace::create` or
`rt::RegionTrace::alloc` call, depending on whether a new region needs to be
allocated or not.

The `parent` specified in the statement could be any object in the target
region; it may not be a reference to the region's entrypoint. On the other
hand, the runtime's `RegionTrace::alloc` method expects to be given the
region's entrypoint. The compiler therefore needs to do some work to find the
entrypoint given the specified parent.

### `CallStmt`

Call a method, on the specified receiver. The method is identified by its name.
At compile-time, the method's selector should be obtained from the given name.

In order to dispatch the call, in the general case, the selector should be used
to index into the object's vtable, to retrieve the address of the target.
Optimizations may enable us to dispatch the call statically, using for example,
the type of the receiver.

The method being called could be static or not (ie. does it accept a `self`
parameter, not to be confused with the dispatch strategy). In either case, the
receiver of the call is always passed as the first argument to the method.  In
the case of a static method call, the receiver may not be an actual object, but
only a pointer to a descriptor, as created by a `StaticTypeStmt`.

### `WhenStmt`

Schedule a behaviour on a set of cown. The `closure_index` corresponds to the
index of the body within the `MethodIR`.

This corresponds to a `rt::Cown::schedule` call. The message sent to the cowns
should provide storage for a pointer to the closure's body, the list of cowns
and the captures.

The closure's body will accept as many parameters as there are cowns and
captures specified. When invoking the body, each cown argument should be
transformed into a reference to its contents, before being passed as an
argument. The captures should be passed as is.

### `ReadFieldStmt`

Read a field from an object. The field is identified by its name. The general
case is similar to method calls: the field's selector is used to look up the
field's offset in the object's vtable.

### `WriteFieldStmt`

Write a field into an object. The procedure for locating the offset of the
field is identical to a field read.

When writing a reference counted value to an object in a region that is
equipped with a RememberedSet, the value should be inserted into the RS.
Otherwise, the reference count should be incremented.

### `IntegerLiteralStmt`, `StringLiteralStmt`, `UnitStmt`

These create values from a literal. The support for strings is very special
cased for demonstration purposes.

### `MatchBindStmt`

This is an unchecked cast, used at the top of the target of a match statement.

### `ViewStmt`

If the argument to the statement is an owned value, the statement produces an
unowned reference to the same object. Otherwise this is equivalent to a copy.

### `EndScopeStmt`

This statement is inserted whenever variables go out of scope. This may not be
the end of a BasicBlock, if a block expression is used for example.

This is where a value's "release" code should be executed. This could be
`rt::Region::release`, `rt::Cown::release` or `rt::Immutable::release`, for
owned, cown and immutable values, respectively. Unowned values do not need to
be released.

### `OverwriteStmt`

This is used after a local variable assignement, on the old SSA value bound to
this statement. The old value should be released, as in an `EndScopeStmt`. For
example, the code `var x = new A; x = new B; ...`, would be lowered to the
following IR:

```
0 <- new A
1 <- new B
overwrite(0)
// ...
end-scope(1)
```

The local `x` is originally assigned the SSA variable `0`, and later `1`. After
the assignement, the `overwrite(0)` statement encodes the fact that the old
value needs to be released.

Note that an SSA value that has been overwritten is never passed to an
"end-scope" statement. Only `x`'s new value, `1`, goes out of scope.

## Terminators

TODO: Describe terminators (Branch, If, Match)

## Builtins

A number of language features don't need special syntax, but are instead
exposed as builtin methods, defined in `stdlib/builtin.verona`. This includes
arithmetic operations and some region operations such as freeze and trace.

The language features they represent isn't even expressible in the IR. Instead,
`codegen/builtins.cc` provides hardcoded lowerings into bytecode.

