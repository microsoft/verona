# Reference Capabilities

## WARNING

This is a preliminary document under heavy discussion.
The ideas here are mostly right, the details, not necessarily.
There will be a tutorial document that will be correct in all counts, but this isn't it.

## Overview

Reference [capabilities](https://en.wikipedia.org/wiki/Capability-based_security) are unforgeable tokens of authority that confer the rights to access something.
They can be delegated to other components of a system and must always respect the *principle of intentionality*: you must present a capability that confers the rights to perform an action when attempting to perform that action, it is not sufficient to simply own the capability.

In Verona, pointers are capabilities: the ability to name an object confers the right to access the object in some way.
Capabilities confer a range of different permissions, which are reflected in the type system.

Variables represent pointers to objects and they must have rights associated that describes what you can do to the objects via those pointers.

There are kinds of rights that can be granted to capabilities:
* Isolated (`iso`).
  Indicates that a capability is *the globally unique* pointer to its target.
  Every `iso` capability is the entry point to a region.
  Whichever running behaviour holds an `iso` reference to a region has the guarantee that no other running behaviour has any access to that region.
  To assign an `iso` reference to another variable *moves* it, removing it from the source and transfering the ownership of the target region.
* Mutable (`mut`).
  Write access to data that is guaranteed to be dominated by an `iso` region.
  Behaviours can freely write to `mut` pointers because they have the guarantee that they can only access them if they hold the corresponding `iso` pointer to their region.
  Every `mut` pointer is associated with an `iso` pointer whose existence guarantees that the region is safe to access.
  A `mut` pointer is invalidated once the type system can no longer guarantee that the associated `iso` pointer remains valid.
* Read-only (`readonly`).
  Behaviour-internal references to objects providing local immutability (ie. restrict write access) through this reference.
  If the object is mutable then other `mut` pointers in the same behaviour may still modify the object.
  When accessing an object via a `readonly` pointer, any read of a field declared `mut` or `iso` will return a `readonly` pointer: you may not use any pointer derived from a `readonly` pointer to modify any reachable object.
* Immutable (`imm`).
  Global immutability, guarantees that no behaviour will be able to write through references to the same objects.
  Isolated regions can be entirely frozen into immutable regions.
  Mutable references are frozen together with the region but can't themselves be frozen directly.
  After freezing, only `imm` pointers can be created to any object from the region that was frozen.
  If the frozen region contained objects with `iso` fields referring to child regions then the child regions are frozen at the same time.
  When accessing an object via an `imm` pointer, any read of a field declared `mut` or `iso` will return an `imm` pointer: after freezing there is no distinction between objects that were allocated in the same or different regions.
  `imm` pointers can be shared across behaviours because their globally immutable nature guarantees data-race freedom.

## Regions

Asynchronous tasks in Verona are performed by behaviours (`when`) on cowns, which own a tree of regions.
The language semantics guarantee (via `iso` pointers) that, at any time, only a single behaviour has access to a region.

Regions are portions of (not necessarily contiguous) memory dominated by an isolated (`iso`) pointer, with any number of mutable (`mut`) pointers in the same region, accessible only through the `iso` pointer.
Regions can be implemented with different memory management strategies or execution models.

## Assignment

Capabilities change the semantics of assignments in the language.

**Sentinel objects** in a region may have only a single pointer to them from outside of the region and that must be an `iso` pointer.
The uniqueness guarantee of the `iso` pointer means that any read of an `iso` value that gives an `iso` capability is destructive.
Verona pointers are not nullable and so an `iso` field must always hold a valid `iso` capability.
Reading an `iso` field can produce a `mut` pointer bound to that `iso` but extracting the `iso` capability requires a swap operation that installs a new value at the same time as removing the old one.
For stack variables, extracting the `iso` capability simply ends the scope of the variable: the variable no longer exists after the read, just as if the block in which it is declared had ended.

```ts
var foo : Foo & iso = Foo; // plus undecided syntax for "in new region type X"
foo.doSomething();
...
var bar = foo; // bar now holds the *only* reference to foo
bar.somethingElse(); // OK: bar is a reference to the same object as foo
foo.cleanup(); // ERROR: foo is not a valid reference any more
```

Verona does not allow uninitialised values to be created or read, there is no such thing as a `nullptr`.
Objects must be created with all values initialised and cannot lose their values as a result of an `iso` move semantics.
Therefore, the example above would be invalid if `foo` was an object's field.

```ts
class Something { foo : Foo & iso; ... }
...
let some = Something; // Initialises `foo` with an `iso` reference
...
var bar = some.foo; // ERROR: This would `move` the `iso` into `bar` and let `foo` with an invalid value
...
doSomething(some); // What if this function does something with `foo`?
```

**Objects within a region** can hold `mut` pointers to each other, including aliased pointers to the same object.
Reading a `mut` field creates a copy of the capability, which can be stored on the stack for as long as the type system can guarantee that the `iso` capability conveying the right to access the region is live and which may be stored into any suitably typed field of any object in the same region.
Given that mutable references can be modified only by the one behaviour that holds their dominating `iso` reference, there can be no data races in any write or reassignment.

```ts
var foo : Foo & mut = Foo::create(10); // Returns a Foo object with value = 10
print(foo.value); // prints "10"
...
// Binding a new variable to the same reference creates a new reference to the same object
var bar = foo;
bar.value = 42;
print(foo.value); // prints "42"
...
// Creating a new object reassigns the variable bar but doesn't change the previous references (foo)
bar = Foo::create(31); // Returns a new Foo object with value = 31, binds it to bar
print(bar.value); // prints "31"
// But foo still points to the previous object
print(foo.value); // still prints "42"
```

**Read-only references** can be created from other capabilities, but cannot be upgraded to any kind of capability that permits writes.

This is to protect passing a read-only reference and the user change it back to a mutable one.
```ts
var foo : Foo & mut = Foo::create(10); // Returns a Foo object with value = 10
print(foo.value); // prints "10"
...
var bar : Foo & readonly = foo; // OK: a read-only reference to a mutable object
...
var baz : Foo & mut = bar; // ERROR: cannot convert readonly to mutable
```

**Immutable capabilities enforce global immutability**.
The `freeze` operation consumes an `iso` capability.
The type system therefore guarantees that there can be no pointers to objects in the region (or any child regions) anywhere other than as pointers within the regions being frozen.
The `freeze` operation returns an `imm` capability and *viewpoint adaptation* guarantees that this cannot be used to materialise any kind of mutable capability.
Once a region has been frozen, no capabilities exist that permit writes to any objects that were in that region.

```ts
var foo : Foo & imm = freeze(Foo::create(10));
print(foo.value); // prints "10"
...
var bar = foo;
print(bar.value); // prints "10"
...
var ro : Foo & readonly = bar; // OK: imm is read-only anyway, so a noop
var imm : Foo & imm = ro; // OK? also a noop but there's a sub-type relationship that I'm not sure is there
```

### Destructive reads

Some reads invalidate the previous reference on assignment.
By design, `iso` references have a destructive read, since the program cannot have two `iso` references to the same region.
But behaviour captures also destroy the previous reference to avoid data races.

Example:
```ts
var foo : Foo & mut = Foo;
// Creates an anonymous asynchronous behaviour
// Captures foo into a field of the lambda, say "_field0".
// That capture is a destructive read.
when ()
{
    // Uses foo here, which in the lambda is the same as _field0.doSomething();
    // Because the original foo is no longer valid.
    foo.doSomething();
} // Schedules the behaviour and return -before- the execution finishes
...
// If this was allowed, there could be a race condition with _field0
foo.somethingElse(); // ERROR: foo was captured into a new reference and this one is now invalid
```

## Capability Conversions

Verona does not permit implicit type conversions and that also applies to capabilities.
Using a variable with one capability in place of another will not coerce the compiler to convert capabilities, unless there is a clear sub-type relationship or conversion function (ex. `freeze`).

### Sub-typing

Sub-typing is the type relationship where a sub-type can be safely used in place of its super type, for example, as function arguments or return values.
Sub-types can, therefore, be passed as arguments or directly assigned to variables that have been declared as their super-types.

Example:
```ts
interface Super {}
class Sub : Super {}
function(arg : Super) { ... }
...
a : Sub = Sub;
function(a); // OK, as Sub is a sub-type of Super
```

The notation is `Sub <: Super` that means `Sub` is a sub-type if `Super`.

### Allowed Conversions

These are the following capability sub-type relationships in Verona:
* `iso <: mut`: An `iso` capability grants the rights to access a sentinel object and to create `mut` capabilities to any object in that region (i.e. that are reachable by following only `mut` pointers from the sentinel object) that are valid for as long as you hold the `iso` capability.
  As such, an `iso` capability trivially authorises constructing a `mut` capability to the object (this is the degenerate case: follow 0 `mut` pointers from the sentinel).
  As with any other `mut` capability, this may be stored only in places where the `iso` capability can be statically proven to be held: in fields of objects in the same region or in lexically-scoped variables dominated by a variable holding the `iso`.
  For example, if one stack frame holds an `iso` capability conferring exclusive ownership of a region then a function called from that stack frame may take a `mut` view of the sentinel object as an argument.
* `mut <: readonly`: Passing a `mut` in a place expecting a `readonly` will _view_ the type as read-only and not allow writes to it.
  The `readonly` capability has the same lifetime restrictions as the `mut` capability from which it was derived.
* `imm <: readonly`:  An `imm` capability guarantees global immutability, `readonly` guarantees local immutability which is a strictly weaker property that is trivially satisfied by the stronger property.

Capability systems require that rights associated with a capability are never elevated except by interacting with a component that already owns these rights.
Any change to a capability involves an upgrade of rights or exchanging the existing rights for a mutually exclusive set then this must be an explicit operation.

* `iso -> imm` via `freeze`: Converts an `iso` region into an `imm` region by freezing it.
The region will become globally immutable and cannot be unfrozen later, to guarantee data-race freedom.
All `mut` references inside the region will also become `imm` and any writes, after being frozen, to the region through those references will be invalid.

Example:
```ts
var foo : Foo & iso = Foo; // "in a region type X"
var bar : Bar & mut = Bar in foo; // new object in the same region
...
var bar : Foo & mut = foo; // OK: creates a mutable ref to foo, doesn't invalidate foo
...
function1(arg : Foo & mut) { ... }
function1(foo); // OK: iso <: mut, doesn't invalidate foo
...
var foo_imm : Foo & imm = foo; // ERROR: needs to freeze the whole region
var bar_imm : Bar & imm = bar; // ERROR: bar is not an iso
...
function2(arg : Foo & imm) { ... }
function2(foo); // ERROR: needs to freeze the whole region
...
var frozen : Foo & imm = freeze(foo); // OK: now everything is frozen
...
function1(foo); // ERROR: foo has been invalidated by freeze
function1(frozen); // ERROR: function1 expects mut
function2(frozen); // OK: function2 expects imm, all good
...
write_to(arg : Foo & mut) { ... }
write_to(bar); // ERROR: bar's region (foo) has been frozen
```

## Capability Matrix

Each capability denies certain permissions and therefore can only be aliased by references that deny the same or less permissions than itself.

For example, an `iso` reference denies the existence of any global read-write aliases, as well as other local owning (`iso`) references, but it allows local read-write (`mut`) aliases without invalidating (moving) the `iso` reference.
This is possible because `iso` is strictly stronger than `mut` and because `mut` references are only valid locally (they cannot be shared across behaviours).

On the other hand, `imm` references deny any local and global write (`mut`, `iso`) aliases, but allow any local and global read (`imm`) aliases.
This is possible because `imm` objects are guaranteed not to change globally.

The only capability that allows global aliases of any kind is `imm` because it is always safe to share immutable objects.

The following table describes what each capability denies, locally and globally.

| Deny | global read-write alias | global write alias
| ---: | :---: | :---: | 
| local owning alias | iso | 
| local write alias | readonly | imm
| local immutable alias | iso, mut
