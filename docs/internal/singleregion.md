# Single active region types

Verona has a concept of regions as its core way of reasoning about ownership.
There are multiple systems that could be developed to enable this to work soundly.
In this document, we develop having a single active region that may be mutated at any point in time.

This document is a working document and should not be consider definitive.  It represents the best current view of how the type system should work.

## Informal

Verona is a structurally type and algebraic programming language.  It is an object based language, where the only values are references to objects.  There are some primitive types that the runtime represents more efficiently, but abstractly everything can be viewed as an object.

Verona's core concept for ownership is regions.  A region is a group of objects.
There are two types of region in Verona, mutable and immutable.  There is a single immutable region.  The immutable region is closed under flowing fields from objects in that region.
```
  ∀ x,f. region_of(x) = immutable ⇒ region_of(x.f) = immutable
```
There can be multiple mutable regions.  There is a single object in a mutable region that is the entry point. There is a single reference to the entry point from outside the region.  There may be multiple references from within the region to any other object, including the entry point.
```
  ∀ ref1,ref2.
    ref1.dst = ref2.dst ∧  region_of(ref1.dst) ≠ immutable ⇒
      region_of(ref1.src) = region_of(ref1.dst) ∨
      region_of(ref2.src) = region_of(ref1.dst) ∨
      ref1.src == ref2.src
```
Here, we use a generic concept of reference, where a reference has a `src` and `dst`.  Examples for srcs of references are a stack locations for variables, fields in objects, and captures in closures.  The `dst` is always an object.  This generalisation to references is required to ensure there is only a single entry point from either the stack or the heap.

[TODO: This explanation falls short when we get to `using`. Well, we at least need to carefully consider how `region_of` interacts with variable scopes.]

The overall topology is a forest of mutable regions, which can all reference the immutable region.

[PICTURE OF HIGH-LEVEL TOPOLOGY HERE]

Importantly, the references between mutable regions must be treated linearly to preserve that a single reference into a region can exist.

[TODO:cown]

### Region capabilities

To ensure the topology above is both preserved, and useful to the programmer, we expose capabilities over regions in the language that can be used to annotate any reference.  They form part of the type of the reference, which we attached to the src of the reference, i.e. fields and variables.

In Verona we use the following three capability types:

* `mut` is the type representing references within the same region,
* `imm` is the type representing references where the destination is in the immutable region, and
* `iso` is the type representing references between different mutable regions.

[TODO: Does this explanation work for stack/`using`?]

We use `&` to mean the intersection of two types, that is `x: t1 & t1` means `x` satisfies both `t1` and `t2`.
To illustrate this with a concrete data-structure, consider the following class definition:
```
class Entry
{
  key:   imm & K
  value: iso & V;
  next:  mut & Entry;
  prev:  mut & Entry;
  ...
}
```
It has four fields.  The first, `key`, references an object in the immutable region that is a subtype of `K`.
The second, `value`, references an object that is an entry point into a different region that is a subtype of `V`.  The final two fields are references in the same region to objects of the `Entry` type.

Here you can see the flexibility we get.  Each `value` in the list is in its own region, but the structure of the list can have an arbitrary structure.  In this instance, given the type, you might expect a doubly-linked list.  As the `key`s are in the immutable region, they can be shared and referenced from any region.

[PICTURE OF DLL WITH REGIONS]

The gives the basic topology that the type system can impose on the structure of memory.

### Accessing regions

Now we will consider how we can access and manipulate the structure of objects, while preserving the topology that is imposed on memory.
In this section, we will informally walk through the typing of various accesses to memory, using the `Entry` class from the previous section.

We consider accessing various fields of `Entry` with different starting capabilities.

#### Immutable

If we access immutable fields of a mutable object, then we get the immutable capability on the target:
```
// x: mut & Entry
let k = x.k
// k: imm & K
```
Now, if we access any field on an object with an immutable capability, then all the accessed are allowed, but must also be immutable capabilities:
```
// i: imm & Entry
let k = x.key
// k: imm & K
let v = x.value
// v: imm & T
let n = x.next
// n: imm & Entry
```

#### Mutable

If we access a mutable object's mutable field, then we get a mutable capability on the target:
```
// x: mut & Entry
let y = x.next
// y: mut & Entry
```

Let us consider mutating a pointer on this class. In Verona, we give assignment the semantics of 'exchange':
```
  e = e'
```
This evaluates `e` to a storage location, and `e'` to a value, and then assigns the value into the storage location.
The result of this expression is the old value of the storage location.
```
// Assume x contains the value 5
 y = (x = 6);
// Assert y contains the value 5, and x contains the value 6.
```
This semantics is a more natural fit for a language with ownership as a core concept, as it allows you to express moving values.

Now returning to our `Entry` class.  Consider a simple pointer swing:
```
// prev: mut & Entry
// new_entry: mut & Entry
let old_n = prev.next = new_entry;
```
This is allowed, as the destination object `prev` has a `mut` capability, and the value being stored is of the type of the field.
We cannot mutate the fields of immutable objects.


#### Isolated transfer

Manipulating references with an `iso` capability is more challenging as it must preserve the topology of being the only external reference to the region.
First, let us consider unlinking an `iso`:
```
// x: mut & Entry,  new_v: iso & V
let old_v = x.value = new_v
// x: mut & Entry,  new_v: undefined,  old_v: iso & V
```
This code assigns a new value into the `value` field.
Due to the semantics of assignment, this returns the previous value contained in the field.
Hence, we have a local variable that is bound to the entry point of the original region.
The original reference to the new value now has the `undefined` type, to mean it cannot be used any more.
We have treated the `iso` linearly, that is the assignment consumed the new value, and produced the old value.
This is the core of ownership transfer in Verona.

For exposition purposes, we use `drop` to signify the region will no longer be used by this part of the execution.
```
// old_v: iso & V
drop old_v
// old_v: undefined
```
It could be sent using the concurrency mechanisms of Verona, but we will return to that later, or in another document, or it could be deallocated.  For the rest of the document, it suffices to consider only deallocation.


#### Isolated access

So far, we have seen that we can manipulate references to isolated regions, but not how we can actually access the state associated with them.
In Verona, the only way to 'use' an isolated region is with a `using` statement:
```
using e { x => e' }
```
This means evaluate `e` to a region entry point, bind that entry point to `x` and then execute the code `e'`.  During the execution of `e'` the region that `e` evaluated is considered active
and all other regions are not mutable.
For instance, consider the following use with our running example
```
// x: mut & Entry
using x.value {
  y =>
  // y: mut & V
  // x is no longer considered mutable here.
  ...
}
```

Prevent mutable capabilities to the context is essential for two key reasons:

* Not allowing cross region pointers
* Preventing deallocation of currently referenced regions

The following example illustrates both of these issues.  Consider that the type `V` is actually `Entry`, then the following could occur:
```
// x: mut & Entry
using x.value {
  y =>
  x.next = y; // This should not be allowed
}
let old_v = x.value = new_v
drop old_v
// x.next is now a dangling pointer.
```
This example creates a link from a parent region into its child region.
It then deallocates the child region, and thus leaves a dangling pointer.
The core issues in this example is that `x` should not be viewed as `mut` inside the `using`.
The capability `mut` means a reference to the currently 'active' region, but `using` is explicitly changing the currently active region.

To handle, this we could take two approaches:
* Do not allow any context to come inside a `using`; or 
* Introduce a new capability for allowing read-only and uncapturable access to the context.

We will take the second approach and introduce a new capability `paused`, to represent region that are currently being 'used', but may not be the 'current' region. So when we pass inside a
`using` block, we have to change all `mut` from the context, to `paused` capabilities.
```
// x: mut & Entry
using x.value {
  y =>
  // x: paused & Entry
  x.next = y; // This does not type check, cannot assign a `paused` object
  y.next = x; // This does not type check, x is not `mut` as required.
}
```

### Semantics

[TODO - Overview of semantics]

An object is an index set of storage locations, and each storage location contains a reference to a value.  The type of a storage location specifies properties of what the storage locations refers to.

```
oid ⇀ (field ⇀ storage_location) × class
storage_location ⇀ value
```

In what follows, we consider `value` to be `oid`.

[TODO: Nested storage locations?]

### Variables and `store[T]`

[TODO: This doesn't account for mutable local variables so far.]
[TODO: Need locals of type `store[T]` to account for taking passing lvalues.]

```
storage_location ⇀ (value ∪ storage_location)
```
This allows the semantics to use `store[T]` as the type of a reference, i.e.
```
x: store[T] & mut
```
is a reference from the storage location `x` to a storage location containing references satisfying `T`.

## Formal

[TODO: Formal type rules will be part of a subsequent PR.]