# Verona ABI

## WARNING

This is a preliminary document under heavy discussion.
The ideas here are mostly right, the details, not necessarily.
There will be a tutorial document that will be correct in all counts, but this isn't it.

## Overview

This document describes the representation of objects in Verona.
Objects can be of various types, including numeric (`U32`, `F64`, `Bool`), pointers to interfaces or classes, type unions, etc.
Depending on the context, objects can have a different representation (ex. inside unions).
Objects can also be passed to functions or inserted into arrays, and this ABI defines how that works so that readers and writers, callers and callees all agree on the same representation.

The following ABI proposal is under construction and will change over time.
The public ABI (when interfacing with foreign languages or exported symbols) will be stable at some point and guaranteed by the compiler.
But the compiler reserves the right to change the internal ABI at any time, where it can guarantee the changes are not visible from user code.

## Memory Model

Verona's memory model is based on isolated regions.
References to sentinel objects (`iso`) point to the entry point of the region, while mutable references (`mut`) point to objects inside the region.
Immutable references (`imm`) point to immutable objects outside of any region.
The Verona runtime guarantees no more than one behaviour has access to the same region at any time, avoiding concurrent mutation problems.

### Regions

A region is a tree of related objects, not necessarily contiguous in memory.
Region objects can be placed anywhere in the heap and will depend on the memory allocator design.
The reason for grouping objects in the same region is so they can be accessed together by the same behaviour.
However, some regions (ex. sandboxes) do require memory to be contiguous (for range access protection).

Objects have their sizes known statically and are allocated directly and assigned to a region.
The Verona compiler guarantees that one behaviour cannot write to another region's memory by not having pointer arithmetic and checking provenance at compile time.
However, dynamic arrays can still have dynamic out-of-bounds access, which should trigger a run time error.
Dynamic arrays can grow their storage size, to add more elements at run time, but each element still has a static size.

### Graphs, Trees and Forests

Internally, regions form a graph of (`mut`, `readonly`) pointers between objects, dominated by a sentinel (`iso`) pointer.

Regions can also point to other regions (via their sentinel objects), recursively.
Sentinel objects can only have a single owning reference, so when another region acquires the owning reference, it is moved from the previous region to the new one.
No two regions can point to the same sentinel object, forming a tree of regions (not a graph).

`cown`s isolate a region (and its sub-regions) to be used by a behaviour.
They can also be reserved together for behaviours that use more than one `cown`.
The set of `cown`s defines a _forest_ of regions.

When regions are frozen (made immutable), their references move outside of the tree, so any behavior can read its objects concurrently and no updates are possible.
After being frozen, the immutable tree allows any other region to point to its members directly.
For that reason, Verona doesn't allow thawing regions.

## Concrete Types

### Machine-word types

Machine-word types, (ex, numeric) are singleton types (no fields) and do not have a _standard_ object representation (see below).
They are treated specially by the compiler and are represented as their machine equivalent bit-widths.
All numeric types are aligned naturally, except `Bool`, for which the backing storage type (and associated alignment) is at the discretion of the compiler.

* `Bool`: bit pattern 0 or 1. The storage is usually 1 byte but can be different.
* `U8`, `U16`, ... `U128`: power-of-two-bits, unsigned, on the platform's native endian.
* `I8`, `I16`, ... `I128`: power-of-two-bits, 2's-complement, on the platform's native endian.
* `F32`, `F64`: IEEE-754 (binary) floating point numbers.

Booleans can be packed (ex. 1 byte = 8 boolean values) for class fields and arrays on specific optimisations.

There are discussions on introducing non-power-of-two integers and other floating point sizes, but they won't be present for the first iteration of the compiler.

### Pointers

Pointers have different sizes depending on the target architecture:
* 32-bit integers representing an address.
* 64-bit integers representing an address.
* 128-bit CHERI capabilities granting access to an address range.

All Verona objects are at least 8-byte aligned and so the low three bits of a pointer will always be zero.
When targeting a system that uses integer addresses, all pointers have the same representation.
When targeting a CHERI system, the following additional rules apply:
* Pointers to `cown`s are sealed with a type reserved for the runtime.
  This ensures that `cown`s in immutable objects cannot be dereferenced by unsafe (sandboxed) code and provides some defense in depth against compiler bugs that would accidentally dereference a `cown` pointer.
* Pointers to immutable objects are not represented by read-only capabilities.
  Doing so would require expensive operations for reference-count manipulation.
  When a pointer to an immutable object is passed via the foreign-code layer, it must have store and load-storeable-capabilities permission removed, such that no reachable capability will ever provide write permission.

### Classes and interfaces

Concrete classes and interfaces each have their own unique header, with a unique descriptor.
Following a pointer and decoding the header gives you the type of the object and therefore there is no need to treat interfaces and classes differently.

Classes and interfaces are stored similar to a _C structure_.
The general layout is:
* A header.
* The list of fields (with `embed` values or pointers to objects).

The header is a pair of values:
* The region meta-data, a pointer-sized value containing information for the runtime library.
* A pointer to the type descriptor (see below).

Each object points to a descriptor that uniquely identifies its type.
Pattern matching by type, when the type of the object is not statically known at compile time, involves the descriptor.
Matching against a concrete type is a simple comparison: does the descriptor pointer in the object point to the descriptor for the concrete type?
Matching a pointer to an unknown concrete type (i.e. whose static type is an interface type) against an interface type requires a more complex lookup.

The remaining fields can, on internal representations, be packed or reordered for optimisation purposes.
Machine-word types (ex. numeric) are always represented by value (their singleton representation).

For example, given this class:
```ts
class Other
{
  var x: U64 & imm;
}
class Foo
{
  // Machine-word types are singletons, represented by their values
  var a: U32 & imm;
  // Embed inserts the structure inside the representation
  embed var b: Other & mut;
  // Objects are just pointers
  var c: Other & mut;

  create() { ... }
}
```

A naive layout could be:
```ts
// Evident problems with alignment...
{ { ptr, ptr }, i32,   i64,    ptr   }
     header,    U32, { U64 }, Other*
```

With `ptr` 32/64/128-bits, depending on the architecture.

As a future optimisation, we could reorder the fields by size.
Some targets have stronger alignment requirements, and unaligned reads can incur in performance penalties.
Having an 8-bit type between 64-bit types can misalign the larger objects.
By sorting the types by size, we guarantee that all 64-bit values are 64-bit aligned, all 32-bit values are 32-bit aligned and so on.

A more optimal layout for the same class above, on either 32 or 64 bit targets, would be:
```ts
// Alignment friendly layout
{ { ptr, ptr },   i64,    ptr,   i32 }
     header,    { U64 }, Other*, U32
```

Because `f64` aligns stronger than `i32` and `ptr` can be either.
On CHERI, the 128-bit `ptr` would be the first field after the header.

Note that `embed` structures could also be broken to keep stronger alignment in order.

For example:
```ts
class Foo
{
  var a: U64 & imm;
  var b: I8 & imm;
}
class Bar
{
  var c: I32 & imm;
  embed var d: Foo & mut;
}

// Bar's optimal layout
{ { ptr, ptr },   i64,      i32,      i8 }
     header,    { U64... }, I32, { ...I8 }
```
_Note: The header is never reordered, to allow for faster pattern-matching against type descriptor._

### Dispatch Tables & Selector Colouring

Object headers have a pointer to their type descriptor, which includes a dispatch table.
A dispatch table contains pointers to the functions that the type provides, at specific offsets.
Those pointers are called when it's not possible to determine the actual function being called at compile time.
There is only one dispatch table per type, not per object.

The offset of each method is calculated at compile time.
The address at each offset has a pointer to the actual functions.
Dynamic dispatch is done by taking the pointer at the offset from a table pointer and calling that.
If an interface provides a method `foo`, all classes that implement that interface will have a method `foo` and their dispatch tables will have an entry for it.

To avoid each type having a different calculation for each method's offset, the compiler will do **selector colouring**.
This process finds all common methods across all types and ensures the same methods all have the same offsets on all tables.
This applies to concrete classes that implement specific interfaces, but it's not limited by it.
Any two classes that have the same methods (signature) will end up with the same offset.

With multiple interfaces and classes implementing similar methods, this can lead to a number of gaps between offsets.
The colouring will compact the representation, removing some gaps, but how to minimise the number of gaps and when to run colouring is an open problem.

## Union Types

Union types need as many representations as there are types in the union.
Objects of union types need to be able to represent all types as well as identify which one for each instance.

Example:
```ts
// Must handle both U8 and F32
// Must also know which one at run time
function(arg : U8 | F32) { ... }
...
var a : U8 & imm = 10;
function(a); // Passes a 1-byte integer
var b : F32 & imm = 3.14;
function(b); // Passes 32-bit floating point
```

All objects are referred to by pointers.
Machine-word types are referred to by value as an optimisation by the compiler.

For that reason, union types can _contain_ only the following representations:
* Machine-word values.
* Pointers to objects.

To identify which type the run-time object has, we need a discriminator value (e.g., a flag or small bit field).
Because values generally take up all their storage, we usually need storage wider than the largest union member to bundle more than one type in the same representation.

Objects have a pointer to their type descriptor, which uniquely identifies their concrete types, so pointers implicitly carry a discriminator in the pointee when stored in the union representation.
So a union of pointers is represented as just a pointer and the discrimination will happen at the object level.
In contrast, machine-word types don't contain anything other than their data and so have no internal state that can serve as a discriminator.
As such, they need special handling when inside union.
Furthermore, mixing pointers and numeric types creates the need to differentiate pointers from the rest.

The type representation will be composed of a _discriminator_ and a _payload_ with the size of the largest object.
There are two ways of storing the discriminator: `wide packing` and `NaN-boxing`.

### Wide packing

This is the naive representation, using an 8-bit descriptor and the largest object's size as the payload.

Example:
```ts
(A | U64) -> (Pointer | i64) -> { i8, i64 } // On both 32-bit and 64-bit machines
// On CHERI, this would be { i8, i128 }, as pointers are 128-bits wide
```

This can be used for all types, but it creates two main problems:
1. It adds at least 8-bits to every number.
   For a `U8`, that's twice the size.
2. Alignment requirements may force us to _pad_ the first `i8`.
   This would add another 24, 56 or 120 bits on structures or arrays, per element.

For this reason, we use NaN-boxing below for all types that we can.
The encoding of the discriminator is shown below.

### NaN-Boxing

IEEE-754 doubles are 64 bits, on both 32-bit and 64-bit platforms, encoded in the exact same way:
```
[ sign ][ exponent ][ mantissa ]
63     62          51          0
```
So:
* 1 sign bit
* 11 exponent bits
* 52 mantissa bits

A `NaN` (not-a-number) is encoded in IEEE-754 as setting the exponent to _all ones_.
All other bits are ignored.
If we set the exponent to _all ones_, we can use the remaining 53 bits to encode a variety of smaller types.

Basically, if the exponent is **not** _all ones_, then the value represented is a 64-bit floating pointer number.
If it is, then we use some bits to encode which type it is (the discriminator), and the rest of the bits left to encode the value.

We use the sign bit to represent pointers, leaving the remaining 52-bits (the entire mantissa) to represent the address of the pointer.
Most 64-bit architectures use 52-bits or less to encode addresses, so that's enough for our purposes.
When the sign bit is zero, we use 4 high bits in the mantissa to encode all representable types, followed by a padding of `(52-4-sizeof(type))` bits, and the value taking the `sizeof(type)` lower bits.

We can encode all 32-bit or less types as well as pointers and `F64`, but not 64-bit integers or higher.
For those, we use the wide packing below.

The list of the types that can be NaN-packed:
* Pointer
* F64
* F32
* U32
* U16
* U8
* I32
* I16
* I8
* Bool
* ISize (on a 32 bit platform)
* USize (on a 32 bit platform)

In the following ways:
```
[*][***********][52 bit mantissa] = F64
[*][00000000000][52 bit mantissa] = F64 (zero and subnormals)
[*][11111111111][52 zeroes] = F64 (infinity)
[1][11111111111][00][50 data bits] = Pointer (active heap)
[1][11111111111][01][50 data bits] = Pointer (active stack)
[1][11111111111][10][50 data bits] = Pointer (paused heap)
[1][11111111111][11][50 data bits] = Pointer (paused stack)
[0][11111111111][0000][40*][8 data bits] = I8
[0][11111111111][0001][32*][16 data bits] = I16
[0][11111111111][0010][16*][32 data bits] = I32
[0][11111111111][0011][16*][32 data bits] = ISize (32)
[0][11111111111][0100][47*][1 data bit] = Bool
[0][11111111111][0101][32*][16 data bits] = F16
[0][11111111111][0110][16*][32 data bits] = F32
[0][11111111111][0111][48 ones] = F64 (quiet NaN)
[0][11111111111][1000][40*][8 data bits] = U8
[0][11111111111][1001][32*][16 data bits] = U16
[0][11111111111][1010][16*][32 data bits] = U32
[0][11111111111][1011][16*][32 data bits] = USize (32)
[0][11111111111][1100][48*] = Unused
[0][11111111111][1101][48*] = Unused
[0][11111111111][1110][48*] = Unused
[0][11111111111][1111][48 ones] = F64 (signalling NaN)
```

The order of types and their bit patterns are arbitrary, and the sub-divisions is provisory.

The following reasoning applies:
* We still need to represent `NaN`s as FP, so we can zero the mantissa
* Pointers with all-zero bits are `null` pointers and not allowed in Verona
* All other types have at least one bit in the discriminator

The reasoning and patterns can change depending on implementation details that will be clearer later.

The remaining types that need to be wide-packed are:
* I64
* U64
* I128
* U128
* ISize (on a 64 bit platform)
* USize (on a 64 bit platform)

Because we have more than 16 types in total, we need to add 1 extra bit to the discriminator for wide packing.
We also need to add another bit for identifying pointers, which we can use the top bit like NaN-boxing.

```
------------------------- Types also represented in NaN-boxing
[1][**][0][0000] Pointer
[0][**][0][0000] Bool
[0][**][0][0001] F32
[0][**][0][****] ... // Same bit-pattern as above
------------------------- New types, only in wide-packing
[0][**][1][0000] I64
[0][**][1][0001] U64
[0][**][1][0010] I128
[0][**][1][0011] U128
[0][**][1][0100] ISize (on a 64 bit platform)
[0][**][1][0101] USize (on a 64 bit platform)
```

This bit pattern is also arbitrary and can change in the actual implementation.

Note that, in 64-bit architectures, pointers (and `F64`) have the same width on both native and NaN-boxed representations.
All other types increase the width to at least 64-bit, even if they were `U8` to begin with.

Also note that the pattern above is an extra byte, so all values will have the whole 64/128 bits of the payload to be encoded.

## Arrays

Arrays are structures that contain a dynamically sized list of elements of a single type.
The memory layout of the array is guaranteed to be consecutive (ex. `addr(array[N+1]) == addr(array[N] + sizeof(type)`).

Example:
```ts
// No surprises here
Array[U16] -> [ i16, i16, i16, ...]
Array[A] -> [ ptr, ptr, ptr, ... ]

// Naive Bool array
Array[Bool] -> [ i8, i8, i8, ... ]

// Compact Bool array
Array[Bool] -> [ { i1, i1, i1, ... }, { i1, ... } ... ]
```

That type, however, can be a union type, so it needs descriptors.
The naive layout for an array of unions is to have an array of packed descriptors.

Example:
```ts
// NaN-boxed types
Array[U16 | I8] -> [ i64, i64, i64, ... ]
Array[U32 | F64] -> [ i64, i64, i64, ... ]
Array[A | F16] -> [ i64, i64, i64, ... ]

// wide-packed types
Array[A | U64] -> [ { i8, i64 }, { i8, i64 }, ... ]
Array[F64 | U128] -> [ { i8, i128 }, { i8, i128 }, ... ]
```

NaN-boxed types have the problem that it increases the size of the object at least to 64-bits (128-bit on CHERI), even when neither of the types are 64-bit wide.
This may not be a problem for individual objects, but an array with multiple objects will multiply the problem.

Example:
```ts
// Single-types
Array[I8] -> [ i8, i8, i8, ... ]
Array[U8] -> [ i8, i8, i8, ... ]

// NaN-boxed types
Array[U8 | I8] -> [ i64, i64, i64, ... ]
```

That, however, is a small problem when compared to the wide packing.
Not only the wide-packed array increases the payload, but it also introduces the descriptor which is not the same size as the payload.
This creates misalignment issues (on platforms where misaligned reads are penalised or forbidden, this is really bad).

An optimisation under discussion is to pack the descriptors into a payload-sized element.
With a 64-bit payload, the first 8 elements would follow the first 64-bit bucket with their 8 descriptors packed.
The following 8 elements would need a new descriptor bucket, placed after the 8th element, and so on.

Example:
```ts
Array[F64 | U128] -> [ { i8, i64 }, { i8, i64 }, ... ]

// to

type Desc = { i8, i8, i8, i8, i8, i8, i8, i8 };
Array[F64 | U128] -> [ Desc, i64, i64, i64, i64, i64, i64, i64, i64, Desc, i64, ...  ]
```

To access the 7-th element, the unpacking does:
```ts
// Takes the 7th descriptor (in pseudo tuple notation starting at _0)
descrip = Array[0]._6;

// Takes the 7th element (at 8th position starting from 0)
payload = Array[6+1];
```

Alternatively, to facilitate vectorisation, it might be more efficient to store parallel arrays of values and arrays of discriminators.
It is possible to optimise this further for cases where the total number of machine-word types in the array is limited.

For example `Array[Foo | U64]` requires only one bit of discriminator state to identify the type.
This could be compressed by providing a two-element map (16 bits of state in the array object) that indexes from a dense type representation to a the wide encoding.

Note that these problems only happen in arrays of unions that need to be wide-packed.
Users should prefer arrays of concrete types as much as possible for code that needs to be fast (inner loops, etc).

## Calling Convention

The Verona compiler must follow the target's calling convention, which could be a particular hardware, some sandbox technology, or calling external (Verona) code via shared libraries.

However, the compiler is allowed to change the encoding and order of the arguments and even remove completely some arguments that are shown not to be used, in order to improve performance, while still fitting to the calling convention used.

Luckily, by using LLVM, we don't need to worry about any calling convention features that "just work" and only need to look at issues if they prove to be a case common enough to show up in profiles.

This section discusses some of those opportunities.

### Concrete Types

Concrete types, including pointers, are always passed by value to functions.
Machine-word types are singleton types and immutable and so copies replicate the machine representation.
Other types are represented by pointer, so a _copy_ of the pointer is passed, but still pointing to the same object, so the object can be mutated (if via a `mutable` reference).

This is very friendly to modern calling conventions, that place values in registers and have plenty of large registers to use.

Most calling conventions have stronger alignment requirements which can add padding between arguments, but some can also pack multiple smaller values into a single register.
In this case, sorting the arguments by size like discussed in the structure layout can help packing more arguments into registers.

The compiler could also try to infer which variables would benefit to be in registers up-front and shuffle them out of the stack.

### Union Types

Union types, when NaN-boxed, are indivisible 64-bit values, even if their actual payload is 32-bit or less.
The only way to optimise this (reduce the width to the actual payload) is to synthesise a new version of the function with each concrete type.

The new version would need additional type inference and most optimisation passes, and it's not clear the gains at such an early stage.
Link-time optimisations, however, could do that with a lot more information about the state of the code and with hopefully less code to deal with.

On the other hand, when wide packed, union types have much clearer gains for packing.

The naive lowering for arguments of wide packed union types is for example:
```ts
function({ i8, i64 }, { i8, i128 });
```

On naive calling conventions, this would be passed on the stack.

To pass this in registers, at least one of two things should happen:
1. The compiler realise they could be split (ex. scalar replacement of aggregates).
2. The target's ABI rules allowing splitting structures and placing them in registers, like Arm's.

We can also change the arguments from two to four, into:
```ts
function(i8, i64, i8, i128);
```

This would allow better register placement without relying on the compiler, which could make a difference on quirky corner cases.

We could even go a step further and reorder the arguments to help packing the descriptors into a single register:
```ts
function(i8, i8, i64, i128);
// or even
function(i16, i64, i128);
```

The second form would require lowering wrappers to pack and unpack the descriptors, which would only make sense if the numbers of registers available for arguments is really restricted, or there are too many wide packed union arguments.
