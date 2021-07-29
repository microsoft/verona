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

## Concrete Types

### Numeric 'natives'

Numeric types are singleton types (no fields) and do not have a _standard_ object representation.
They are treated special by the compiler and are represented as their machine equivalent bit-widths.
All numeric types are aligned naturally, except booleans, which align to 1 byte.

* `Bool`: 8-bits, with the bit pattern 0 or 1 in the platform's native endian.
* `U8`, `U16`, ... `U128`: power-of-two-bits, unsigned, on the platform's native endian.
* `I8`, `I16`, ... `I128`: power-of-two-bits, 2's-complement, on the platform's native endian.
* `F16`, `F32`, `F64`, `F128`: IEEE-754 (binary) floating point numbers.

There are discussions on introducing non-power-of-two integers, but they won't be present for the first iteration of the compiler.

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
  When a pointer to an immutable object is passed via the foreign-code layer, it must have the load-storable-capabilities permission removed, such that no reachable capability will ever provide write permission.

### Classes and interfaces

Concrete classes and interfaces each have their own unique header, with a unique descriptor.
Following a pointer and decoding the header gives you the type of the object and therefore there is no need to treat interfaces and classes differently.

Classes and interfaces are stored similar to a _C structure_.
The general layout is:
* A header.
* The list of `embed` fields.
* Pointers to the remaining fields.

The header is a pair of values:
* The region meta-data, a 64-bit value containing information for the runtime library.
* The descriptor, a pointer to the `vtable` (see below) and additional meta-data.

Example:
```ts
class Other { ... }
class Foo
{
    var a : U32 & imm;
    embed var b : F64 & imm;
    var other : Other & mut;
    create() { ... }
}
// The layout on a 32-bit machine could be:
// { { i64, i32 }, i32,  f64,  i32   }
//      header,    U32*, F64, Other*
//
// The layout on a 64-bit machine could be:
// { { i64, i64 }, i32,  f64,  i32   }
//      header,    U32*, F64, Other*
```

The remaining fields can, on internal representations, be packed or reordered for optimisation purposes.
But the `embed` fields will always be _in-place_ and the rest will always be pointers to the actual data.

As a future optimisation, we could reorder the fields by size.
Some targets have stronger alignment requirements, and unaligned reads can incur in penalties.
Having an 8-bit type between 64-bit types can misalign the larger objects.
By sorting the types by size, we guarantee that all 64-bit values are 64-bit aligned, all 32-bit values are 32-bit aligned and so on.

Each type has its own unique descriptor which uniquely identify the type.
Matching types of dynamic objects means comparing them to a known value or the value of the descriptor from another type.

### VTables & Selector Colouring

Object headers have a pointer to their type's virtual dispatch table.
A `vtable` contains pointers to the functions that the type provides at specific offsets.
Those pointers are called when it's not possible to determine the actual function being called at compile time.

The offset of each method is calculated at compile time.
The address at each offset has a pointer to the actual functions.
Dynamic dispatch is done by taking the pointer at the offset from a vtable pointer and calling that.
If an interface provides a method `foo`, all classes that implement that interface will have a method `foo` and their vtables will have an entry for it.

To avoid each type to have a different calculation for each method's offset, the compiler will do **selector colouring**.
This process finds all common methods across all types and ensure the same methods all have the same offsets on all vtables.
This applies to concrete classes that implement specific interfaces, but it's not limited by it.
Any two classes that have the same methods (signature) will end up with the same offset.

With multiple interfaces and classes implementing similar methods, this can lead to a number of gaps between offsets.
The vtables will be compacted (either during colouring or afterwards) to minimise that.

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

Union types are algebraic, so `((A | B) | C) == (A | (B | C)) == (A | B | C)`.
Intersection types create a new type, so `(A & B) =/= A && (A & B) =/= B`.
Excluding native numeric types, all objects are referenced to by a pointer.
For that reason, they can only _contain_ the following types:
* Native numeric types.
* Pointers to classes or interfaces.

Because all objects have a descriptor that identifies their types, pointers don't need a special descriptor in the union representation.
So a union of pointers is represented as just a pointer and the discrimination will happen at the object level.
However, numeric types don't have discriminators, so they need special handling when inside union.
Furthermore, mixing pointers and numeric types creates the need to differentiate pointers from the rest, so they then need special representation.

The type representation will be composed of a descriptor and a _payload_ whit the size of the largest object.
There are two ways of storing the descriptor: wide packing and NaN-boxing.

### Wide packing

This is the naive representation, using an 8-bit descriptor and a `max(sizeof(type...))` as the payload.

Example:
```ts
(A | U64) -> (Pointer | i64) -> { i8, i64 } // On both 32-bit and 64-bit machines
// On CHERI, this would be { i8, i128 }, as pointers are 128-bits wide
```

This can be used for all types, but it creates two main problems:
1. It adds at least 8-bits to every number.
   For a `U8`, that's twice the size.
2. Alignment requirements may force us to _pad_ the first `i8`.
   This would add another 24 or 56 bits on structures or arrays, per element.

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
For those, we use the wide packing above.

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
[1][11111111111][52 data bits] = Pointer
[0][11111111111][0000][47*][1 data bit] = Bool
[0][11111111111][0001][40*][8 data bits] = F8
[0][11111111111][0010][32*][16 data bits] = F16
[0][11111111111][0011][16*][32 data bits] = F32
[0][11111111111][0100][40*][8 data bits] = I8
[0][11111111111][0101][32*][16 data bits] = I16
[0][11111111111][0110][16*][32 data bits] = I32
[0][11111111111][0111][16*][32 data bits] = ISize (32)
[0][11111111111][1000][40*][8 data bits] = U8
[0][11111111111][1001][32*][16 data bits] = U16
[0][11111111111][1010][16*][32 data bits] = U32
[0][11111111111][1011][16*][32 data bits] = USize (32)
```

The order of types and their bit patterns is arbitrary.

The remaining types that need to be wide-packed are:
* I64
* U64
* I128
* U128
* F128
* ISize (on a 64 bit platform)
* USize (on a 64 bit platform)

Because we have more than 16 types, we need to add 1 extra bit to the discriminator for wide packing.
We also need to add another bit for identifying pointers, which we can use the top bit like NaN-boxing.

```
[1][**][00000] Pointer
[0][**][10000] I64
[0][**][10001] U64
[0][**][10010] I128
[0][**][10011] U128
[0][**][10100] F128
[0][**][10101] ISize (on a 64 bit platform)
[0][**][10110] USize (on a 64 bit platform)
```

This bit pattern is also arbitrary and can change in the actual implementation.

Note that, in 64-bit architectures, pointers (and `F64`) have the same width on both native and NaN-boxed representations.
All other types increase the width to at least 64-bit, even if they were `U8` to begin with.

Also note that the pattern above is an extra byte, so all values will have the whole 64/128 bits of the payload to be encoded.

## Arrays

Arrays are structures that contain a dynamically sized list of elements of a single type.
The memory layout of the array isn't guaranteed to be consecutive memory (ex. `addr(array[N+1]) == addr(array[N] + sizeof(type)`).
Since there is no pointer arithmetic in Verona, the compiler is free to change the offset calculations to suite the allocation strategy.

Example:
```ts
// Bool has 8-bits (we can pack it later using bitfields)
Array[Bool] -> [ i8, i8, i8, ... ]
// No surprises here
Array[U16] -> [ i16, i16, i16, ...]
Array[A] -> [ ptr, ptr, ptr, ... ]
```

That type, however, can be a union type, so it too needs descriptors.
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

That, however, is a small problem when compared to the wide packing.
Not only the wide-packed array increases the payload, but it also introduces the descriptor which is not the same size as the payload.
This creates misalignment issues (on platforms where misaligned reads are penalised or forbidden, this is really bad).

An optimisation under discussion is to pack the descriptors into a payload-sized element.
The first 8 elements would follow the first bucket with their 8 descriptors packed.
The following 8 elements would need a new descriptor bucket, placed after the 8th element, and so on.

These problems only happens in arrays of unions that need to be packed.
Users should prefer arrays of concrete types as much as possible for code that needs to be fast (inner loops, etc).

## Calling Convention

The Verona compiler must follow the target's calling convention, which could be a particular hardware, some sandbox technology, or calling external (Verona) code via shared libraries.

However, the compiler is allowed to change the encoding and order of the arguments and even remove completely some arguments that are shown not to be used, in order to improve performance, while still fitting to the calling convention used.

Luckily, by using LLVM, we don't need to worry about any calling convention features that "just work" and only need to look at issues if they prove to be a case common enough to show up in profiles.

This section discusses some of those opportunities.

### Concrete Types

Concrete types, including pointers, are always passed by value to functions.
A native numeric type will create a copy of the value to be used inside the function.
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
