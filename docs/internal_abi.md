Internal Verona ABI (Draft)
===========================

Verona expects to use foreign-language ABIs (C/C++) for interoperability and reserves the right to change the internal ABI for Verona components between compiler releases.
That said, it is useful for people working on the compiler, producing additional tooling, and writing tests for the ABI used internally to be documented.
This draft is a proposal for the v1 ABI.

Primitive types
---------------

The primitive types in Verona are:

 - Boolean
 - Integers, signed or unsigned, with 1-128 bits of precision
 - Floating point values, with various power-of-two sizes
 - Pointers to objects

The representation of these may change when used in a complex type expression but in the absence of other constraints they will be stored as simple values, as described in the following sections.
The compiler is free to relax any of these constraints for on-stack variables that are not passed by reference (for example, leaving some bits in a value undefined if the compiler can prove that they are never read).

Verona places the following restrictions on target architectures:

 - The size of a byte is 8 bits.
 - The architecture can load and store a single byte without read-modify-write operations
 - The order of bits within different-sized values is consistent
 - Signed arithmetic on integers uses twos complement

This is a slightly tighter set of requirements than the C specification but is looser than the de-facto C standard as used in large deployed codebases.

### Booleans

Boolean values are stored as bytes, with the bit pattern 0 or 1 in the platform's native endian.
The compiler is responsible for ensuring that no bits other than the least-significant bit in a boolean are ever non-zero.

### Integers

Power-of-two-sized integers in Verona can be used for arithmetic and so are expected to use the representation that the target architecture uses for integer operations.
The types `U8`, `U16`, `U32`, `U64`, and `U128 are all unsigned fixed-width integers with the platform's native endian.
The types `S8`, `S16`, `S32`, `S64`, and `S128 are all signed fixed-width integers represented as twos complement numbers with the platform's native endian.
These types are always stored naturally aligned.
For example, a `U64` is always stored on an 8-byte boundary.

Other integer types, for example `U56` are *storage-only* types.
These must be extended to a power-of-two sized type for arithmetic.
When stored on the stack, these should also be sign- or zero-extended (for signed and unsigned types, respectively) to the next native integer type.
The value of these types is primarily for use in structures or union types, where special layout rules apply.

### Floating-point values

The types `Float16`, `Float32`, `Float64` and `Float128` correspond to IEEE 754 binary16, binary32, binary64, and binary128 types, respectively.
*Note:* A future version of the language may provide IEEE decimal types, binary256, bfloat32, or smaller floating point types but these are not necessary for MVP.
All floating-point types are stored in the target's natural floating-point byte order (which is commonly, but not always, the same as the integer byte order) and are naturally aligned.
If the target does not support hardware floating point then floating point values are stored according to the C ABI for the platform.

### Pointers

We anticipate three common representations for pointers on Verona target systems:

 - 32-bit integers representing an address
 - 64-bit integers representing an address
 - CHERI capabilities granting access to an address range.

All Verona objects are at least 8-byte aligned and so the low three bits of a pointer will always be zero.
When targeting a system that uses integer addresses, all pointers have the same representation.
When targeting a CHERI system, the following additional rules apply:

 - Pointers to `cown`s are sealed with a type reserved for the runtime.
   This ensures that `cown`s in immutable objects cannot be dereferenced by unsafe (sandboxed) code and provides some defense in depth against compiler bugs that would accidentally dereference a `cown` pointer.
 - Pointers to immutable objects are *not* represented by read-only capabilities.
   Doing so would require expensive operations for reference-count manipulation.
   When a pointer to an immutable object is passed via the foreign-code layer, it must have the load-storable-capabilities permission removed, such that no reachable capability will ever provide write permission.

*Note:* We currently assume that pointers point to the start of objects.
Most instruction sets; however, use signed offsets for loads and stores.
For example, an x86 `mov` instruction can access offsets -128 to 127 in a three-byte instruction and requires (at least) a five-byte instruction for larger displacements.
It may therefore make sense for object and array pointers to be biased by the displacement of this offset.
The SPARC ABI, for example, uses this trick to bias the stack pointer to maximise the available displacement in a single instruction.

Pointers to interface types
---------------------------

Pointers to concrete types are simply pointers.
Pointers to interface types are a pair of a pointer to the object followed by a pointer to the vtable for the interface.
There are no additional requirements on alignment for interfaces: the pair of pointers is aligned on the same boundary as a single pointer.

Union types
-----------

Variables or fields of union types are *discriminated unions*.
The simplest representation would be to store a buffer of the largest type and an integer large enough to store a unique value for each type.
For example, a `U8 | Float64 | Foo*` could be represented as a 64-bit buffer followed by a 2-bit integer.
At an abstract level, the Verona internal ABI does exactly that, with two observations that allow denser packing:

 - Singleton types (equivalent to Pony primitives [*note:* do we have a name for these yet]) have a single instance and so the body can be elided in some cases and the value stored entirely in the tag discriminator.
 - Many types have some 'unused' bit patterns that can be used to embed the tag bits.

In the above example, a `U8` defines 8 bits but, if stored in a 64-bit buffer, leaves 56 unused.
The `Float64` does not have any unused bits but if the 11 exponent bits and the signalling bit are all 1, the value is a signalling not-a-number (NaN) encoding and the remaining 52 bits are unused.
The `Foo*`, because it points to a Verona object, has zero in the low three bits and, on a typical 64-bit system with a 48-bit address space has zeroes in the top 16 bits and so needs only 45 bits to encode all valid values (on a system with a 56-bit address space, it still requires only 52 bits to represent any userspace address and so can fit in the NaN space).

It is therefore possible to fully encode this example in a single 64-bit word on any common 64-bit system.
For example, in userspace on a little-endian 64-bit system with a 48-bit virtual address space that uses integers as pointers:

 - If any of bits 50-62 inclusive is non-zero, then the entire value is a `Float64`.
 - Otherwise, if the value of bit 0 is 1, then bits 8-15 inclusive contain a `U8` value.
 - Otherwise, bits 0-47 contain should be zero extended to give a 64-bit address.

This form of encoding is one of the motivations for supporting storage-only types such as `U42`.
Any type that does not use the full width of a machine word is amenable to dense packing in discriminated unions.
This is a sufficiently common pattern in systems code that Verona has chosen to bake it into the language.

The rules for determining the correct encoding for a union type begins by identifying the number of payload bits and the number of states required for the type discriminator:

 - For singleton types, there are two possible answers:
   - Each type requires one state in the discriminator but no bits to encode the value.
   - Each type requires one pointer's worth of state to encode the value but no bits in the discriminator.
   We choose the former, to maximise our ability to use the CPU's branch-on-zero instructions.
 - Each integer requires one state in the discriminator and as many bits as its width defines for the payload.
   For example, in the union `U8 | U16`, each requires one state in the discriminator (giving a 1-bit discriminator to represent both state) and 8 and 16 bits of payload, respectively.
 - Floating point types follow the same rules as integers, with special handling if a floating point type is the type with the largest payload in a union type, described below.
 - Pointers to object types (including external reference types) require as many bits as are required for an unambiguous pointer; however, all pointer types require only a single discriminator state because all pointees can be discriminated by inspecting their descriptor.
   The number of bits required for a pointer varies depending on the pointer representation:
   - 32-bit integer pointers require 29 bits of state.
   - 64-bit integer pointers require either 45 or 52 bits, depending on the virtual address space size of the target.
   - CHERI capabilities require 125 bits of state but provide a 'free' discriminator bit because the tag bit can be used to discriminate CHERI capabilities from non-pointer data.
 - Value types are still being defined; however, it seems plausible that they will each require one descriptor state and as many payload bits as it has non-padding bits.
   Making efficient use of padding bits as discriminators seems very difficult and so is likely to be left to a later ABI, if ever implemented.

Once the payload sizes are all known, the algorithm for implementing the packing depends on the type of the largest payload.
There are two special cases:

If the largest type is a floating-point type and the payload of all other types plus the bits required for the discriminator will fit in the unused bits of the significand of a signalling NaN minus one then the top bit of the significand is reserved to discriminate between a real signalling NaN and another value and the remainder of the types are encoded in the spare bits, starting this algorithm again.
If the remaining bits of the significand are not sufficient to encode the remaining states then this rule is ignored.

If the largest type is a CHERI capability then the tag bit is used to differentiate pointer and non-pointer types.
This leaves 128 bits to use for the remainder of the values.

The size of the final object is the number of payload bits, plus the base-two logarithm of the number of discriminator states rounded up.
The alignment of the result is the alignment of the most strongly aligned type.

If one of the types is a pointer, the object size is pointer sized, and the discriminator is three or fewer bits, then the low bits of the pointer are used for the discriminator.
Pointer values are stored with their bits in the natural positions and must have the low bits subtracted away on use.
Other values are stored in the most significant bits.

*Rationale:* Most ISAs make subtracting a 3-bit immediate trivial (often something that can be folded into another instruction).
Other values that fit in the payload space can be shifted down as required in a single-cycle instruction.

If the payload is an integer number of bytes, the payload is stored first (independent of byte order) so that the payload can benefit from strong alignment, which avoids the payload accidentally spanning a cache line.
In this case, all elements of the payload are stored starting from the beginning of the payload.

*Rationale:* The payload can be loaded with a single aligned load on any architecture and depending on the target byte order then needs to do either a single zero-extending shift right or a left shift followed by a zero-extending right shift.

If storing the discriminator bits at either end of the space results in different numbers of complete bytes for the payload then the discriminator should stored at the end that results in the larger number of complete bytes for the payload.
If there are no discriminator bits, this rule is applied as if there were a single bit.
Payload values that are smaller than the payload space are then stored aligned to the opposite end of the value to the discriminator.

*Rationale:* This case is reached if we are packing a union type inside the significand bits of a signalling NaN and guarantees that we can load the payload in a single aligned load and need to zero bits at only one end to extract the payload.

In other cases, the discriminator bits are stored in the low bits (as defined by the platform byte order) of the object and the payload in the high bits.

*Rationale:* The discriminator is easy to mask off with a right shift and extract with an xor with an immediate operand for comparisons.

The types identified in a discriminator are sorted lexically and discriminator values are assigned in order.
Singletons are sorted before all other types.

*Rationale:* The representation of `A | B` and `B | A` should be the same.
It is useful to sort singletons before pointers so that the common case of `T* | NotAThing` is able to use null-pointer checks and branches, which are typically optimised aggressively in modern microarchitectures.

{*Note:* I'm not sure lexical sorting will work because types imported from the same package by two other packages will have different names and so we may need some other stable ordering for types.}

*Open question*: As described, this requires a pointer dereference do disambiguate pointer types.
It would be possible to use any space in the low three bits that is not used to disambiguate non-pointer types to differentiate a small number of pointee types.
This would save a load in pattern matching and might be useful, though given that the main reason for pattern matching is to then call a method on the concrete type, it's not clear that this is actually a win.
Note that, in the CHERI case, the low bits are still available and so it may be useful to use them to differentiate between pointee types.

These rules are somewhat complex in the abstract.
The following sections give some concrete examples of how they are applied.

### Example `U64 | Float64`

The size of `U64` is 64 bits and it requires one discriminator state.
The size of `Float64` is 64 bits and it requires one discriminator state.

We first try the special-case rule for the case that the largest value is a floating-point value.
The remaining type (`U64`) requires 64 bits for payload and 0 bits for discriminator, and so does not fit into the significand part of a `Float64` signalling NaN.
The first special case is therefore not used
The second special case is not used because neither type is a pointer.

We are now left requiring one bit of discriminator and 64 bits of payload.
The payload size is a power of two, and so the address of the union type is the address of the start of a 64-bit payload.
The 1-bit discriminator is stored afterwards, with a value of 0 representing a `Float64` and a value of `1` representing a `U64`.
The field-packing algorithm is free to use the remaining 7 bits in the discriminator for other values.

### Example: `Float32 | U20 | U16`

The size of `Float32` is 32 bits and requires one discriminator state.
The size of `U20` is 20 bits and requires one discriminator state.
The size of `U16` is 16 bits and requires one discriminator state.

The first special case applies: the largest type is a floating-point type and the remaining types require 20 bits of payload and one of discriminator, which fits in the significand for a `Float32` signalling NaN.

Placing the discriminator in the low bits would give one byte and two partial bytes for the payload.
Placing the discriminator in the high bits would give two complete bytes and one nybble for the payload.
The disciminator is therefore placed in bit 20, leaving bits 0-19 for the payload.

The `U20` uses all of the available payload bits and so is stored in bits 0-19.
The `U16` follows the rule that it should be aligned to the end of the opposite end to the discriminator and is therefore stored in bits 0-15.

A match expression first loads the entire 32-bit value and compares bits 21-31 against an all-ones bit pattern.
If the comparison result is not-equal, the entire value is a `Float32`.
If the comparison result is equal, then a value of bit 20 is tested, with zero indicating a `U16` and one indicating a `U20`, either of which can be extracted by a bit shift (note: On Arm systems, either value can be extracted with a single `UBFX` instruction, on x86 systems that support the BMI1 instructions a BEXTR can do the same).

### Example: `Float64 | T* | U*`

The payload size of `Float64` is 64 bits and it requires one discriminator state.
The payload size of `T*` and `U*` is each either 32, 48, or 128 bits, depending on the target architecture and they require one discriminator state between them.
The lowering for this union therefore depends on the target.

On 32-bit systems, the largest type is `Float64` and the remaining types all fit easily in the significand bits.
There are no discriminator bits required but the 'as-if' rules triggers as if there were a single bit for determining which end of the space should be used and the remaining values are stored at the end of the space that is strongly aligned: bits 0-31.

On 64-bit systems, the same rules apply, packing the used bits of the pointer into bits 0-50.

On CHERI systems, the pointer types are larger than the `Float64` and so the payload is a 128-bit value, distinguished by the tag bit.

### Example: `T* | None`

In this example, `None` is a singleton type with no fields, used as a placeholder indicating a not-present value.
The `T*` requires one pointer's worth of space for the payload (as above, this varies depending on the target) and one discriminator state.
The `None` value requires one discriminator state and no payload.

The largest value is a pointer and so on CHERI systems the tag bit is used as the discriminator, on other architectures the low bit of the pointer is used (because the pointer is the largest type and the discriminator fits in a three-bit value).
The discriminator values are assigned starting from the singletons, so zero is used for `None` and one for `T*`.

A pattern match expression on this type is compiled into a compare-and-branch-on-{non}-zero, which is a single instruction on most architectures.
Accessing the `T*` on a CHERI system requires no further operations.
On integer-pointer systems, it requires subtracting one.
On any mainstream architecture other than RISC-V, this subtraction can be folded into the immediate offset on a load instruction and so requires no additional instructions.

If the `None` value is used (rather than just being pattern matched on and ignored) then the compiler must be able to materialise the address (or capability) to the singleton.
This is no worse than if we stored the address inline; however, because in that case the match would require a comparison against the address and so the address would need to be loaded before the pattern match, rather than conditionally in one path.

Objects
-------

Verona objects contain a header with metadata describing the object, followed by user-defined fields.

### Object headers

Object headers are two pointer-size words.
These are described in detail in [src/rt/object/object.h](../src/rt/object/object.h).

The first word contains the region metadata.
This is a discriminated union where the low three bits describe how the remainder is used (for example, as a reference count for one immutable object in an SCC, a pointer to that object for other immutable objects in an SCC, and a pointer to the region metadata for sentinel objects).

The second is a pointer to the descriptor for the object.
The low bits of this are used by the runtime for memory management state and so must be zeroed before accesses.

The object descriptor layout is still likely to change, the current layout is:

64-bit integer size.
Pointer to a trace function, used in traced regions to walk reachable objects.
Pointer to a finalise function, called when an object is dead but before other objects in the same region are deleted (though they may have already been finalised.
Pointer to a notify function, invoked when an object is notified of a noticeboard event.
Pointer to a destructor, which is called after some objects in a region have been deallocated and may not follow any `mut` pointers.
The remainder of the descriptor is the vtable.

Methods in the vtable use *selector colouring*.
Each selector (method name and types) is assigned a separate vtable index.
Pairs of selectors that do not appear in the same interface are then combined to use the same index.
Selector colouring is not essential but can dramatically reduce the size of vtables.

*Note:* If we can distinguish between sentinel objects and others at object-creation time, the region metadata field for non-sentinel objects could be a 32-bit value that is either an integer or an index into a table, rather than a pointer.
This would be a noticeable saving on CHERI systems and a small saving on 64-bit systems, at the cost of an extra indirection for some memory-management operations.
We could also use a 32-bit index into a descriptor table, as the Fuchsia C++ ABI does: Programs that have more than 2^32 distinct types arise only as a result of poor life choices.

*Note:* The above choices are visible to the runtime but are part of the internal ABI and so it would be possible to have size and performance optimised variants available on a per-program basis.

*Note:* On AArch64 systems where the top-byte-ignore feature is available, using the top bits in the descriptor pointer instead of the bottom ones would improve efficiency of descriptor accesses.

### Field layout

In C, `struct` layout is a trade between storage density and accidental aliasing.
If two fields are accessed concurrently, false sharing in the cache can cause significant performance bottlenecks.
In Verona; however, no mutable objects are ever shared and so, aside from reference count manipulations on immutable objects and interactions between adjacently allocated objects in different regions, there are no negative interactions with modern caches.
The Verona field layout algorithm is therefore free to optimise for space.

All fields whose size is a power-of-two should be sorted by size and laid out largest to smallest.
The end of each section is more strongly aligned than the next requires and so this does not require any padding.

Next, any union types that have a power-of-two sized payload but a smaller discriminator should be split into payload and discriminator.
The payloads should be inserted into the sections for the same sized power-of-two objects.
The discriminators should be packed into a single larger value.

*Rationale:* Extracting from a bitfield is no more expensive if the bitfield is densely packed and if a function is pattern matching on multiple fields then they may be able to reuse a register for the descriptor values.

The remaining values should be stored in the order in which they are declared, between the power-of-two region and the descriptors.

*Rationale:* We could do some clever bin packing here but systems programmers who are using odd-sized storage-only types are likely to be careful about arranging them usefully already.
A future version can improve bin packing and not break anything.


*Note:* Java implementations typically pack all pointers together such that the trace logic just needs to look at a range, rather than have data or code describing the complete layout of the object.
This may be possible for Verona, depending on the constraints for value types as fields and on pointers to value types.
If value types cannot be passed by reference then they do not need to be stored contiguously in fields, because they will always be either copied when passed down the stack or accessed from functions that are aware of the location of their components.

External references
-------------------

External references are structures with the same header as a Verona object followed by two pointers, one to the table that stores the external reference and one to the object.
The rules for operations on this are defined in [`externalreference.h`](../src/rt/region/externalreference.h).
External references are never embedded in other structures, they are always referenced by pointer and so fields or arrays of external references follow the same rules as any other kind of object pointer.

Arrays
------

Arrays are objects (with the standard header) followed by a run of values of the same type.

*Open question*: Should `Array[Bool]` be equivalent to `Array[U1]` or `Array[U8]` in storage?

*Open question*: Should descriptors for array elements be collected in the same way as for structures?
