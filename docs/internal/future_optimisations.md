Future Optimisation Opportunities
=================================

The Verona language affords a number of optimisation opportunities that do not need to be implemented for a minimum viable product.
This document exists to capture them so that they can be implemented in a future version and so that the design of the compiler doesn't preclude their implementation.
They are not in any particular order.

Late selector colouring
-----------------------

In the MVP compiler, selector colouring is done in the front end.
This means that we need to assign descriptor state even for interface calls that are possible to elide.
The front end can make some early decisions:

 - If only one concrete type is ever cast to an interface type then elide the interface entirely and reify everything for the concrete type.
 - If a method on an interface type is never called, don't assign a selector for it.

After inlining, some of these cases become clearer.
For example, if a function that takes an interface and uses `match` expressions is inlined into the callers then it may be possible to resolve the matches entirely at compile time and completely eliminate the interface creation.
This, in turn, means that there is no need to assign selectors for the methods that are used only by this interface (and no need to generate code for them if they are inlined).

Late reification
----------------

In the MVP compiler, generics will be reified in the front end, methods that take interfaces will not be reified.
After inlining the caller of a method that takes an interface parameter, we may discover that it is always called with the same concrete type at a specific call site.
Reifying the called method at this point may make it a better inlining candidate if it eliminates some code in `match` expressions.

For example, consider a function of the following shape:

```verona
someFunction(object : Interface)
{
	match (object)
	{
	  _MyPrivateConcreteType => fastPath(object);
	   => slowPath(object;
	}
}
```

If this function is reified for a call site where the type is known to be `_MyPrivateConcreteType` then the reified version is a single tail call and so is trivially inlined.

Conditional jumps for interface calls
-------------------------------------

Method invocations via interfaces are indirect jumps via the descriptor, roughly equivalent to the following in C:

```c
object->descriptor[selector](object);
```

For low-order polymorphic call sites, this can be more efficiently lowered to:

```c
if (object->descriptor == ClassA)
	ClassAMethod(object);
else if (object->descriptor == ClassB)
	ClassBMethod(object);
```

This transform is often done by dynamic language VMs because it both plays well with branch predictors and because it exposes inlining opportunities.
It may be beneficial to inline one or both of the methods, which improves locality of the code and then exposes further optimisation opportunities from specialising the method for this specific call site.

Descriptor Elision
------------------

The descriptor is used for dynamic dispatch and for `match` expressions.
If a class is never cast to an interface type (including a union type) then it should be possible to completely elide its descriptor.
This is particularly useful for small objects, where the descriptor pointer is a large part of the total size.

Region-metadata elision
-----------------------

Objects created as `embed` fields or as entries in arrays need region metadata because they can have their address taken and be passed to other functions that are not aware of the container.
If reachability determines that they are not captured by the called functions then there are two cases to consider:

 - If the region is used then the called functions could be specialised to pass the region of the container as a separate argument.
 - If the region is not used then the region metadata is never accessed.

The second case is a clear size reduction, particularly for objects used as keys in hash tables and so on.
The first case may result in significant code-size increases and so is worth doing only after applying a cost model.

The combination of this and descriptor elision should make it possible to have efficient unit types.
For example, consider something like this trivial example:

```verona
class Seconds
{
	var _value : U64;
	+(self: Self & mut, inc : U64)
	{
		_value = inc;
	}
}
```

In the MVP compiler, this will require three words (the region metadata, the descriptor, and the 64-bit integer).
On a CHERI system, this ends up being 40 bytes to store an 8-byte quantity.
Now consider the following (assuming `Nanoseconds` has the same layout as `Seconds`):

```verona
class TimeInterval
{
	embed _seconds     : Seconds;
	embed _nanoseconds : Nanoseconds;
	// Methods here.
	<(self : Self, other : TimeInterval) { ... }
}
```

This class stores 16 bytes of state but would require 96 bytes in the MVP compiler (on a non-CHERI 64-bit system).
Both `_seconds` and `_nanoseconds` are used by only this class and so we should be able to elide both of their descriptors and region metadata fields, shrinking this class to 32 bytes.
If `TimeInterval` is never used in an interface or union type, then we can shrink it to 24 bytes.
If nothing uses `TimeInterval` for region identification then we can shrink `embed` fields of type `TimeInterval` to 16 bytes.

Note that it is not possible to remove the region metadata for objects that are not allocated within other objects in the general case because it is needed for memory management.
It may be possible to elide region metadata for objects if all following hold:

 - They are allocated in arena regions.
 - They do not have a finaliser.
 - Nothing uses objects of this type to dynamically identify a region.

The first two means that there is no need to track state for memory management.
The objects remain life until the region is deallocated and are then cleaned up by a bulk free operation.
The last means that nothing outside of the memory manager needs to inspect the region metadata.

As a further optimisation, if another object from the same region is visible in all callers of the function that access the region and the first two conditions are satisfied, then an additional region parameter could be added to the functions to propagate the region information.

Tagged unions for closed-world types
------------------------------------

A union type with a small number of concrete types (for example `A | B`) will be represented in the MVP compiler as an interface that contains all of the methods in `A` and all of the methods in `B` (and possibly some others generated by the compiler to support later casts).
Casting back to `A` or `B` requires comparing the descriptor pointer.
This requires materialising the global addresses of the descriptors and loading from the object.

We could alternatively represent these using the low bit(s) as a discriminator.
This is useful only in combination with converting dispatch to conditionals, where a method call now becomes:

```c
if (object & 1)
	MethodOnA(object & ~1);
else
	MethodOnB(object);
```

Modern ISAs (AArch64, recent x86) have bitfield extract instructions.
For the two-type case, AArch64 has a conditional branch on a single bit and so this lowers to:

```
        tbnz    w0, #0, .LBB0_2
        bl      MethodOnB(obj*)
        b       .LBB0_3
.LBB0_2:
        and     x0, x0, #0xfffffffffffffffe
        bl      MethodOnA(obj*)
.LBB0_3:
```

The same pattern applies if either method is inlined.

The transform would also expose more opportunities for descriptor elision.
If a class is only ever stored in tagged unions with the tag embedded in the pointer, then dynamic dispatch and `match` do not need to read the descriptor field and it can be elided.
