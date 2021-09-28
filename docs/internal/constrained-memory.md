Proposal for dealing with constrained memory environments
=========================================================

*Note* This is a draft proposal for a feature that is not required for the MVP and so will not be implemented in the near future.

This RFC details some Verona extensions that may make it easier to deploy Verona in environments with limited memory.

Background
----------

Verona is an infrastructure language and so is intended to be usable in the higher-level parts of an OS kernel (i.e. those that are not responsible for providing the language runtime, such as the memory manager or scheduler) and similar environments.
Unlike application development, which can treat memory as an infinite resource and someone else's problem to manage, these environments must treat memory exhaustion as a real possibility.

Most kernels provide two variants of memory allocation, one that can fail and one that can block indefinitely.
For example, BSD's [`malloc(9)`](https://www.freebsd.org/cgi/man.cgi?query=malloc&sektion=9) requires either a `M_NOWAIT` or `M_WAITOK` flag to indicate whether blocking is allowed, Linux's [`kmalloc`](https://www.kernel.org/doc/htmldocs/kernel-api/API-kmalloc.html) provides a `GFP_NOWAIT` flag to disallow sleeping.
This dichotomy is necessary because code holding non-sleepable kernel locks must not block on allocation failure.

This exact problem should not apply in Verona because the memory allocator sits at the same abstraction layer as the rest of the runtime and strictly below any Verona code.
This means that the memory allocator cannot depend on any cowns in the Verona world.
There is a closely related problem: Verona behaviours are cooperatively scheduled.
Blocking for an extended period waiting for memory can prevent the scheduler thread from making forward progress and so if there are as many behaviours blocked attempting to get memory as there are threads then the entire program will deadlock.

Correctly handling allocation failure in Verona is difficult because the runtime itself needs to allocate memory for any `when` clause and so any response to out-of-memory (a global property) must be local and must not involve explicit or implicit allocation.
Itanium C++ ABI runtimes are required, for example, to pre-allocate the memory for out-of-memory exceptions and to be able to throw an exception without allocation.
This allows at least some threads to make forward progress, as long as they can recover without allocating any memory.

### What do you do when you can't allocate?

When a kernel runs out of memory, typically the first thing that it will try to do is discard memory from caches.
A typical modern virtual memory subsystem has a set of 'clean' pages: ones that have been read from some backing store (e.g. an `mmap`ed file) and not modified.
These can be discarded immediately.
Note that this is a global property, not necessarily visible to the thread that cause memory exhaustion.

The next thing that it will try to do is flush some things to the backing store.
In a system with a unified buffer cache, there is no difference between anonymous memory backed by swap and memory-mapped files.
Both can be written back to some backing store to free physical memory.
This is somewhat more complex for the kind of deployment where we expect to see Verona: cloud services that may not have local storage.
Swapping in general often requires some allocation, swapping to a remote blob store by sending some network messages may require a lot of memory, especially given that everything that is swapped will typically require encrypting.

If this still fails, the kernel will often look for things to kill.
Linux, for example, identifies the process with the most unsaved state of value to the user and kills it.
XNU provides an opt-in mechanism called [Sudden Termination](https://developer.apple.com/documentation/foundation/nsprocessinfo#1651129) where a process can opt into being killed abruptly (the equivalent of `kill -9`) in low-memory scenarios.
This is used a lot on iOS, where applications are expected to enter this state when they are in the background.

Finally, the kernel will report allocation failure to callers and make kernel threads handle the failure.
For example, if a device driver fails to allocate memory during device attachment then it will often deallocate all driver-specific state and return the device to the unattached state.
In more extreme scenarios, a network stack may drop connections in low-memory conditions (not ideal, but better than crashing the kernel and high-level protocols typically have to deal with unreliable networks and so will reconnect).

### How do you avoid running out of memory?

The simplest way of avoiding out-of-memory conditions is common in embedded software: don't allocate memory.
In a typical embedded system, all memory is 'allocated' up front in global variables and so dynamic conditions can never cause memory exhaustion.
This is not a generalisable solution.

In kernel code that holds non-sleepable locks the common idiom is to allocate all memory up-front.
This has the advantage of providing a single failure point.
The bulk allocation either fails or succeeds but if it succeeds then the later code cannot experience out-of-memory conditions.

Some operating systems (for example, NT and XNU as explicit APIs and most UNIX-like systems via `MADV_FREE` or equivalents) provide user code with a discardable or purgeable data abstraction.
These provide a way of requesting memory that the OS may take away if it runs low on memory.
These are typically exposed in higher-level code with an abstraction that reference counts each page and toggles it between the normal and discardable states when the reference count alternates between zero and one.

Android, Windows, and XNU all provide variations on low-memory notifications for userspace.
Inside most kernels there are also thresholds that trigger some of the memory reclaiming behaviour before memory is actually exhausted (for example, when the ratio of clean to dirty pages exceeds some threshold the swapper will write back dirty pages that have not been recently modified and mark them as clean).

Discarded solutions
-------------------

There are several possible approaches that we have discarded as possible approaches.

### `new` returns an option type

The simplest approach is for the `new` operator to return an option type (or throw on failure, which is roughly equivalent in Verona).
This is to memory failure what `malloc` and `free` are to general memory management: a 'solution' by making it 100% the programmer's problem.

This kind of approach may be useful in an OS kernel or embedded system but experience with C/C++ has shown that almost all code that does handle allocation failures does so by exiting the program.
Requiring a checked exception on every allocation would be likely to lead to the same thing: a program littered with explicit termination on allocation failure without any graceful handling.

### Allow pre-allocating `when` closures

Handling an out-of-memory situation in a highly concurrent program is likely to involve communication.
Unfortunately, the act of communication will allocate memory.
In theory, we could allow user code to pre-allocate the space for the message (including closure) that implements a `when` clause.

This would be a very difficult mechanism to surface in user code (the set of captures for the `when` clause that's dispatched must fit in the allocated space and we would need new `when` syntax to take the extra buffer space).
It is also difficult to use because the out-of-memory message may be queued behind others and so would need to be sent to a dedicated cown (which would then not be able to do anything to deallocate state other than its locally owned objects).

### Scheduling a behaviour on low-memory conditions

In theory, we could define a closure to handle low-memory conditions and have the runtime schedule it on a designated cown when memory is exhausted.
This is very difficult to make work correctly because the message may come behind others and may not run immediately.
It also has the same problem as pre-allocated `when` blocks: out-of-memory conditions are global and a `cown` receiving the behaviour may not be able to address them.
Even if scheduled to run on a cown with a lot of state, it my be scheduled while that cown is running a behaviour that allocates a lot of memory and so not be run until after the cown has hit unrecoverable low-memory conditions.

Possible solutions
------------------

There are a few solutions that might be useful.

### Pre-allocate regions

The pattern of pre-allocating objects for a specific task is very easy to map to Verona with a new region kind.
Existing region implementations will call `snmalloc` to allocate either individual objects or chunks, on demand.
It would be easy to extend this to provide a region that pre-allocated sufficient memory for all allocations.
The constructor for the region would need to throw an exception on failure allowing code to handle potential failure of a set of allocations in a single point.

Currently, Verona lacks an equivalent of C's `sizeof` operator and so does not have a way of specifying the amount of memory required in terms of something relevant to the programmer.
It might be possible for the constructor for such a region to be a generic that took a type list.
For example:

```verona
try
{
  var region = PrereservedRegion[Array[SomeCls, 42], SomeBigCls, Array[SomeSmallCls, 1000]];
  allocateABunchOfStuff(region);
}
catch
{
  { OutOfMemory => /* report failure here */ }
}
```

### Low-memory noticeboards

XNU, Android, and Windows all provide some form of low-memory notifications.
The most useful forms of these are when memory is fairly low but allocation remains possible.
These can be used by software to avoid allocation or free existing objects when memory is low but not yet exhausted.
These are particularly useful in garbage-collected systems where it's often beneficial to avoid running the GC until the system starts to experience some memory pressure.

In a Verona world, traced regions are only collected in response to explicit invocation of the collector.
A cown may wish to avoid tracing until memory starts to be constrained.
Signalling a noticeboard and scheduling the cowns that have requested notifications does not require memory allocation and so providing a low-memory noticeboard would allow a cown to perform some cleanup when memory is low, without requiring any additional allocation.

### Sudden-termination cowns

The concept of sudden termination could be extended to `cown`s.
Cowns can already be held by weak references and so this would require a small tweak to the weak reference semantics such that cowns supporting sudden termination are not deallocated until their weak and strong reference count reference counts *both* drop to zero, or until an out-of-memory condition is reached.

When a cown is running, it can opt into sudden termination.
This would add it to a list in the runtime.
When the runtime detects an out-of-memory condition then it would walk the list of cowns that support sudden termination and free any that are not currently strongly referenced.

Cown references are not required to be option types and so there is no way of making cowns that can go away unexpectedly other than by ensuring that all references to it are weak references.
Scheduling a behaviour on a cown implicitly references it and so a cown that supports sudden termination will not be deallocated if it has pending behaviours to process or if it is running.

When a `cown` is deallocated (either explicitly or when killed by low-memory conditions) then it will need to be removed from the list.
This is likely to be a fairly uncommon operation and so it would probably be acceptable to have a simple doubly linked list protected by a lock.

In the current implementation, `cown`s are not deallocated while they still have strong references but all objects that they point to are.
This implementation fits naturally with the requirements for `cown`s that support sudden termination.
Adding them to the sudden termination list (which can be an intrusive list in the `cown` structure if we are willing to add two pointer-sized words to that structure) can implicitly add a strong reference, preventing deallocation of referenced objects, and when the weak count drops to zero and additional check can detect if the only reference is from the sudden-termination list and drop that reference and remove it from the list, triggering finalisation and deallocation.

### Discardable regions

The region abstraction could be used to provide discardable regions, which could have their contents discarded in between behaviour invocations.
The first time such a region is opened, it is locked in memory until the end of the current behaviour but any time between the end of a behaviour running and the first time the region is accessed the runtime may discard it.

Such a region must be held in a field that represents some form of option type.
There are a few ways that this could be surfaced to the programmer.
The fact that regions are isolated means that we could couple the discardable property with the type of the field, for example:

```verona
class HoldsDiscardable
{
	var neverGoesAway : SomeType & iso;
	var discardableThing : (SomeType & iso) | Builtin.Discardable;
};
```

In this possible surface syntax, the act of assigning an `iso` to the second field marks it as discardable and the pattern match on the field to extract the `iso` implicitly marks it as non-discardable until the end of the current behaviour.
Other syntaxes are possible and it may be useful to make the fact that a region is discardable part of the type of the region to allocate extra space in the region metadata for the runtime to use to maintain the list of discardable regions.

The runtime would need to capture both the location of the field containing the discardable region as well as the region itself.
This has similar synchronisation problems to sudden-termination cowns, deallocating a region must be possible either explicitly (unregistering it from the runtime) or by the runtime (with sufficient ordering to ensure that it is not discarded while open).

