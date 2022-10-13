# Wait-free Weak References for Cowns

## Problem statement

[Reference counting](https://en.wikipedia.org/wiki/Reference_counting) can suffer from not being able to collect cycles.
A common approach to dealing with this in reference counted systems is to introduce a concept of *Strong* and *[Weak](https://en.wikipedia.org/wiki/Weak_reference)* references.
A strong reference keeps its target live, while a weak does not.
A weak reference can be used to access its target if it is still alive.
The challenge with any system is safely promoting a weak reference to a strong reference in the presence of concurrency.
To achieve this past implementations (for examples, [Rust's ARC](https://doc.rust-lang.org/src/alloc/sync.rs.html#1979)) use a CAS loop to ensure an atomic creation of the strong reference from a weak. 
This means the implementation is lock-free and not wait-free. 
Hence, under contention the promotion can be arbitrarily delayed. 

This document describes Verona's approach to reference counting for cowns, 
which provides a wait-free implementation of all the operations for  both strong and weak references.


The rest of the document assumes a familarity with strong and weak references.


# API
In what follows, we will use a separation logic style to present the correct usage of the API. 

We will assume the client of this API will provide two function
```
  { NoStrong * WeakRef}
    destructor()
  { emp }

  { NoWeak }
    deallocator()
  { emp }
```
These two function represent the two stage reclaimation of an object with strong and weak references,
the first is called when their are no more strong references in the system, and the second is called
when all the strong and weak reference have gone.


If you have a `StrongRef`, then you can create a second one:
```
  { StrongRef }
  acquire_strong()
  { StrongRef * StrongRef }
```

If you have a `WeakRef`, then you can create a second one:
```
  { WeakRef }
  acquire_weak()
  { WeakRef * WeakRef }
```

If you have a `StrongRef`, then you can drop it:
```
  { StrongRef }
  release_strong()
  { emp }
```
If this was the last `StrongRef`, then it will internally call the `destructor` and possibly the `deallocator` if there are no `WeakRef`s. 

Similarly, we can drop a `WeakRef`:
```
  { WeakRef }
  b = release_weak() 
  { emp }
```
If this was the last `WeakRef` that was dropped, then it will call the `deallocator`.


Finally, we have two operations for converting between `StrongRef` and `WeakRef`:
```
  { StrongRef }
  acquire_weak_from_strong()
  { WeakRef * StrongRef }

  { WeakRef }
  b = acquire_strong_from_weak()
  { WeakRef * if b { StrongRef } }
```
The first is guaranteed to succeed. While, the second can fail if all the strong references have been removed. 

# Implementation

The key challenge this implementation addresses is to make all the operations "wait-free", they are guaranteed to terminate in finite time not mater what interference comes from other threads.
This is not typically the case for `acquire_strong_from_weak` that normally involves a `CAS` loop, and can only be described as "obstruction-free", which is a much weaker property. 

We will assume three global variables:
* `rc` - the strong reference count
* `wrc` - the weak reference count
* `closed` - the strong reference count is now closed.

In the actually implementation, `rc` and `closed` are combined into a single machine word.
We signify atomic blocks by enclosing them inside `<` and `>`.

The implementation of `acquire_strong`, `acquire_weak`, `release_weak`, `acquire_weak_from_strong` are all straight forward:

```
void acquire_strong()
{
  <rc++>
}

void acquire_weak()
{
  <wrc++>
}

void release_weak()
{
  <wrc--; last = (wrc == 0);>
  if (last)
    deallocator();
}

void acquire_weak_from_strong()
{
  <wrc++>
}
```

The core challenge is their is no machine instruction that can effectively do
```
< old = rc--; if (old == 1) { closed = true } >
```
Instead, we must separate the operation into two instructions, the first releases the reference count using an atomic decrement, and the second
applies the `closed` using a compare-and-set operation.  
```
  <r = rc-->
  // A
  < b = (!closed && rc == 0); if b (closed := true) > // B
```
On line `//A` it is possible for other threads to successfully perform `acquire_strong_from_weak`, and thus multiple threads can be a attempting to perform `//B`, thus it needs to be a CAS that checks it has not already been applied.
Unfortunately, this also means that the underlying storage for the reference count could potentially be deallocated.  We need any thread at location `//A` to effectively hold a weak reference to prevent this deallocation.

To achieve this, we make any `acquire_strong_from_weak` effectively relinquish its `WeakRef` if it moves the strong reference count up from 0, when the closed is not set.
This ensures that if there are `n+1` threads at `//A` and the `rc==0`, then we have `n` weak references protecting there access, and if `rc != 0`, then we have `n+1` weak references protecting their access.


```
void release_strong()
{
  <r = rc-->

  if (r!=0) return

  // Try to set the closed
  < b = (!closed && rc == 0); if b (closed := true) > // CAS

  if (b)
    destructor();

  release_weak();
}

bool acquire_strong_from_weak()
{
  < closed,old = rc++ >
  if (closed)
    return false;

  if (old == 0)
    acquire_weak()

  return true;
}
```

# Starling based proof

A starling based proof can be found in [verona_rc_wrc.cvf](./verona_rc_wrc.cvf).