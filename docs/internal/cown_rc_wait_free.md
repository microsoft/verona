# Wait-free Weak References for Cowns

## Problem statement

[Reference counting](https://en.wikipedia.org/wiki/Reference_counting) can suffer from not being able to collect cycles.
The common approach to dealing with this in reference counted systems is to introduce a concept of *Strong* and *[Weak](https://en.wikipedia.org/wiki/Weak_reference)* references.
A strong reference keeps its target live, while a weak does not.
A weak reference can be used to access its target if it is still alive.
The challenge with any system is safely promoting a weak reference to a strong reference in the presence of concurrency.
To achieve this past implementations (for examples, [Rust's ARC](https://doc.rust-lang.org/src/alloc/sync.rs.html#1979)) use a CAS loop to ensure an atomic creation of the strong reference from a weak. 
This means the implementation lock-free and not wait-free. 
Hence, under contention the promotion can be arbitrarily delayed. 

## This document

This document describes Verona's approach to [reference counting](https://en.wikipedia.org/wiki/Reference_counting) for cowns.
Verona uses both strong and [weak reference](https://en.wikipedia.org/wiki/Weak_reference)s.
This document assumes a familarity with those concepts.  



# API

In what follows, we will use a separation logic style to present the correct usage of the API. 

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
  b = release_strong()
  { (b * NoStrong) | ¬b }
```
If this was the last `StrongRef`, then it will return `true`, otherwise it returns `false`. In the specification, we use `NoStrong` to represent there are no other strong references in the system.

Thus, we say that it is not possible for `NoStrong` to coexist with `StrongRef`:
```
NoStrong * StrongRef => False
```

Similarly, we can drop a `WeakRef`:
```
  { WeakRef }
  b = release_weak() 
  { (b * NoWeak) | ¬b }
```
If this was the last `WeakRef` that was dropped, then it will return `true`, otherwise it returns `false`.  In the specification, we use `NoWeak` to represent there are no other strong or weak references in the system.
```
  WeakRef * NoWeak => False
  StrongRef * NoWeak => False
```

To make use of both `NoWeak` and `NoStrong`, we need some additional axioms that the library provides. Firstly, there is only one `NoWeak` or `NoStrong`:
```
NoStrong * NoStrong => False
NoWeak * NoWeak => False
NoWeak * NoStrong => False
```
You can effectively view `NoStrong` as the permission to run the destructor of an object and thus release its held references, and `NoWeak` as the permission to actually deallocate the object.

We provide an additional axiom to convert `NoStrong` into a `WeakRef`:
```
NoStrong => WeakRef
```
[This is sometimes called a view shift.  It consumes the ownership of `NoStrong` and provides a `WeakRef`.]
This axiom should really be integrated with the destructor in a larger scale proof.  It can be seen as representing the permission to call the destructor.


Finally, we have two operations for converting between `StrongRef` and `WeakRef`:
```
  { StrongRef }
  acquire_weak_from_strong()
  { WeakRef * StrongRef }

  { WeakRef }
  b = acquire_strong_from_weak()
  { ((b * StrongRef) | ¬b) * WeakRef }
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

bool release_weak()
{
  bool b;
  <b = (wrc-- == 1)>
  return b
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

To achieve this, we make any `acquire_strong_from_weak` effectively relinquish its `WeakRef` if it moves the strong reference count up from 0, when the closed is not set.  This ensures that if there are `n+1` threads at `//A` and the `rc==0`, then we have `n` weak references protecting there access, and if `rc != 0`, then we have `n+1` weak references protecting their access.


```
bool release_strong()
{
  <r = rc-->
  if (r!=0)
    // Not the last.
    return false
  // Try to set the closed
  < b = (!closed && rc == 0); if b (closed := true) > // CAS
  if (~b)
    // Failed to set closed, `acquire_strong_from_weak` has provided `WeakRef` to protect access
    // Release the weak reference.
    release_weak()  // B
  return b
}

bool acquire_strong_from_weak()
{
  < closed,old = rc++ >
  if (closed)
    // Mark has been set we failed.
    return false;
  if (old == 0)
    // We have interrupted a `release_strong`.
    // Thus, we have relinquished our weak reference.
    // Acquire a new one to meet the specification.
    acquire_weak()
  return true;
}
```

On complexity the real implementation has to handle is that the `release_weak` at `// B` can actually be the final deallocation of the object. This requires a more complex return that we give above.


# Starling based proof

A starling based proof can be found in [verona_rc.cvf](./verona_rc.cvf).