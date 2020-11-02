# Alternative semantics for Verona.

## Abstract

This document is a thought experiment in how runtime support can reduce the complexity of the type system.  The key insight is to delay certain operations to a point where we can trivially know that there is no aliasing.  Aliases into a region are difficult to track accurately at compile time, by relaxing the semantics of when a programmer can assume an event occurs allows for a simpler type system.

## Motivation

The current design of Verona requires accurate tracking of the 
subregion relationship to enable the invalidation of references for safety.
For instance, if we have a class
```
class Foo
{
    f: iso & Bar;
    ...
}
```
and an access to its `f` field such as
```
// x: Foo & mut
var y = x.f;
```
then we need to track that `y` has come from a subregion of `x`.
Certain operations on `x`'s region must also invalidate `y`.
If we drop the region containing `x`, then that will also deallocate the subregion, and thus `y` would be dangling if we did not invalidate it.

The same is true for a `when` that captures a region in its closure:
```
var a = new C
a.next = new D
// a : iso & C
var b = a.next;
// a : iso & C, b : mut & D
when (c) {
    //do something with a 
}
// Use b
```
Here `a` is sent to the `cown` `c` by the runtime.  Once that message is sent, we must also invalidate the references into subregions. The `use b` after the
`when` is invalid as it could be in parallel with the `do something with a`.
This leads to us having to track subregion information precisely, so that we know that `a` being captured by the `when` invalidates `b`.

We have explored various restrictions to reduce the required precision, but not found a solution that we can agree upon.  A key requirement is any subregion information must be expressible as a method signature otherwise we will not be able to outline code.  This however makes the design visible and constrains future extensions.

The core reason for tracking this information precisely is that dropping a region, or sending a message, happens immediately.  If we alter the semantics to dropping a region or sending a message to happen once there are no interior pointers into the region or subregions of that region, then we have a different programming model.

We know that once a behaviour completes then it will have no interior references and thus it can send all its messages, and deallocate all the dropped regions, thus it will be safe to deallocate the regions and send the messages. 
Delaying message send and deallocation until the behaviour completes may reduce concurrency and increase memory pressure.
However, this does not require any clever type system or analysis to make this work, and we would not have to track subregion information.
We could specify the semantics as messages are only sent at the end of a behaviour, but that would constrain future optimisations.
If the semantics simply states that messages can be sent once a behaviour is guaranteed not to access the region (and subregions), then we open the possibility of adding a better static type system, or compiler optimisation passes at a future point.

## Proposal

We change the semantics of the language in the following two ways:

>  Message send happens after all the internal references to the region tree have been removed.

and

> Deallocation of a region happens after all internal references to the region have been removed.

The slight difference allows for a region to be deallocated, but for the sub-regions deallocation to be delayed longer.

**[Luke, Paul, This might lead to an interesting formal semantics. I think it breaks the current thinking on keeping the region structure and the cown concurrency separate.]**

## Type system

```
   Capabilities ::= iso | mut | imm
   tau : Types ::= iso  |  mut  |  imm  |  tau|tau  |  tau&tau  |  r  |  I  |  C
     |  tau ~>_r tau  |  tau <~_r tau  |  X
```
where 
   `C` are class names
   `I` are interfaces
   `r` are region variables
   `X` are type variables

Most of the syntax of types is standard, but  `tau ~>_r tau` and `tau <~_r tau` are special region operations, known as "view point adapation"s.  There are two types aliasing, `~>`, and extracting `<~`.

We restrict types on fields and generic parameters to not mention `r`, i.e. they cannot contain `r`, `tau ~>_r tau` and `tau <~_r tau`.

**[Sylvan is this too restrictive]**

We interpret the semantics of types with 
```
[[ _ ]] : 
    (region var -> region) 
        -> (type var -> P(C, region, cap)) 
            -> P(C, region, cap)
```
where
```
  [[ iso ]](R,T) = { C,r,iso | C \in Class, r \in region}
  [[ r ]](R,T) = { C,R(r),cap | C \in Class, cap \in Caps }
  [[ C ]](R,T) = { C,r,cap | r \in region, cap \in Caps }
  [[ X ]](R,T) = T(X)
  [[ tau ~>_r tau ]](R, T) = 
    { V1 ~>_R(r) V2 | V1 \in [[tau1]](R,T) /\ V2 \in [[tau2]](R,T)}
  [[ tau <~_r tau ]](R, T) = 
    { V1 <~_R(r) V2 | V1 \in [[tau1]](R,T) /\ V2 \in [[tau2]](R,T)}
```

Aliasing viewpoint adaptation
```
C1,r1,mut ~>_r C2,_,mut = C2,r1,mut
C1,r1,iso ~>_r C2,_,mut = C2,r1,mut
C1,r1,imm ~>_r C2,_,mut = C2,r1,imm
C1,r1,mut ~>_r C2,_,iso = C2,r,mut
C1,r1,iso ~>_r C2,_,iso = C2,r,mut
C1,r1,imm ~>_r C2,_,iso = C2,r1,imm
C1,r1,mut ~>_r C2,_,imm = C2,r1,imm
C1,r1,iso ~>_r C2,_,imm = C2,r1,imm
C1,r1,imm ~>_r C2,_,imm = C2,r1,imm
```
**[Could be a relational definition to allow for unrestricted r on the right hand side.]**

Extracting viewpoint adaptation
```
C1,r1,mut <~_r C2,_,mut = C2,r1,mut
C1,r1,iso <~_r C2,_,mut = C2,r1,mut
C1,r1,imm <~_r C2,_,mut = C2,r1,imm
C1,r1,mut <~_r C2,_,iso = C2,r,iso
C1,r1,iso <~_r C2,_,iso = C2,r,iso
C1,r1,imm <~_r C2,_,iso = C2,r1,imm
C1,r1,mut <~_r C2,_,imm = C2,r1,imm
C1,r1,iso <~_r C2,_,imm = C2,r1,imm
C1,r1,imm <~_r C2,_,imm = C2,r1,imm
```

The most important difference between extracting and aliasing
viewpoint adaption is how they behave on `iso` fields. Extracting provides an `iso` in the supplied region, and aliasing provides a `mut` in the supplied region.

The definitions take a region parameter so that "freshness" can be suitably defined. 

### Type Rules

### Read versus Write types for a field
```
   field_read(tau1 | tau2, f) = field_read(tau1, f) | field_read(tau2, f)
   field_read(tau1 & tau2, f) = field_read(tau1, f) & field_read(tau2, f)
   field_write(tau1 | tau2, f) = field_write(tau1, f) & field_write(tau2, f)
   field_write(tau1 & tau2, f) = field_write(tau1, f) & field_write(tau2, f)
```
]

### Aliasing assignment generates a fresh region
if it follows an `iso`. If it follows an `imm` then it is in the immutable space, otherwise it is in the same region. 

```
field_read(f, tau_x) = tau_f
fresh r
-------------------------------------
G |- z = x.f -| G, z: tau_x ->r tau_f
```

### Extraction uses a fresh region
```
G(x) : tau_x
G(y) : tau_y
field_write(f, tau_x) = tau_f_w
field_read(f, tau_x) = tau_f_r
exists r'. tau_y <: tau_x <~r' tau_f_w
fresh r
--------------------------------------
G |- z = (x.f = y) -| G, z: tau_x <~r tau_f_r
```
Is y still accessible?  If it was an iso, then no.  But otherwise, yes.  So remove y, if we can prove it was not an iso?


### Method call?

No longer has to invalidate any regions, may consume locals of type `iso`.
```
  foo (x: iso&C) {...}

  var y = new C
  foo (y)
  foo (y) /// Error `y` was invalidate by previous line.
```

**[Type rule goes here]**

##  Work required

* Runtime support for delaying reclaimation
* Runtime support for delaying message send
* Feasability work for optimisations 
    - What can we optimise?
    - What user annotation might we provide?
    - 
* Type system implementation 
* Optimisation

## Concerns

### Additional root sets

Not tracking regions and subregions precisely means we may need to be more careful with codegen for the additional root set.  For instance, we can use have the same region multiple times, which means we use the region additional roots in a globally stack based way.  I.e.
```
  r1.add_root(a)
  r2.add_root(b)
  ...
  r1.remove_root(a)
  ...
  r2.remove_root(b)
```
Is not allowed as if r1 and r2 are the same region, then we will not be following the stack discipline.

**[This may already be the case with cown aliasing, where we don't know if two cowns are the same or different.]**

### Performance

This approach can cause higher memory usage and less concurrency of messages.  This may be bad for performance.

Using compiler optimisations and global analysis could mitigate the default performance by finding cases where messages can be sent promptly.

This however leads to hard to predict performance for the programmer.  We can mitigate this by providing annotations the programmer provides to guarantee certain properties must be true for optimisation, but then we are effectively implementing the complex type system.  This may still be acceptable as the barrier to entry is lower, and the performance work will require a more specialist view.  We can also potentially forbid more in the perf setting, allowing programmers to trade off flexibity of implementation with performance.

A correct design here will take time and experience.
This proposal can be adopted without resolving this design, but we should be careful not to restrict future optimisations and annotations.