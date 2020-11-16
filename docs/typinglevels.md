# Typing levels

This document describes factoring the type system into multiple levels.  This 
is based on the thought experiment in [alternativesemantics.md](./alternativesemantics.md)
that was considering putting more checking onto the runtime.  That idea liberates us to consider
type systems to remove some but not all of the runtime requirements.

This document is considering factor the type system into three levels

1. Structural and algebraic types with linear `iso`
2. Statically preventing cross region references
3. Statically preventing references into regions to enable strict semantics of operations.


### Linearity of isos.

**[The type system for ensuring linearity of `iso`s, and standard algebraic and structural 
types should go here]**

View point adapation

Mut and iso

### No cross region references

The only aim of this type system is to prevent cross region pointers. 
In this proposal, we introduce a new command for the IR, `open`, and a new type, `lifetime(r)`.
They are connected with the following type rule
```
//  x: tau
r = open x
//  x: tau & lifetime(r)
```
We can always get an identifier for the lifetime/region.  This can be added liberally to any field read or extraction.  We can have multiple `lifetime` annotations they semantically just mean there are many names for the region:

> If `lifetime(r) & lifetime(r')` is inhabited, then `r = r'`.


They interact with the view point operations as follows:
```
mut         ~> mut          = mut
lifetime(r) ~> mut          = lifetime(r)
iso         ~> mut          = mut
imm         ~> mut          = imm

*           ~> lifetime(r)  = undefined // lifetime not part of field annotations

mut             ~> iso          = mut
lifetime(r)     ~> iso          = derived_from(r)
derived_from(r) ~> iso          = derived_from(r)
iso             ~> iso          = mut
imm             ~> iso          = imm

*           ~> imm          = imm
```
The two most important operations are reading a `mut` fields preserves the lifetime, 
and reading an `iso` field does not.

This is the same for extracting.  The key difference is whether you get an `iso` or a `mut` when you read an `iso` field:
```
mut         <~ mut          = mut
lifetime(r) <~ mut          = lifetime(r)
iso         <~ mut          = mut
imm         <~ mut          = imm

*           <~ lifetime(r)  = undefined // lifetime not part of field annotations

mut         <~ iso          = iso
lifetime(r) <~ iso          = Top
derived_from(r) <~ iso      = Top
iso         <~ iso          = iso
imm         <~ iso          = imm

*           <~ imm          = imm
```

[Paul: needs two forms of extracting view point adaptation :-( ]


Field extraction requires same `lifetime`.
```
G(x) : tau_x
G(y) : tau_y
field_write(f, tau_x) = tau_f_w
field_read(f, tau_x) = tau_f_r
tau_y <: (tau_x & lifetime(r)) <~ tau_f_w     (fresh r)
tau_x <: mut | iso
--------------------------------------
G |- z = (x.f = y) -| G, z: tau_x <~ tau_f_r, y: tau_x ~> tau_y

```

By adding `, y: tau_x ~> tau_y` to the output typing, we can maintain `y`s access, but without the ownership of an `iso`.


The trick here is that `& lifetime(r)` forces `x` to have a lifetime.  If it doesn't already have one, then it is impossible for `tau_y` to have the same one. E.g.
```
  tau_x = mut & lifetime(r1) & D
  tau_f_w = mut & C
  tau_y = mut & lifetime(r1) & C
```
Then the subtyping query is
```
  mut & lifetime(r1) & C <: (mut & lifetime(r1) & D & lifetime(r)) <~ (mut & C)
= mut & lifetime(r1) & C <: mut & lifetime(r1) & lifetime(r) & C
```
Now, any inhabitant of the left-hand-side is also an inhabitant of the right. Yay!

But if we didn't have a `lifetime` for `x`, then we would have something like
```
  tau_x = mut & D
  tau_f_w = mut & C
  tau_y = mut & lifetime(r1) & C

  mut & lifetime(r1) & C <: (mut & D & lifetime(r)) <~ (mut & C)
= mut & lifetime(r1) & C <: mut & lifetime(r) & C
```
Which is not true for any interpretation where `r1` doesn't equal `r`, so subtyping does not hold.



So for the larger example this would be
```
  // z: lifetime(r) & mut
  x = z.f_maybe_iso;
  // x: (mut & lifetime(r)) ~> (iso&C | mut&D)
  // x: ((lifetime(r)~>iso) & mut & C) | (lifetime(r) & mut & D)
  match x 
  | iso =>
      // x: (lifetime(r)~>iso) & mut & C)
      // But can't use unless we open
      r' = open x
      // w: lifetime(r') & (lifetime(r)~>iso) & mut & C
      // w is a different region to x and z, so can't assign into previous region
      //  z.h_mut = w  would not be allowed as different region

  | mut =>
      // x: lifetime(r) & mut & D
      z.g_mut = x
```

What about open on non-iso state?

```
// x: mut & tau
r' = open x
// x: mut & lifetime(r') & tau
```

``` 
// x: imm & tau
r' = open x
// x: imm & tau & lifetime(r')
```

Need to be careful about this, but I think it works.



### Enabling strict operations

**[Thinking required to fill in this section]**

Paul's example:

```ts 
m(x: mut)

// x: mut & lifetime(r)
{
  var y = new B;
  // y : iso & B & lifetime(r')
  // r' : ~r
  // r # r'
  var z = y.f_mut;
  // z: mut & C & lifetime(r')
  drop y
  // drop r'
  // use x okay
  // use z bad
}

```


{ Eq  }
var x = (y.f = z);
{ Eq + y~z~x }

{ Eq }
var x = y.f;
{ Eq + x~y }

{ Gamma, Eq }
  filter_mut (y)
{ Gamma', Eq' }
  Gamma' = { x |-> Gamma(x)[noaccess/mut] | x~y\in Eq }  u { x |-> Gamma(x) | x~y\notin Eq }
  Eq' = Eq \ { x~y | x~y \in Eq}

filter_mut y



drop y

// x: lifetime(r), 
pin (r)
{
  
}

// y : mut & lifetime(r) & derived_from(r1)
var x_mut = y.f_iso;
// x_mut: mut & lifetime(r') & derived_from(r)
var z_mut = y.f_mut;
// z_mut: mut & lifetime(r)
var w_mut = y.f_iso1;
// w_mut: mut & lifetime(r'') & derived_from(r)
var x_iso = (y.f_iso = z)
// x_iso: iso & derived_from(r)
- invalidate:
  - x_mut: yes
    - x_mut~y and x_iso~y therefore x_mut~x_iso
    - x_mut: lifetime(r'), r' \notin {r}
  - z_mut: no
    - z_mut: lifetime(r), r \in {r}
  - y: no
    - y: lifetime(r), r \in {r}
  - w_mut: yes, unnecessarily
    - w_mut~y and x_iso~y therefore w_mut~x_iso
    - w_mut: lifetime(r''), r'' \notin {r}
    - but really, r'' guaranteed not to be r', but we don't know this

// x: derived_from(rbar)
filter(x)
  - Invalidate any y, y~x \in Eq unless y:lifetime(r) /\ r \in rbar
  - Invalidate means G[y |-> G(y)[true/mut]]
  - Not quite right: invalidate disjunctions separately


// x_mut: mut & lifetime(r) & derived_from(r')
// y_mut: mut & lifetime(r')
f(x_mut)
// `filter` each argument 

// y: lifetime(r0)
// y': lifetime(r1)
// x_mut: mut & lifetime(r2) & derived_from(r0)
var x_mut = y.f_iso;
// x: iso
var x = y.f_iso = z;
y'.f_iso_or_imm = x;
  // x_mut is now a cursor into something derived from r1, not r0

  // Only drop mut, in other regions, lifetime(r)
var x = (y.f_iso = z);
  // z is 

        Gamma, y:mut |- e -| Gamma'
------------------------------------
Gamma, x: iso , y~x \in Eq |- e -| x:iso, Gamma'


So Eq, is actually may alias on regions...  

But also need iso?


### Formal strict

They interact with the view point operations as follows:
```
mut(rbar_d; rbar_c)   ~> mut          = mut(rbar_d; rbar_c)
iso(rbar)             ~> mut          = mut(rbar; emptyset)
imm                   ~> mut          = imm

*           ~> lifetime(r)  = undefined // lifetime not part of field annotations

mut(rbar_d; rbar_c) ~> iso          = mut(rbar_d u rbar_c; emptyset)
iso(rbar_d)         ~> iso          = mut(rbar_d; emptyset)
imm                 ~> iso          = imm

*           ~> imm          = imm
```
The two most important operations are reading a `mut` fields preserves the lifetime, 
and reading an `iso` field does not.

This is the same for extracting.  The key difference is whether you get an `iso` or a `mut` when you read an `iso` field:
```
mut         <~ mut          = mut
lifetime(r) <~ mut          = lifetime(r)
iso         <~ mut          = mut
imm         <~ mut          = imm

*           <~ lifetime(r)  = undefined // lifetime not part of field annotations

mut         <~ iso          = iso
lifetime(r) <~ iso          = Top
derived_from(r) <~ iso      = Top
iso         <~ iso          = iso
imm         <~ iso          = imm

*           <~ imm          = imm
```

[Paul: needs two forms of extracting view point adaptation :-( ]


Field extraction requires same `lifetime`.
```
G(x) : tau_x
G(y) : tau_y
field_write(f, tau_x) = tau_f_w
field_read(f, tau_x) = tau_f_r
tau_y <: (tau_x & lifetime(r)) <~ tau_f_w     (fresh r)
tau_x <: mut | iso
--------------------------------------
G |- z = (x.f = y) -| G, z: tau_x <~ tau_f_r, y: tau_x ~> tau_y

```


##  Example

```
processMessage(
    message: iso
    latest_data: cown/iso)
{
    var hash_table = latest_data.map;
    // 
    for each (update in message)
    {
        
    }
}

```




## ADT predicates

```ts

D |- P N
--- [any]
D |- any P N

D |- any P T1 \/ D |- any P T2
--- [any-isect]
D |- any P (T1 & T2)

D |- any P T1 \/ D |- any P T2
--- [any-union]
D |- any P (T1 | T2)

D |- P N
--- [all]
D |- all P N

D |- all P T1 \/ D |- all P T2
--- [all-isect]
D |- all P (T1 & T2)

D |- all P T1
D |- all P T2
--- [all-union]
D |- all P (T1 | T2)

```

## Type predicates

```ts

--- [mutable]
D |- mutable (owned_r | free_r)

```

## Silly Generic Example

```ts

f[T](x: T & imm)
  where T: I
{
  // send x
}

f[C & mut](x);

```





// w: iso
x = w.f_mut
// x: lifetime(r) & mut
y = x.f_iso
// y : derived_from(r)
z = (y.f_iso = w)
// y : derived_from(r)
// z : derived_from(r)


We need to `filter` on `w` and `y` I think.  
  `w` is required to prevent cycles?  It would drop `x` and `y`, so can't keep cursor into it. Only required if the Eq says they may be in the same region.



So need to lose derived_from facts too

x1 = x0.f_iso
r1 = open x1
x2 = x1.f_iso
r2 = open x2
x3 = (x2.f_iso = None)
// x3 : df(r1,r2) 
x4 = x3.f_iso
// x4 : df(r1,r2)
_ = (x4.f_iso = x1)
// x3: df(r1,r2) and x4: df(r1, r2)  should not still be true

## Itertools

```ts

class Option[T]
{
  class EmptyMarker
  {
    static create(): Empty { new () }
  }

  type Empty = EmptyMarker & imm;

  private value: T | Empty;

  ?(self: Option[T] & (mut | imm)): Bool
  {
    // ?opt, opt ? (), opt.?(), ?(opt)
    match (self.value)
    {
      let v: S~>T { true }
      _ { false }
    }
  }

  *[S](self: S): S~>T throws Empty
    where S: Option[T] & (mut | imm)
  {
    // *opt, opt * (), opt.*() *(opt)
    match (self.value)
    {
      let v: S~>T { v }
      _ { throw Empty }
    }
  }

  |[S, U](self: S, def: U): S~>T | U
    where S: Option[T] & (mut | imm)
  {
    // opt | def, opt.|(def), |(opt, def)
    match (self.value)
    {
      let v: S~>T { v }
      _ { def }
    }
  }

  <-[S, U](self: S, value: S<~T | Empty = Empty, def: U = Empty): S<~T | U
    where S: Option[T] & mut
  {
    // <-opt, opt <- (), opt.<-(), <-(opt)
    // opt <- val, opt.<-(val) <-(opt, val)
    // opt <- (val, def), opt.<-(val, def), <-(opt, val, def)
    match (self.value = value)
    {
      let v: S<~T { v }
      _ { def }
    }
  }
}

```
