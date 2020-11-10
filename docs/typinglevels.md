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

mut         ~> iso          = mut
lifetime(r) ~> iso          = Top
iso         ~> iso          = mut
imm         ~> iso          = imm

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