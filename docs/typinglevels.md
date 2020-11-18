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
lifetime(r)     ~> iso          = Top
iso             ~> iso          = mut
imm             ~> iso          = imm

*           ~> imm          = imm
```
The two most important operations are reading a `mut` fields preserves the lifetime, 
and reading an `iso` field does not.

This is the same for extracting.  The key difference is whether you get an `iso` or a `mut` when you read an `iso` field:
```
mut             <~ mut          = mut
lifetime(r)     <~ mut          = lifetime(r)
iso             <~ mut          = mut
imm             <~ mut          = imm

*           <~ lifetime(r)  = undefined // lifetime not part of field annotations

mut             <~ iso          = iso
lifetime(r)     <~ iso          = Top
iso             <~ iso          = iso
imm             <~ iso          = imm

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

We have two core concepts,
* A type annotation `dervied_from(r)` meaning that this mutable reference is under the region `r`.
* Aliasing relation, Eq, that represents which variables can potentially alias the same region.

We extend the previous operations to add `derived_from` facts, when we follow `iso` fields with 
the aliasing viewpoint adapation. 
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
The extracting adaptation throws away the `derived_from` information
as it severs the link.
```
mut             <~ mut          = mut
lifetime(r)     <~ mut          = lifetime(r)
derived_from(r) <~ iso          = derived_from(r)
iso             <~ mut          = mut
imm             <~ mut          = imm

*           <~ lifetime(r)  = undefined // lifetime not part of field annotations

mut             <~ iso          = iso
lifetime(r)     <~ iso          = Top
derived_from(r) <~ iso          = Top
iso             <~ iso          = iso
imm             <~ iso          = imm

*           <~ imm          = imm
```
We define a concept `filter` that removes `mut`able references that could be a problem for consuming.
```
filter(x, Gamma)  = Gamma'
  Define rs_x s.t. Gamma(x) <: derived_from(rs)
  Define r_x s.t. Gamma(x) <: lifetime(r)
  Where forall y, Gamma'(y) = Gamma(y)[(~lifetime(r) & ~lifetime(rs_x_1) & ... & ~lifetime(rs_x_n)) | mut)/mut] 
```

Predicate form:
```
filter(x, Gamma)  = 
  Define rs_x s.t. Gamma(x) <: lifetime(r) & derived_from(r_1) & ... & derived_from(r_n)
  Define r_x s.t. Gamma(x) <: 
  Requires forall y, 
    Gamma(y) & mut <: lifetime(r) | lifetime(rs_x_1) | ... | lifetime(rs_x_n)
```

Filtering is used to ensure that internal references to a region that could be consumed are removed before that point.

```
filter(x,Gamma)
Gamma(x) <: iso
-------------------------------
Gamma |- drop(x) -| Gamma \ x
```

This is however very restrictive as it means completely unrelated regions mutable references must be dropped.

We track an over-aproximation of the may-alias set in the system to account for this.
[TODO - How does this relate to types.  Can types improve it?]

The second core aspect is the potential aliasing relation: MayBeSameRegionTree. If x~y in MayBeSameRegionTree, and 
'x' is an iso or mut, and 'y' is a mut, then 'x' and 'y' can refer to the same region.
```
{ MayBeSameRegionTree  }
var x = (y.f = z);
{ MayBeSameRegionTree + y~z~x }

{ MayBeSameRegionTree }
var x = y.f;
{ MayBeSameRegionTree + x~y }
```
We always close MayBeSameRegionTree with reflexivity, sym, and trans to form an equivalence relationship.


We can adapt filter to this
Predicate form:
```
filter(x, Gamma, MayBeSameRegionTree)  = 
  Define rs_x s.t. Gamma(x) <: lifetime(r) & derived_from(r_1) & ... & derived_from(r_n)
  Define r_x s.t. Gamma(x) <: 
  Requires forall y, y~x\in MayBeSameRegionTree => 
    Gamma(y) & mut <: lifetime(r) | lifetime(rs_x_1) | ... | lifetime(rs_x_n)
```


We can drop
```
filter(x, Gamma, MayBeSameRegionTree)
Gamma(x) <: iso
-------------------------------
MayBeSameRegionTree; Gamma |- drop(x) -| MayBeSameRegionTree \ x; Gamma \ x
```

Field extraction requires same `lifetime`.
```
G(x) : tau_x
G(y) : tau_y
field_write(f, tau_x) = tau_f_w
field_read(f, tau_x) = tau_f_r
tau_y <: (tau_x & lifetime(r)) <~ tau_f_w     (fresh r)
tau_x <: mut | iso
filter(x, Gamma, MayBeSameRegionTree)
MayBeSameRegionTree(x,y) => filter(y, Gamma, MayBeSameRegionTree)
tau_z =  tau_x <~ tau_f_r
If (tau_z <: iso) then MayBeSameRegionTree' = MayBeSameRegionTree + (x~y) else MayBeSameRegionTree' = MayBeSameRegionTree+(x~y~z)
--------------------------------------
MayBeSameRegionTree; G |- z = (x.f = y) -| G, z: tau_x <~ tau_f_r, y: tau_x ~> tau_y; MayBeSameRegionTree' 

```

If y is mut, then must have the same liftime, so filter will keep them both.

If y is iso and x is a mut under y, then y and x must MayBeSameRegionTree.
This means that `filter(y, Gamma, MayBeSameRegionTree)` will remove `x`. So we can't create a cycle.

We only need to say `z` aliases `x` and `y` if it is not guaranteed to be an iso.

Field Alias
```
G(x) : tau_x
field_read(f, tau_x) = tau_f_r
tau_x <: iso|mut|imm               // alright to read
------------------------------------------
MayBeSameRegionTree, G |- z = x.f -| G, z : tau_x ~> tau_f_r;  MayBeSameRegionTree + (z~x)
```

Thinking through some uses:
```
// y : mut & lifetime(r) & derived_from(r1)    ;  
var x_mut = y.f_iso;
// x_mut: mut & lifetime(r') & derived_from(r) & derived_from(r1)   ;  y~x_mut
var z_mut = y.f_mut;
// z_mut: mut & lifetime(r) & derived_from(r1)                      ;  y~x_mut~z_mut
var w_mut = y.f_iso1;
// w_mut: mut & lifetime(r'') & derived_from(r) & derived_from(r1)  ;  y~x_mut~z_mut~w_mut
var x_iso = (y.f_iso = z)
// x_iso: iso & derived_from(r)                                     ;   y~x_mut~z_mut~w_mut,   x_iso
- invalidate:
  - x_mut: yes
    - x_mut~y 
    - x_mut: lifetime(r'), r' \notin {r,r1}
  - z_mut: no
    - z_mut: lifetime(r), r \in {r,r1}
  - y: no
    - y: lifetime(r), r \in {r,r1}
  - w_mut: yes, unnecessarily
    - w_mut~y
    - w_mut: lifetime(r''), r'' \notin {r,r1}
    - but really, r'' guaranteed not to be r', but we don't know this
```

So what does function call do? filter each argument?
```
// x_mut: mut & lifetime(r) & derived_from(r')
// y_mut: mut & lifetime(r')
f(x_mut)
// `filter` each argument 
```

## Packet example
This example is a simpler approach to combining a bunch of packets into a single message and processing it.
```
recv_packet(packet: Array[byte] & iso)
{
    when (assembler)
    {
        // packet: Array[byte] & iso
        // assembler: mut & lifetime(r)     ; assembler,   packet
        header = get_header(packet);
        // header: mut ;  header~packet

        var packet_list = assembler.packet_map(header.id)
        // packet_list: mut & derived_from(r) & List[Array[byte] & iso] ; header~packet, assembler~packet_list
        packet_list.append(packet);
        // filters packet_list,  we don't need to filter packet.
        // So example where we want both possibilities in a signature.
        // packet: Array[byte] & mut ;  header~packet~packet_list~assembler

        if (header.final())
        {
            // We are about to send so need the iso.alias at this point
            var final_packet_list = assembler.packet_map.extract(header.id)
            // This filters based on `assembler` which removes `header`, `packet`, `packet_list` but not `assembler` 
            // final_packet_list : iso & List[Array[byte] & iso] ;  assembler  ,  final_packet_list
            when () {
                process_message(final_packet_list)
            }
        }
    }
}
```



**[File dump from this point on]**

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
