# Region Typing

## Region Operations

gc/cycle-detect: shadow stack of roots
discharge (must be iso, not mut)
  no aliases in our region can exist
  no aliases with provenance that intersects our region can exist [???]
  no aliases with provenance that intersects discharge can exist [???]
  delete
  send
  freeze (could re-type aliases as imm)
  merge (could re-type aliases as in the new region)
destructive read
  region going in is discharged (could re-type aliases as having origin provenance)
  region coming out has discharge regions instead of provenance regions

## Function Calls

the callee must either assume subregion aliases exist or be guaranteed they don't exist
if subregion aliases are assumed to exist, then
  no extraction is allowed, which means no linking either
    because there's the old value that has to be discharged
  extraction always creates a discharge obligation
if subregion aliases are guaranteed not to exist, then
  the caller has to drop subregion references when it makes a call
  `iso` and `imm` can be args
  `mut` can be an arg only if discharge of something extracted from the region is safe at the point the function call is made

## Problem Examples

we might have an alias to the region
  if so, it will have overlapping provenance but not be in a parent region
we might have an alias into the region
  if so, it will have the same region
we might have an alias to a subregion
  if so, it will have our region in its provenance
  if it exists via an alias to us, we handle it via that alias

it's safe to drop a region name from a context
but not safe to add or remove regions from provenance

```ts

case 1: c1: mut(0, r1), c2: mut(0, r2)
case 2 (maybe): c1: mut({r2}, r1), c2: mut({r1}, r2)
case 2 (yes): c1: mut(0, r1), c2: mut(0, r1)
case 3: c1: mut(0, r1), c2: mut({r1}, r2)
case 4: c1: mut({r2}, r1), c2: mut(0, r2)

when (c0, c1)
{
  // can't guarantee c0 and c1 aren't aliases
  x0: mut({r1}, r0), x1: mut({r0}, r1) =>
  let x0a: mut({r1}, r0) = x0.f_mut;
  let x02: mut({r1, r0}, r2) = x0.f_iso;
  let x15: mut({r0, r1}, r5) = x1.f_iso;
  let i03: iso({r1, r0}, r3) = x0.f_iso = new;
  let x14: mut({r0, r1}, r4) = x1.f_iso;
  let x03a: mut({r1, r0}, r3) = i03.f_mut;
  drop(i03);
  // possible aliases:
  // x0: no, is in a parent region
  // x1: no, is in a parent region
  // x0a: no, is in a parent region
  // x02: yes, shares a parent region
  // x15: yes, shares a parent region
  // x14: no, created after extraction
  // x03a: yes, is in the same region
}

```

## Option

```ts

class Option[T]
{
  class Empty {}

  value: T | Empty;

  ?[K: mut(self) | imm](self: Option[T] & K): Bool
  {
    match self.value
    {
      { v: self~>T => true }
      { false }
    }
  }

  apply[K: mut(self) | imm](self: Option[T] & K): K~>T | throw Empty
  {
    match self.value
    {
      { v: K~>T => v }
      { throw Empty }
    }
  }

  update(self: Option[T] & mut(self), value: self<~T): self<~T | Empty
  {
    self.value = value
  }
}

```

## Viewpoint Subtyping

distribute
  isect
  union
  tuple

discard
  C = class, iface

P = typeparam
K = iso, mut, imm

iso(region)
mut(provenance, region)

```ts

C~>T = ()
(T1, T2)~>T = T
T~>(T1, T2) = (T~>T1, T~>T2)
(T1 & T2)~>T = T1~>T & T2~>T
(T1 | T2)~>T = T1~>T | T2~>T
T~>(T1 & T2) = T~>T1 & T~>T2
T~>(T1 | T2) = T~>T1 | T~>T2
P~>T = P.upper~>T & [P~>T] ???

iso(prov, r)~>T =
  mut(prov+r, fresh-r) if T = iso
  mut(prov, r) if T = mut
  imm(r) if T = imm
  iso(prov, r)~>P.upper & [P~>T] if T = P ???
  T otherwise
mut(prov, r)~>T =
  mut(prov+r, fresh-r) if T = iso
  mut(prov, r) if T = mut
  imm(r) if T = imm
  mut(prov, r)~>P.upper & [P~>T] if T = P ???
  T otherwise
imm(r)~>T =
  imm(r) if T <: {iso, mut, imm}
  imm(r)~>P.upper & [P~>T] if T = P ???
  T otherwise

iso(prov, r)<~T =
  iso(prov+r, fresh-r) if T = iso
  mut(prov, r) if T = mut
  imm(r) if T = imm
  iso(prov, r)<~P.upper & [P<~T] if T = P ???
  T otherwise
mut(prov, r)<~T =
  iso(prov+r, fresh-r) if T = iso
  mut(prov, r) if T = mut
  imm(r) if T = imm
  mut(prov, r)<~P.upper & [P<~T] if T = P ???
  T otherwise
imm(r)<~T = false

?
---
D |- T1~>T2 <: T3

?
---
D |- T1 <: T2~>T3

---
D |- iso(_, _) <: iso(_, _)

---
D |- mut(_, r1) <: mut(_, r1)

---
D |- imm <: imm

G(x) = T1
D |- T1.f = T2
D |- T1~>T2 <: G(y)
---
G, D |- (assign (ref y) (field-load (ref x) f))

// subtype rules for T1<~T2 are sensitive to whether it's on the lhs or rhs
// if T1 is a union, T1.f is a union on read, an isect on write
// if T1 is an isect, it's an isect on read, a union on write
G(x) = T1
D |- T1.f = T2
D |- T1<~T2 <: G(y)
D |- G(z) <: T1<~T2
---
G, D |- (assign (ref y) (field-store (ref x) f (ref z)))

G(x) <: iso(P, r) =>
  forall y, P', r' . y in G /\ mut(P', r') <: G(y) =>
    (P \cap P' = 0 \/ r' in P) /\ r != r'
---
G, D |- (discharge (ref x))

```

## Destructive Read AST

```ts
x0 = x1.f = x2.f = x3;
x0 = (x1.f = (x2.f = x3));

(assign
  (ref x0)
  (assign (select (ref x1) f ()) (assign (select (ref x2) f ()) (ref x3))))
->
(let $0)
(local-store (ref $0) (field-store (ref x2) f (ref x3)))
(let $1)
(local-store (ref $1) (field-store (ref x1) f (ref $0)))
(local-store (ref x0) (ref $1)

x0 = x1.f;

(assign (ref x0) (select (ref x1) f ()))
->
(local-store (ref x0) (field-load (ref x1) f))
```
