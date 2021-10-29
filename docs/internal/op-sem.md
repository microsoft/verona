# Operational Semantics

Regions:
- acquire
- release

Async:
- new cown
- new promise
- when
- fulfill

TODO:
- behaviour starts with a stack region
  - how do we deal with initial cowns? "paused but openable"
- track gc roots for gc regions
- `using` doesn't consume
  - to re-root, use a storage location and assign to it
  - can re-root if `z` is a storage location, not if its a let binding?
  - need some kind of special storage location?
- we don't want memory management operations on `paused`
  - when `using`, we can't give it "the only reference" to mut or stack
  - but we can on iso or imm

## Types

Assignable fields are `store T`. Embed fields and methods are `T`.

```
Γ       ∈ TypeEnv     ::= x: T | Γ₁, Γ₂ | α
T       ∈ Type        ::= α
                        | iso | mut | imm | paused | stack
                        | (T₁ | T₂)
                        | (T₁ & T₂)
                        | (T₁, T₂, ...)
                        | {f: T}
                        | store T
                        | ClassID
                        | Γ₁ ⊢ T₁→T₂ ⊣ Γ₂
```

## Regions

```ts
imm: no region
  <- ∅
  -> { mut, stack }
iso: disjoint region
  <- ∅
  -> { mut, stack }
mut: top region
  <- { iso, imm, mut }
  -> { mut, stack }
stack: all regions
  <- { iso, imm, mut, paused, stack }
  -> { stack }
paused: subset of regions not including top
  <- ∅
  -> { stack }
```

## Definitions

```
x, y, z ∈ Id
l       ∈ Label
P       ∈ Program     ::= (Id → Function) × (Label → Label*)
ϕ       ∈ Frame       ::= Region* × (Id → Value) × Id × Expression*
σ       ∈ State       ::= Frame*
                        × (ObjectId → Object)
                        × (StorageLoc → Value)
                        × (Region → Strategy)
                        × Bool
v       ∈ Value       ::= ObjectId
                        | StorageLoc
                        | MachineWord
                        | Function
                        | Undefined
ι       ∈ ObjectId
f       ∈ StorageLoc  ::= ObjectId × Id
          MachineWord ::= Bool | i8 | i16 | i32 | i64 | i128 | f32 | f64 | ...
          Bool        ::= true | false
λ         Function    ::= Id* × Expression*
undef   ∈ Undefined
ω       ∈ Object      ::= Label × Region* × (Id → Value)
ρ       ∈ Region
Σ       ∈ Strategy    ::= GC | RC | Arena
e       ∈ Expression  ::= x = var y
                        | x = dup y
                        | x = load y
                        | x = store y z
                        | x = lookup y z
                        | x = typetest x l
                        | x = new l y*
                        | x = call y(y*)
                        | x = using y(z, z*)
                        | x = create Σ y(z*)
                        | tailcall x(x*)
                        | branch x y(y*) z(z*)
                        | return x
                        | error
                        | x = catch
                        | acquire x
                        | release x
                        | fulfill x
```

## Shape

```ts
σ₁, e₁* → σ₂, e₂*
```

## Simplifying Notation

```ts
// Program operations.
P.functions = P↓₁
P.subtypes = P↓₂

// Object operations.
ω.label = ω↓₁
ω.regions = ω↓₂
ω.lookup = ω↓₃

ω <: l = l ∈ P.subtypes(ω.label)
dom(ω) = dom(ω.lookup)
x ∈ ω = x ∈ dom(ω)
ω(x) = ω.lookup(x) if x ∈ dom(ω)
       undef otherwise
ω[x↦v] = ω.label, ω.regions, ω.lookup[x↦v]

// Function operations.
λ.args = λ↓₁
λ.expr = λ↓₂

// Field operations.
f.id = f↓₁
f.loc = f↓₂

// Frame operations.
ϕ.regions = ϕ↓₁
ϕ.lookup = ϕ↓₂
ϕ.ret = ϕ↓₃
ϕ.cont = ϕ↓₄

x ∈ ϕ = x ∈ dom(ϕ.lookup)
ϕ(x) = ϕ.lookup(x)
ϕ[x↦v] = ϕ.regions, ϕ.lookup[x↦v], ϕ.ret, ϕ.cont
ϕ\{x*} = ϕ.regions, ϕ.lookup\{x*}, ϕ.ret, ϕ.cont

// State operations.
x ∈ σ = x ∈ σ.frame
ι ∈ σ = ι ∈ dom(σ.objects)
f ∈ σ = f ∈ dom(σ.fields)
ρ ∈ σ = ρ ∈ dom(σ.regions)

σ.frames = σ↓₁
σ.frame = ϕ where σ↓₁ = ϕ*;ϕ
σ.objects = σ↓₂
σ.fields = σ↓₃
σ.regions = σ↓₄
σ.except = σ↓₅

// Note that P is implicit and immutable.
σ(x) = σ.frame(x) if x ∈ σ
       P.functions(x) if x ∈ dom(P.functions)
       undef otherwise
σ(ι) = σ.objects(ι) if ι ∈ σ
       undef otherwise
σ(f) = σ.fields(f) if f ∈ σ
       undef otherwise

σ[x↦v] =
  (ϕ*; ϕ[x↦v]), σ.objects, σ.fields, σ.regions, σ.except where σ.frames = ϕ*; ϕ
σ\{x*} =
  (ϕ*; ϕ\{x*}), σ.objects, σ.fields, σ.regions, σ.except where σ.frames = ϕ*; ϕ

σ[f↦v] = σ.frames, σ.objects, σ.fields[f↦v], σ.regions, σ.except

σ[ι↦ω] = σ.frames, σ.objects[ι↦ω], σ.fields, σ.regions, σ.except
σ\{ι*} =
  σ.frames, σ.objects\{ι*},
  σ.fields\{(ι, x) | ι ∈ ι* ∧ (ι, x) ∈ dom(σ.fields)},
  σ.regions, σ.except

σ[ρ↦Σ] = σ.frames, σ.objects, σ.fields, σ.regions[ρ↦Σ], σ.except
σ[ρ₁*→ρ₂*] =
  (σ.frames,
   σ.objects[ι↦σ(ι)\{ρ₁*}∪{ρ₂*} | ι ∈ σ ∧ ρ₁* ⊆ σ(ι).regions],
   σ.fields,
   σ.regions\{ρ₁},
   σ.except)

dom(x*) = { xᵢ | i ∈ 1…|x*| }
live(σ, x*) = norepeat(x*) ∧ (dom(σ.frame) = dom(x*))
norepeat(x*) = (|x*| = |dom(x*)|)

σ(()) = ()
σ(x; x*) = (σ(x); σ(x*))

[()↦()] = []
[(x; x*)↦(v; v*)] = [x↦v][x*↦v*]

newframe(σ, ρ*, x, y, z*, e*) =
  ((ϕ*; ϕ₁\{y, z*}; ϕ₂), σ.objects, σ.fields, σ.regions, σ.except), λ.expr
  if λ ∈ Function
  where
    λ = σ(y),
    σ.frames = (ϕ*; ϕ₁),
    ϕ₂ = (ϕ₁.regions; ρ*), [λ.args↦σ(z*)], x, e*

unpin(σ, ()) = ()
unpin(σ, z; z*) =
  (release z; unpin(σ, z*)) if mut(σ, σ(zᵢ)) ∨ stack(σ, σ(zᵢ))
  unpin(σ, z*) otherwise

imm(σ, v) =
  σ(v).regions = () if v ∈ ObjectId
  true otherwise
iso(σ, v) =
  ρ ∉ σ.frame.regions if v ∈ ObjectId where σ(v).regions = ρ
  false otherwise
mut(σ, v) =
  σ(v).regions = ρ₂ if v ∈ ObjectId where σ.frame.region = (ρ₁; ρ*; ρ₂)
  false otherwise
stack(σ, v) =
  σ(v).regions = σ.frame.regions if v ∈ ObjectId
  false otherwise

store(σ, ι, v) =
  imm(σ, v) ∨ iso(σ, v) ∨ mut(σ, v) if mut(σ, ι)
  true if stack(σ, ι)
  false otherwise

store(σ, ι, ()) = true
store(σ, ι, (x; x*)) = store(σ, ι, σ(xᵢ)) ∧ store(σ, ι, x*)
```

## Rules

```ts
// Allocate a cell on the stack.
// x = var
// ->
// x0 = stack l x
// x = lookup x0 x
// release x0
x ∉ σ
ι ∉ σ
f = ι, y
ω = l, σ.frame.regions, [y↦f]
--- [var]
σ, x = var; e* → σ[ι↦ω][f↦undef][x↦f], acquire x; e*

// Duplicate a stack identifier.
// We can't duplicate an iso value.
x ∉ σ
¬iso(σ, σ(y))
--- [dup]
σ, x = dup y; e* → σ[x↦σ(y)], acquire x; e*

// Create a stack identifier to the content of the StorageLoc.
// We can't if the target of the StorageLoc is iso.
x ∉ σ
f = σ(y)
¬iso(σ, σ(f))
--- [load]
σ, x = load y; e* → σ[x↦σ(f)], acquire x; e*

// Load a StorageLoc and replace its content in a single step.
x ∉ σ
norepeat(y; z)
f = σ(y)
v = σ(z)
store(σ, f.id, v)
--- [store]
σ, x = store y z; e* → σ[f↦v][x↦σ(f)]\{z}, e*

// Look in the descriptor table of an object.
// We can't lookup an ObjectId or a StorageLoc unless the object is not iso.
x ∉ σ
ι = σ(y)
v = σ(ι)(z)
v ∈ (StorageLoc ∪ ObjectId) ⇒ ¬iso(σ, ι)
--- [lookup]
σ, x = lookup y z; e* → σ[x↦v], acquire x; e*

// Check abstract subtyping.
// TODO: stuck if not an object?
x ∉ σ
v = σ(ι) <: l if ι ∈ ObjectId where ι = σ(y)
    false otherwise
--- [typetest]
σ, x = typetest y l; e* → σ[x↦v], e*

// Create a new object in the current open region, i.e. a heap object.
norepeat(x*; y*)
norepeat(z*)
x ∉ σ
ι ∉ σ
σ₁.frame.regions = (ρ*; ρ)
ω = l, ρ, [x*↦(ι, x*)][y*↦σ₁(z*)]
σ₂ = σ₁[ι↦ω][(ι, x*)↦undef][x↦ι]
store(σ₂, ι, z*)
--- [new]
σ₁, x = new l x* (y = z)*; e* → σ₂\{z*}, e*

// Create a new object in all open regions, i.e. a stack object.
norepeat(x*; y*)
norepeat(z*)
x ∉ σ
ι ∉ σ
ω = l, σ₁.frame.regions, [x*↦(ι, x*)][y*↦σ₁(z*)]
σ₂ = σ₁[ι↦ω][(ι, x*)↦undef][x↦ι]
--- [stack]
σ₁, x = stack l x* (y = z)*; e* → σ₂\{z*}, e*

// Push a new frame.
norepeat(y; z*)
x ∉ σ
σ₂, e₂* = newframe(σ₁, (), x, y, z*, e₁*)
--- [call]
σ₁, x = call y(z*); e₁* → σ₂, e₂*

// Push a new frame with the specified heap region.
// TODO: don't consume z
norepeat(y; z; z*)
x ∉ σ
ι = σ(z)
ρ = σ(ι).regions
iso(σ, ι)
σ₂, e₂* = newframe(σ₁, ρ, x, y, (z; z*), (unpin(σ₁, z*); e₁*))
--- [using]
σ₁, x = using y(z, z*); e₁* → σ₂, e₂*

// Create a new heap region.
x ∉ σ
ρ ∉ σ
σ₂, e₂* = newframe(σ₁[ρ↦Σ], ρ, x, y, z*, (unpin(σ₁, z*); e₁*))
--- [create]
σ, x = create Σ y(z*); e* → σ₂, e₂*

// Reuse the current frame.
live(σ, x; y*)
λ = σ(x)
--- [tailcall]
σ, tailcall x(y*) → σ[λ.args↦σ(y*)], λ.expr

// Conditional tailcall.
live(σ₁, x; y; z; y*)
live(σ₁, x; y; z; z*)
λ₁ = σ₁(y)
λ₂ = σ₁(z)
σ₂, e* = σ₁[λ₁.args↦σ₁(y*)], λ₁.expr if σ₁(x) = true
         σ₁[λ₂.args↦σ₁(z*)], λ₂.expr if σ₁(x) = false
--- [branch]
σ₁, branch x y(y*) z(z*) → σ₂, e*

// Pop the current frame.
// Can only return iso or imm across a `using`.
// TODO: discard everything in the stack region
live(σ₁, x)
σ₁.frames = (ϕ*; ϕ₁; ϕ₂)
σ₂ = (ϕ*; ϕ₁[ϕ₂.ret↦ϕ₂(x)], σ₁.objects, σ₁.fields, σ₁.except)
(ϕ₁.regions ≠ ϕ₂.regions) ⇒ iso(σ₂, ϕ₂(x)) ∨ imm(σ₂, ϕ₂(x))
--- [return]
σ₁, return x → σ₂, ϕ₂.cont

// Unset the success flag.
--- [error]
σ, error; e* → (σ.frames, σ.objects, σ.fields, σ.regions, false), e*

// Test and set the success flag.
x ∉ σ₁
σ₂ = (σ.frames, σ.objects, σ.fields, σ.regions, true)
--- [catch]
σ₁, x = catch; e* → σ₂[x↦σ₁.except], e*

// Destroy a region and freeze all objects that were in it.
x ∉ σ
ι = σ(y)
iso(σ, ι)
--- [freeze]
σ, x = freeze y; e* → σ[σ(v).regions→∅]\{y}[x↦v], acquire x; e*

// Destroy a region and move all objects that were in it into the currently
// open heap region.
x ∉ σ
ι = σ(y)
iso(σ, ι)
(ρ*; ρ) = σ.frame.regions
--- [merge]
σ, x = merge y; e* → σ[σ(v).regions→ρ]\{y}[x↦v], acquire x; e*

// TODO: rcinc
// address: ?
// object: ?
ι = ϕ(x)
ω = χ(ι) => ???
--- [acquire]
σ, acquire x; e* → σ, e*

// TODO:
// if ι is on the stack, destroy it
// if ι is from an object, rcdec the object
ι = ϕ(x)
--- [release]
σ, release x; e* → σ, e*

// TODO: fulfill the promise
--- [fulfill]
σ, fulfill x → σ, ∅

```
