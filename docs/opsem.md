# Operational Semantics

Still to do:
* Do we need to prevent cyclic regions?
* Region send and free.
  * When external RC is 0?
  * Could do freeze this way too? Only needed with a static type checker.
  * If external region RC is limited to 1, we can check send-ability statically.
    * But we can also delay send or free until RC is 0.
* Freeze.
  * For a static type checker, how do we know there are no `mut` aliases?
  * Could error if external RC isn't 0 or 1?
    * Or delay?
  * Extract and then freeze for sub-graphs. The extract could fail.
* Merge.
  * External RC from the destination is removed.
  * Other external RC is added to the destination.
  * If tracking on a per-parent basis, this is easy.
* Undecided region.
  * Is this a per-stack region, where we extract from it?
  * Could be implemented as "allocate in many regions" and "merge often".
  * How to distinguish a merge from a subregion reference? Only need to do this at the language level, the semantics can be explicit.
* Efficient frame teardown.
  * Disallow heap and "earlier frame" objects from referencing frame objects?
  * All function arguments and return values are "heap or earlier frame".
  * Treat each frame as a region, external RC is 0?
  * Each frame could be an Arena, disallowing extract, not needing ref counting.
* Behaviors and cowns.
* Embedded object fields?
* Arrays? Or model them as objects?
* GC or RC cycle detection.
* Non-local returns.

## Shape

```rs

n ∈ ℕ
w, x, y, z ∈ Ident
xs, ys, zs ∈ 𝒫(Ident)
τ ∈ TypeId
𝕗 ∈ FuncId
ρ ∈ RegionId
𝔽 ∈ FrameId
ι ∈ ObjectId
ιs ∈ 𝒫(ObjectId)

T ∈ Type = Bool | Signed × ℕ | Unsigned × ℕ | Float × ℕ | TypeId | Ref TypeId

𝕥 ∈ TypeDesc =
    {
      supertypes: 𝒫(TypeId),
      fields: Ident ↦ Type,
      methods: Ident ↦ FuncId
    }

F ∈ Func =
    {
      params: {name: Ident, type: Type}*,
      result: Type,
      body: Stmt*
    }

P ∈ Program =
    {
      primitives: Type ↦ TypeDesc,
      types: TypeId ↦ TypeDesc,
      funcs: FuncId ↦ Func
    }

𝕣 ∈ Reference = {object: ObjectId, field: Ident}
p ∈ Primitive = Bool | Signed × ℕ | Unsigned × ℕ | Float × ℕ
v ∈ Value = ObjectId | Primitive | Reference
ω ∈ Object = Ident ↦ Value

ϕ ∈ Frame =
    {
      id: FrameId,
      vars: Ident ↦ Value,
      ret: Ident,
      cont: Statement*
    }

σ ∈ Stack = Frame*

R ∈ RegionType = RegionRC | RegionGC | RegionArena
    Region = {
      type: RegionType,
      heap_rc: RegionId ↦ ℕ,
      stack_rc: ℕ
    }

    // An object located in another object is an embedded field.
    Metadata =
    {
      type: TypeId,
      location: RegionId | FrameId | ObjectId | Immutable,
      rc: ℕ
    }

χ ∈ Heap =
    {
      data: ObjectId ↦ Object,
      metadata: ObjectId ↦ Metadata,
      regions: RegionId ↦ Region,
      frames: 𝒫(FrameId)
    }

Heap, Stack, Statement* ⇝ Heap, Stack, Statement*

```

## Helpers

```rs

// Frames.
x ∈ φ ≝ x ∈ dom(φ.vars)
φ(x) = φ.vars(x)
φ[x↦v] = φ[vars(x)↦v]
φ\x = φ\{x}
φ\xs = φ[vars\xs]

𝔽 ∈ χ ≝ φ ∈ dom(χ.frames)
χ∪𝔽 = χ[frames∪𝔽]
χ\𝔽 = χ[frames\𝔽]

// Heap objects.
ι ∈ χ ≝ ι ∈ dom(χ.data)
χ(ι) = χ.data(ι)
χ[ι↦(ω, τ, 𝔽)] = χ[data(ι)↦ω, metadata(ι)↦{type: τ, location: 𝔽, rc: 1}]
χ[ι↦(ω, τ, ρ)] = χ[data(ι)↦ω, metadata(ι)↦{type: τ, location: ρ, rc: 1}]

// Regions.
ρ ∈ χ ≝ ρ ∈ dom(χ.regions)
χ[ρ↦R] = χ[regions(ρ)↦(R, ∅)]

// Deallocation.
χ\ι = χ\{ι}
χ\ιs = χ[data = data\ιs, metadata = metadata\ιs]

```

## Dynamic Types

```rs

// Dynamic type of a value.
typeof(χ, v) =
  P.primitives(Bool) if v ∈ Bool
  P.primitives(Signed × ℕ) if v ∈ Signed × ℕ
  P.primitives(Unsigned × ℕ) if v ∈ Unsigned × ℕ
  P.primitives(Float × ℕ) if v ∈ Float × ℕ
  χ.metadata(ι).type if ι = v
  Ref typeof(χ, χ(𝕣.object)(𝕣.field)) if 𝕣 = v

// Subtype test.
typetest(χ, v, T) =
  T = typeof(χ, v) if v ∈ Reference
  T ∈ P.types(typeof(χ, v)).supertypes otherwise

```

## Reachability

```rs

// Transitive closure.
reachable(χ, σs) = ∀σ ∈ σs . ⋃{reachable(χ, σ)}
reachable(χ, σ) = ∀φ ∈ σ . ⋃{reachable(χ, φ)}
reachable(χ, φ) = ∀x ∈ dom(φ) . ⋃{reachable(χ, φ(x))}
reachable(χ, v) = reachable(χ, v, ∅)
reachable(χ, p, ιs) = ιs
reachable(χ, 𝕣, ιs) = reachable(χ, 𝕣.object, ιs)
reachable(χ, ι, ιs) =
  ιs if ι ∈ ιs
  ιsₙ otherwise
  where
    xs = [x | x ∈ dom(χ(ι))] ∧
    n = |xs| ∧
    ιs₀ = (ι ∪ ιs) ∧
    ∀i ∈ 1 .. n . ιsᵢ = reachable(χ, χ(ι)(xsᵢ), ιsᵢ₋₁)

// Region.
loc(χ, p) = Immutable
loc(χ, 𝕣) = loc(χ, 𝕣.object)
loc(χ, ι) = χ.metadata(ι).location

same_loc(χ, v₀, v₁) = (loc(χ, v₀) = loc(χ, v₁))

// Mutability.
mut(χ, p) = false
mut(χ, 𝕣) = mut(χ, 𝕣.object)
mut(χ, ι) = loc(χ, ι) ≠ Immutable
mut-reachable(χ, σ) = {ι′ | ι′ ∈ reachable(χ, σ) ∧ mut(χ, ι′)}
mut-reachable(χ, φ) = {ι′ | ι′ ∈ reachable(χ, φ) ∧ mut(χ, ι′)}
mut-reachable(χ, ι) = {ι′ | ι′ ∈ reachable(χ, ι) ∧ mut(χ, ι′)}

```

## Well-Formedness

```rs

// Deep immutability.
wf_immutable(χ) =
  ∀ι ∈ χ . ¬mut(χ, ι) ⇒ (mut-reachable(χ, ι) = ∅)

// Data-race freedom.
wf_racefree(χ, σs) =
  ∀σ₀, σ₁ ∈ σs . σ₀ ≠ σ₁ ⇒ (mut-reachable(σ₀) ∩ mut-reachable(σ₁) = ∅)

// Stack allocations are reachable only from that stack.
wf_stacklocal(χ, σs) =
  ∀σ₀, σ₁ ∈ σs . ∀φ ∈ σ₀ . (reachable(χ, σ₁) ∩ ιs = ∅)
  where
    ιs = {ι | loc(χ, ι) = φ.id}

```

## Reference Counting

Reference counting is a no-op unless the object is in a `RegionRC` or is `Immutable`.

```rs

region_stack_inc(χ, p) = χ
region_stack_inc(χ, 𝕣) = region_stack_inc(χ, 𝕣.object)
region_stack_inc(χ, ι) =
  χ if loc(χ, ι) = Immutable
  χ[regions(ρ)[stack_rc↦(rc + 1)]] otherwise
  where
    loc(χ, ι) = ρ ∧
    χ.regions(ρ).stack_rc = rc

region_stack_dec(χ, p) = χ
region_stack_dec(χ, 𝕣) = region_stack_dec(χ, 𝕣.object)
region_stack_dec(χ, ι) =
  χ if loc(χ, ι) = Immutable
  χ[regions(ρ)[stack_rc↦(rc - 1)]] otherwise
  where
    loc(χ, ι) = ρ ∧
    χ.regions(ρ).stack_rc = rc

region_heap_inc(χ, ι, p) = χ
region_heap_inc(χ, ι, 𝕣) = region_heap_inc(χ, ι, 𝕣.object)
region_heap_inc(χ, ι, ι′) =
  χ if loc(χ, ι′) = Immutable
  χ if same_loc(χ, ι, ι′)
  χ[regions(ρ′)[heap_rc(ρ)↦(rc + 1)]] otherwise
  where
    (loc(χ, ι) = ρ) ∧ (loc(χ, ι′) = ρ′) ∧
    χ.regions(ρ′).heap_rc(ρ) = rc

region_heap_dec(χ, ι, p) = χ
region_heap_dec(χ, ι, 𝕣) = region_heap_dec(χ, ι, 𝕣.object)
region_heap_dec(χ, ι, ι′) =
  χ if loc(χ, ι′) = Immutable
  χ if same_loc(χ, ι, ι′)
  χ[regions(ρ′)[heap_rc(ρ)↦(rc - 1)]] otherwise
  where
    (loc(χ, ι) = ρ) ∧ (loc(χ, ι′) = ρ′) ∧
    χ.regions(ρ′).heap_rc(ρ) = rc

enable-rc(χ, ι) =
  (loc(χ, ι) = ρ ∧ ρ.type = RegionRC) ∨ (loc(χ, ι) = Immutable)

inc(χ, p) = χ
inc(χ, 𝕣) = dec(χ, 𝕣.object)
inc(χ, ι) =
  inc(χ, ι′) if loc(χ, ι) = ι′
  χ[metadata(ι)[rc↦metadata(ι).rc + 1]] if enable-rc(χ, ι)
  χ otherwise

dec(χ, p) = χ
dec(χ, 𝕣) = dec(χ, 𝕣.object)
dec(χ, ι) =
  dec(χ, ι′) if loc(χ, ι) = ι′
  free(χ, ι) if enable-rc(χ, ι) ∧ (χ.metadata(ι).rc = 1)
  χ[metadata(ι)[rc↦metata(ι).rc - 1]] if enable-rc(χ, ι)
  χ otherwise

free(χ, ι) = χₙ\ι where
  xs = [x | x ∈ dom(χ(ι))] ∧
  n = |xs| ∧
  χ₀ = χ ∧
  ∀i ∈ 1 .. n . (ιᵢ = χ(ι)(xsᵢ)) ∧ χᵢ₊₁ = dec(region_heap_dec(χᵢ, ι, ιᵢ), ιᵢ)

```

## New

For an "address-taken" local variable, i.e. a `var` as opposed to a `let`, allocate an object in the frame with a single field to hold the value.

```rs

newobject(χ, τ, (y, z)*) =
  ω where
    f = P.types(τ).fields ∧
    {y | y ∈ (y, z)*} = dom(f) ∧
    ω = {y ↦ φ(z) | y ∈ (y, z)*} ∧
    ∀y ∈ dom(ω) . typetest(χ, f(y).type, ω(y))

x ∉ φ
--- [new primitive]
χ, σ;φ, bind x (new p);stmt* ⇝ χ, σ;φ[x↦p], stmt*

x ∉ φ
ι ∉ χ
zs = {z | z ∈ (y, z)*} ∧ |zs| = |(y, z)*|
ω = newobject(χ, τ, (y, z)*)
--- [new stack]
χ, σ;φ, bind x (new τ (y, z)*);stmt* ⇝ χ[ι↦(ω, τ, φ.id)], σ;φ[x↦ι]\zs, stmt*

x ∉ φ
ι ∉ χ
ρ = loc(χ, φ(w))
zs = {z | z ∈ (y, z)*} ∧ |zs| = |(y, z)*|
ω = newobject(χ, τ, (y, z)*)
--- [new heap]
χ, σ;φ, bind x (new w τ (y, z)*);stmt* ⇝ χ[ι↦(ω, τ, ρ)], σ;φ[x↦ι]\zs, stmt*

x ∉ φ
ι ∉ χ
ρ ∉ χ
zs = {z | z ∈ (y, z)*} ∧ |zs| = |(y, z)*|
ω = newobject(χ, τ, (y, z)*)
--- [new region]
χ, σ;φ, bind x (new R τ (y, z)*);stmt* ⇝ χ[ρ↦R][ι↦(ω, τ, ρ)], σ;φ[x↦ι]\zs, stmt*

```

## Drop, Duplicate

Local variables are consumed on use. To keep them, `dup` them first.

```rs

φ(x) = v
χ₁ = region_stack_dec(χ₀, v)
χ₂ = dec(χ₁, v)
--- [drop]
χ₀, σ;φ, drop x;stmt* ⇝ χ₂, σ;ϕ\x, stmt*

x ∉ ϕ
ϕ(y) = v
χ₁ = region_stack_inc(χ₀, v)
χ₂ = inc(χ₁, v)
--- [dup]
χ₀, σ;φ, bind x (dup y);stmt* ⇝ χ₂, σ;φ[x↦v], stmt*

```

## Fields

The `load` statement is the only operation other than `dup` or `drop` that can change the reference count of an object.

The containing object in `load` and `store` is not consumed.

```rs

x ∉ ϕ
ι = ϕ(y)
w ∈ dom(P.types(typeof(χ, ι)).fields)
𝕣 = {object: ι, field: w}
--- [bind field ref]
χ, σ;ϕ, bind x (ref y w);stmt* ⇝ χ, σ;ϕ[x↦𝕣]\y, stmt*

x ∉ ϕ
ϕ(y) = {object: ι, field: w}
w ∈ dom(P.types(typeof(χ₀, ι)).fields)
v = χ₀(ι)(w)
χ₁ = region_stack_inc(χ₀, v)
χ₂ = inc(χ₁, v)
--- [bind load]
χ₀, σ;ϕ, bind x (load y);stmt* ⇝ χ₂, σ;ϕ[x↦v], stmt*

x ∉ ϕ
ϕ(y) = {object: ι, field: w}
w ∈ dom(P.types(typeof(χ₀, ι)).fields)
mut(χ₀, ι)
v₀ = χ₀(ι)(w)
v₁ = φ(z)
ω = χ₀(ι)[w↦v₁]
χ₁ = region_stack_inc(χ₀, v₀)
χ₂ = region_heap_inc(χ₁, ι, v₁)
χ₃ = region_stack_dec(χ₂, v₁)
χ₄ = region_heap_dec(χ₃, ι, v₀)
--- [bind store]
χ₀, σ;ϕ, bind x (store y z);stmt* ⇝ χ₄[ι↦ω], σ;ϕ[x↦v₀]\z, stmt*

```

## Type Test

The local variable being type-tested is not consumed.

```rs

x ∉ ϕ
v = typetest(χ, φ(y), T)
--- [typetest]
χ, σ;φ, bind x (typetest T y);stmt* ⇝ χ, σ;φ[x↦v], stmt*

```

## Conditional

The condition is not consumed.

```rs

φ(x) = true
--- [cond true]
χ, σ;φ, cond x stmt₀* stmt₁*;stmt₂* ⇝ χ, σ;φ, stmt₀*;stmt₂*

φ(x) = false
--- [cond false]
χ, σ;φ, cond x stmt₀* stmt₁*;stmt₂* ⇝ χ, σ;φ, stmt₁*;stmt₂*

```

## Call

All arguments are consumed. To keep them, `dup` them first. As such, an identifier can't appear more than once in the argument list.

```rs

newframe(χ, ϕ, F, x, y*, stmt*) =
  {id: 𝔽, vars: {F.paramsᵢ.name ↦ ϕ(yᵢ) | i ∈ 1 .. |y*|}, ret: x, cont: stmt*}
  where
  𝔽 ∉ dom(χ.frames) ∧
  |F.params| = |y*| = |{y*}| ∧
  ∀i ∈ 1 .. |y*| . typetest(χ, φ(yᵢ), F.paramsᵢ.type)

x ∉ φ₀
F = P.funcs(𝕗)
φ₁ = newframe(χ, φ₀, F, x, y*, stmt*)
--- [call static]
χ, σ;φ₀, bind x (call 𝕗 y*);stmt* ⇝ χ∪(φ₁.id), σ;φ₀\{y*};φ₁, F.body

x ∉ φ₀
τ = typeof(χ, φ(y₀))
F = P.funcs(P.types(τ).methods(w))
φ₁ = newframe(χ, φ₀, F, x, y*, stmt*)
--- [call dynamic]
χ, σ;φ₀, bind x (call w y*);stmt* ⇝ χ∪(φ₁.id), σ;φ₀\{y*};φ₁, F.body

```

## Return

This checks that:
* Only the return value remains in the frame, to ensure proper reference counting.
* No objects that will survive the frame reference any object allocated on the frame, to prevent dangling references.

> TODO: how to make this efficient?

```rs

dom(φ₁.vars) = {x}
ιs = {ι | loc(χ, ι) = φ₁.id}
∀ι ∈ χ . ι ∉ ιs ⇒ (∀z ∈ dom(χ(ι)) . χ(ι)(z) ∉ ιs)
--- [return]
χ, σ;φ₀;φ₁, return x;stmt* ⇝ (χ\ιs)\(φ₁.id), σ;φ₀[φ₁.ret↦φ₁(x)], ϕ₁.cont

```

## Freeze

Dynamic freeze is suitable for a dynamic type checker. A static type checker will have incorrect mutability information if there are mutable aliases.

```rs

x ∉ φ
ι = φ(y)
ιs = mut-reachable(χ, ι)
∀ι′ ∈ ιs . loc(χ, ι′) ∉ FrameId
χ₁ = χ₀[∀ι′ ∈ ιs . metadata(ι′)[location↦Immutable]]
--- [dynamic freeze]
χ₀, σ;φ, bind x (freeze y);stmt* ⇝ χ₁, σ;φ[x↦ι]\y, stmt*

```

## Extract

> TODO: Doesn't work. Doesn't allow sub-regions or immutable objects.

```rs

x ∉ φ
ι = φ(y)
ρ₀ = loc(χ₀, ι)
ρ₁ ∉ χ₀
ιs = reachable(χ, ι)
∀ι′ ∈ χ₀.regions(ρ₀).members . (ι′ ∉ ιs ⇒ ∀z ∈ dom(χ₀(ι′)) . χ₀(ι′)(z) ∉ ιs)
χ₁ = χ₀[regions(ρ₀).members\ιs]
       [regions(ρ₁)↦{type: χ₀.regions(ρ₀).type, members: ιs}]
       [∀ι′ ∈ ιs . metadata(ι′).location↦ρ₁]
--- [extract]
χ₀, σ;φ, bind x (extract y);stmt* ⇝ χ₁[φ\y][φ(x)↦ι], σ;φ, stmt*

```
