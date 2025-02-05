# Operational Semantics

Still to do:
* Region safety.
* Region entry points.
* Region deallocation.
* Immutability.
  * SCCs? No, keep it abstract. Use same cycle detection algo as RC/GC.
* Undecided region.
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
      type: RegionType
    }

    // An object located in another object is an embedded field.
    Metadata =
    {
      type: TypeId,
      location: RegionId | FrameId | ObjectId,
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

## Reachability and Safety

```rs

// Transitive closure.
reachable(χ, v) = reachable(χ, v, ∅)
reachable(χ, p, ιs) = ιs
reachable(χ, 𝕣, ιs) = reachable(χ, 𝕣.object, ιs)
reachable(χ, ι, ιs) =
  ιs if ι ∈ ιs
  ιsₙ otherwise
  where
    xs = [x | x ∈ dom(χ(ι))] ∧
    n = |xs| ∧
    ιs₀ = ιs ∧
    ∀i ∈ 0 .. (n - 1) . ιsᵢ₊₁ = reachable(ιsᵢ, χ(ι)(xsᵢ))

// Tree structured regions.
// TODO: stack references?
regiondom(χ, ρ₀, ρ₁) =
  ∀ι₀, ι₁ ∈ χ .
    (∃z . χ(ι₀)(z) = ι₁) ∧ (χ.metadata(ι₁).location = ρ₁) ⇒
    χ.metadata(ι₀).location ∈ {ρ₀, ρ₁}

// This checks that it's safe to discharge a region, including:
// * deallocate the region, or
// * freeze the region, or
// * send the region to a behavior.
// TODO: this doesn't allow a region to reference another region
// needs to allow references in to immutable objects.
// TODO: this doesn't require stacks not to reference this region
dischargeable(χ, ρ) =
  ∀ι ∈ χ . ι ∉ ιs ⇒ ∀z ∈ dom(χ(ι)) . χ(ι)(z) ∉ ιs ∧
  ∀ι ∈ ιs . reachable(χ, ι) ⊆ ιs
  where
    ιs = {ι | χ.metadata(ι).location = ρ}

```

## Reference counting.

Reference counting is a no-op unless the object is in a `RegionRC`.

```rs

inc(χ, p) = χ
inc(χ, 𝕣) = dec(χ, 𝕣.object)
inc(χ, ι) =
  inc(χ, ι′) if χ.metadata(ι).location = ι′
  χ[metadata(ι)[rc↦metadata(ι).rc + 1]] if
    χ.metadata(ι).location = ρ ∧ ρ.type = RegionRC
  χ otherwise

dec(χ, p) = χ
dec(χ, 𝕣) = dec(χ, 𝕣.object)
dec(χ, ι) =
  dec(χ, ι′) if χ.metadata(ι).location = ι′
  free(χ, ρ, ι) if
    χ.metadata(ι).rc = 1 ∧
    χ.metadata(ι).location = ρ ∧ ρ.type = RegionRC
  χ[metadata(ι)[rc↦metata(ι).rc - 1]] if
    χ.metadata(ι).location = ρ ∧ ρ.type = RegionRC
  χ otherwise

free(χ, ρ, ι) = χₙ\ι where
  xs = [x | x ∈ dom(χ(ι))] ∧
  n = |xs| ∧
  χ₀ = χ ∧
  ∀i ∈ 0 .. (n - 1) . χᵢ₊₁ = dec(χᵢ, χ(ι)(xsᵢ))

```

## New

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
ρ = χ.metadata(φ(y)).location
zs = {z | z ∈ (y, z)*} ∧ |zs| = |(y, z)*|
ω = newobject(χ, τ, (y, z)*)
--- [new heap]
χ, σ;φ, bind x (new y τ (y, z)*);stmt* ⇝ χ[ι↦(ω, τ, ρ)], σ;φ[x↦ι]\zs, stmt*

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

--- [drop]
χ, σ;φ, drop x;stmt* ⇝ dec(χ, φ(x)), σ;ϕ\x, stmt*

x ∉ ϕ
ϕ(y) = v
--- [dup]
χ, σ;φ, bind x (dup y);stmt* ⇝ inc(χ, v), σ;φ[x↦v], stmt*

```

## Fields

The `load` statement is the only operation other than `dup` or `drop` that can change the reference count of an object.

```rs

x ∉ ϕ
ι = ϕ(y)
w ∈ dom(P.types(typeof(χ, ι)).fields)
𝕣 = {object: ι, field: w}
--- [bind field ref]
χ, σ;ϕ, bind x (ref y w);stmt* ⇝ χ, σ;ϕ[x↦𝕣]\y, stmt*

x ∉ ϕ
ϕ(y) = {object: ι, field: w}
w ∈ dom(P.types(typeof(χ, ι)).fields)
v = χ(ι)(w)
--- [bind load]
χ, σ;ϕ, bind x (load y);stmt* ⇝ inc(χ, v), σ;ϕ[x↦v], stmt*

x ∉ ϕ
ϕ(y) = {object: ι, field: w}
w ∈ dom(P.types(typeof(χ, ι)).fields)
v = χ(ι)(w)
--- [bind store]
χ, σ;ϕ, bind x (store y z);stmt* ⇝ χ[ι↦χ(ι)[w↦φ(z)]], σ;ϕ[x↦v]\z, stmt*

```

## Type Test

```rs

x ∉ ϕ
v = typetest(χ, φ(y), T)
--- [typetest]
χ, σ;φ, bind x (typetest T y);stmt* ⇝ χ, σ;φ[x↦v], stmt*

```

## Conditional

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
  {id: 𝔽, vars: F.paramsᵢ.name ↦ ϕ(yᵢ) | i ∈ 0 .. |y*|, ret: x, cont: stmt*}
  where
  𝔽 ∉ dom(χ.frames) ∧
  |F.params| = |y*| = |{y*}| ∧
  ∀i ∈ 0 .. |y*| . typetest(χ, φ(yᵢ), F.paramsᵢ.type)

x ∉ φ₀
F = P.funcs(𝕗)
φ₁ = newframe(χ, φ₀, F, x, y*, stmt*)
--- [call static]
χ, σ;φ₀, bind x (call 𝕗 y*);stmt* ⇝ χ∪(φ₁.id), σ;φ₀\{y*};φ₁, F.body

x ∉ φ₀
τ = typeof(χ, φ(z₀))
F = P.funcs(P.types(τ).methods(y))
φ₁ = newframe(χ, φ₀, F, x, z*, stmt*)
--- [call dynamic]
χ, σ;φ₀, bind x (call y z*);stmt* ⇝ χ∪(φ₁.id), σ;φ₀\{z*};φ₁, F.body

```

## Return

This checks that:
* Only the return value remains in the frame, to ensure proper reference counting.
* No objects that will survive the frame reference any object allocated on the frame, to prevent dangling references.

```rs

dom(φ₁.vars) = {x}
ιs = {ι | χ.metadata(ι).location = φ₁.id}
∀ι ∈ χ . ι ∉ ιs ⇒ (∀z ∈ dom(χ(ι)) . χ(ι)(z) ∉ ιs)
--- [return]
χ, σ;φ₀;φ₁, return x;stmt* ⇝ (χ\ιs)\(φ₁.id), σ;φ₀[φ₁.ret↦φ₁(x)], ϕ₁.cont

```

## Extract

> Doesn't work. Doesn't allow sub-regions or immutable objects.

```rs

x ∉ χ(φ)
ι = χ(φ, y)
ρ₀ = χ₀.metadata(ι).location
ρ₁ ∉ χ₀
ιs = reachable(χ, ι)
∀ι′ ∈ χ₀.regions(ρ₀).members . (ι′ ∉ ιs ⇒ ∀z ∈ dom(χ₀(ι′)) . χ₀(ι′)(z) ∉ ιs)
χ₁ = χ₀[regions(ρ₀).members\ιs]
       [regions(ρ₁)↦{type: χ₀.regions(ρ₀).type, members: ιs}]
       [∀ι′ ∈ ιs . χ₀.metadata(ι′).location↦ρ₁]
--- [extract]
χ₀, σ;φ, bind x (extract y);stmt* ⇝ χ₁[φ\y][φ(x)↦ι], σ;φ, stmt*

```
