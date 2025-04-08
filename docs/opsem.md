# Operational Semantics

Still to do:
* Type tests and `typeof` for Ref Type.
* Arrays? Or model them as objects?
* For a reference to a primitive on the stack, build a wrapper object.
* Undecided region.
* Region extraction.
* Region entry points.
* Region deallocation.
* Immutability.
* Behaviors.
* GC or RC cycle detection.

## Shape

```rs

x, y, z ∈ Ident
τ ∈ TypeId
𝕗 ∈ FuncId
ρ ∈ RegionId
Φ ∈ FrameId
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
// TODO: this doesn't allow embedded object fields
ω ∈ Object = Ident ↦ Value

ϕ ∈ Frame = {
      id = FrameId,
      vars: Ident ↦ Value,
      ret: Ident,
      cont: Statement*
    }

σ ∈ Stack = Frame*

R ∈ RegionType = RegionRC | RegionGC | RegionArena
    Region = {
      type: RegionType,
      members: ObjectId ↦ ℕ
    }

    // An object located in another object is an embedded field.
    Metadata =
    {
      type: TypeId,
      location: RegionId | FrameId | ObjectId
    }

χ ∈ Heap =
    {
      data: ObjectId ↦ Object,
      metadata: ObjectId ↦ Metadata
      frames: 𝒫(FrameId),
      regions: RegionId ↦ Region
    }

Heap, Stack, Statement* ⇝ Heap, Stack, Statement*

// Frame variables.
x ∈ φ ≝ x ∈ dom(φ.vars)
φ(x) = φ.vars(x)
φ[x↦v] = φ[vars(x)↦v]
φ\x = φ[vars\{x}]
φ\{x} = φ[vars\{x}]

// Heap objects.
ι ∈ χ ≝ ι ∈ dom(χ.data)
χ(ι) = χ.data(ι)
χ[ι↦(ω, τ, Φ)] = χ[data(ι)↦ω, metadata(ι)↦(τ, Φ)]
χ[ι↦(ω, τ, ρ)] = χ[data(ι)↦ω, metadata(ι)↦(τ, ρ), regions(ρ).members[ι↦1]]

// Regions.
ρ ∈ χ ≝ ρ ∈ dom(χ.regions)
χ[ρ↦R] = χ[regions(ρ)↦(R, ∅)]

// Frame management.
χ∪Φ = {χ.data, χ.metadata, χ.frames∪{Φ}, χ.regions}
χ\Φ = {χ.data, χ.metadata, χ.frames\{Φ}, χ.regions}

// Stack deallocation.
χ\ι = {χ.data\{ι}, χ.metadata\{ι}, χ.frames, χ.regions}
χ\{ιs} = {χ.data\ιs, χ.metadata\ιs, χ.frames, χ.regions}

// Object in region deallocation.
χ\(ι, ρ) = {χ.data\{ι}, χ.metadata\{ι}, χ.frames, χ.regions[ρ\{ι}]}

// Dynamic type of a value.
typeof(χ, v) =
  P.primitives(Bool) if v ∈ Bool
  P.primitives(Signed × ℕ) if v ∈ Signed × ℕ
  P.primitives(Unsigned × ℕ) if v ∈ Unsigned × ℕ
  P.primitives(Float × ℕ) if v ∈ Float × ℕ
  χ.metadata(ι).type if ι = v
  // TODO: dynamic type of a reference is not a τ !!
  Ref typeof(χ, χ(𝕣.object)(𝕣.field)) if 𝕣 = v

// Subtype test.
// TODO: what if it's a reference?
typetest(χ, v, T) = T ∈ P.types(typeof(χ, v)).supertypes

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
    ∀i ∈ 0 .. n . ιsᵢ₊₁ = reachable(ιsᵢ, χ(ι)(xsᵢ))


// It's safe to return an object from a function if it's:
// * in a region, or
// * in a parent frame on the same stack, or
// * embedded in an object that is safe to return.
returnable(χ, σ, ι) =
  χ.metadata(ι).location = ρ ∨
  ∃ϕ ∈ σ . χ.metadata(ι).location = ϕ.id ∨
  ((χ.metadata(ι).location = ι′) ∧ returnable(χ, σ, ι′))

// Reference counting.
inc(χ, p) = χ
inc(χ, 𝕣) = dec(χ, 𝕣.object)
inc(χ, ι) =
  χ if χ.metadata(ι).location = Φ
  inc(χ, ι′) if χ.metadata(ι).location = ι′
  incref(χ, ι) if χ.metadata(ι).location = ρ ∧ ρ.type = RegionRC

dec(χ, p) = χ
dec(χ, 𝕣) = dec(χ, 𝕣.object)
dec(χ, ι) =
  χ if χ.metadata(ι).location = Φ
  dec(χ, ι′) if χ.metadata(ι).location = ι′
  decref(χ, ι) if χ.metadata(ι).location = ρ ∧ ρ.type = RegionRC

incref(χ, ι) =
  χ[ρ↦χ(ρ)[members(ι)↦(rc + 1)]]
  where
    ρ = χ.metadata(ι).location ∧
    rc = χ(ρ).members(ι)

decref(χ, ι) =
  free(χ, ρ, ι) if rc = 1
  χ[ρ↦χ(ρ)[members(ι)↦(rc - 1)]] otherwise
  where
    ρ = χ.metadata(ι).location ∧
    rc = χ(ρ).members(ι)

free(χ, ρ, ι) = χₙ[ρ\ι] where
  χ₀ = χ ∧
  n = |xs| ∧
  xs = [x | x ∈ dom(χ(ι))] ∧
  ∀i ∈ 0 .. n . χᵢ₊₁ = dec(χᵢ, χ(ι)(xsᵢ))

```

## New

```rs

x ∉ φ
--- [new primitive]
χ, σ;φ, bind x (primitive p);stmt* ⇝ χ, σ;φ[x↦p], stmt*

newobject(χ, τ, (y, z)*) =
  ω where
    f = P.types(τ).fields ∧
    ys = {y | y ∈ (y, z)*} = dom(f) ∧
    zs = {z | z ∈ (y, z)*} ∧
    |zs| = |dom(f)| ∧
    ω = {y ↦ φ(z) | y ∈ (y, z)*} ∧
    ∀y ∈ dom(ω) . typetest(χ, f(y).type, ω(y))

x ∉ φ
ι ∉ χ
ω = newobject(χ, τ, (y, z)*)
--- [new stack]
χ, σ;φ, bind x (new τ (y, z)*);stmt* ⇝ χ[ι↦(ω, τ, φ.id], σ;φ[x↦ι], stmt*

x ∉ φ
ι ∉ χ
ρ = χ.metadata(φ(y)).location
ω = newobject(χ, τ, (y, z)*)
--- [new heap]
χ, σ;φ, bind x (new y τ (y, z)*);stmt* ⇝ χ[ι↦(ω, τ, ρ)], σ;φ[x↦ι], stmt*

x ∉ φ
ι ∉ χ
ρ ∉ χ
ω = newobject(χ, τ, (y, z)*)
--- [new region]
χ, σ;φ, bind x (new R τ (y, z)*);stmt* ⇝ χ[ρ↦R][ι↦(ω, τ, ρ)], σ;φ[x↦ι], stmt*

```

## Drop, Duplicate

```rs

--- [drop]
χ, σ;φ, drop x;stmt* ⇝ dec(χ, φ, φ(x)), σ;ϕ\x, stmt*

x ∉ ϕ
ϕ(y) = v
--- [dup]
χ, σ;φ, bind x (dup y);stmt* ⇝ inc(χ, v), σ;φ[x↦v], stmt*

```

## Fields

```rs

// TODO: should this consume y instead of inc?
x ∉ ϕ
ι = ϕ(y)
z ∈ dom(P.types(typeof(χ, ι)).fields)
𝕣 = {object: ι, field: z}
--- [bind ref]
χ, σ;ϕ, bind x (ref y z);stmt* ⇝ inc(χ, ι), σ;ϕ[x↦𝕣], stmt*

x ∉ ϕ
𝕣 = ϕ(y)
𝕣 = {object: ι, field: z}
z ∈ dom(P.types(typeof(χ, ι)).fields)
v = χ(ι)(z)
--- [bind load]
χ, σ;ϕ, bind x (load y);stmt* ⇝ inc(χ, v), σ;ϕ[x↦v], stmt*

// TODO: should this consume z instead of inc?
x ∉ ϕ
ϕ(y) = {object: ι, field: z}
z ∈ dom(P.types(typeof(χ, ι)).fields)
v₀ = χ(ι)(z)
v₁ = φ(z)
χ₁ = χ₀[ι↦χ₀(ι)[z↦v₁]]
--- [bind store]
χ₀, σ;ϕ, bind x (store y z);stmt* ⇝ inc(χ₁, v₁), σ;ϕ[x↦v₀], stmt*

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

Θ(x) = false
--- [cond false]
χ, σ;φ, cond x stmt₀* stmt₁*;stmt₂* ⇝ χ, σ;φ, stmt₁*;stmt₂*

```

## Call

All arguments are consumed. To keep them, `dup` them first. As such, an identifier can't appear more than once in the argument list.

```rs

newframe(χ, ϕ, F, x, y*, stmt*) =
  {id: Φ, vars: F.paramsᵢ.name ↦ ϕ(yᵢ) | i ∈ 0 .. |y*|, ret: x, cont: stmt*}
  where
  Φ ∉ dom(χ.metadata.frames) ∧
  |F.params| = |y*| = |{y*}| ∧
  ∀i ∈ 0 .. |y*| . typetest(χ, φ(yᵢ), F.paramsᵢ.type)

x ∉ φ₀
F = P.funcs(𝕗)
φ₁ = newframe(χ, φ₀, F, x, y*, stmt*)
--- [call static]
χ, σ;φ₀, bind x (call 𝕗 y*);stmt* ⇝ χ∪φ₁.id, σ;φ₀;φ₁\{y*}, F.body

x ∉ φ₀
τ = typeof(χ, φ(z₀))
F = P.funcs(P.types(τ).methods(y))
φ₁ = newframe(χ, φ₀, F, x, z*, stmt*)
--- [call dynamic]
χ, σ;φ₀, bind x (call y z*);stmt* ⇝ χ∪φ₁.id, σ;φ₀;φ₁\{z*}, F.body

```

## Return

```rs

|dom(φ₁)| = 1
ιs = {ι | χ.metadata(ι).location = φ₁}
∀ι ∈ reachable(χ, φ₁(x)) . returnable(χ, σ, ι)
--- [return]
χ, σ;φ₀;φ₁, return x;stmt* ⇝ (χ\ιs)\φ₁.id, σ;φ₀[φ₁.ret↦φ₁(x)], ϕ₁.cont

```
