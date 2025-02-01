# Operational Semantics

Still to do:
* Region safety.
* Undecided region.
* Region entry points.
* Region deallocation.
* Immutability.
* Behaviors and cowns.
* Embedded object fields?
* Arrays? Or model them as objects?
* Stack references? The container is a frame instead of an object.
* GC or RC cycle detection.
* Non-local returns.

## Shape

```rs

n ∈ ℕ
x, y, z ∈ Ident
xs, ys, zs ∈ 𝒫(Ident)
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
      frames: FrameId ↦ {members: ObjectId ↦ ℕ}
      regions: RegionId ↦ Region
    }

Heap, Stack, Statement* ⇝ Heap, Stack, Statement*

```

## Helpers

```rs

// Frame variables.
x ∈ φ ≝ x ∈ dom(φ.vars)
φ(x) = φ.vars(x)
φ[x↦v] = φ[vars(x)↦v]
φ\x = φ\{x}
φ\xs = φ[vars\xs]

// Heap objects.
ι ∈ χ ≝ ι ∈ dom(χ.data)
χ(ι) = χ.data(ι)
χ[ι↦(ω, τ, Φ)] = χ[data(ι)↦ω, metadata(ι)↦(τ, Φ), frames(Φ).members[ι↦1]]
χ[ι↦(ω, τ, ρ)] = χ[data(ι)↦ω, metadata(ι)↦(τ, ρ), regions(ρ).members[ι↦1]]

// Regions.
ρ ∈ χ ≝ ρ ∈ dom(χ.regions)
χ[ρ↦R] = χ[regions(ρ)↦(R, ∅)]

// Frames.
χ∪Φ = {χ.data, χ.metadata, χ.frames[Φ↦∅], χ.regions}
χ\Φ = {χ.data, χ.metadata, χ.frames\Φ, χ.regions}

// Stack deallocation.
χ\ι = χ\{ι}
χ\ιs = {χ.data\ιs, χ.metadata\ιs, χ.frames, χ.regions}

// Object in region deallocation.
χ\(ι, ρ) = {χ.data\{ι}, χ.metadata\{ι}, χ.frames, χ.regions[ρ\{ι}]}

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
    ∀i ∈ 0 .. n . ιsᵢ₊₁ = reachable(ιsᵢ, χ(ι)(xsᵢ))

// This checks that it's safe to discharge a region, including:
// * deallocate the region, or
// * freeze the region, or
// * send the region to a behavior.
// TODO: this doesn't allow a region to reference another region
// TODO: this doesn't require other regions or stacks not to reference this region
dischargeable(χ, ρ) =
  ∀ι ∈ χ . ι ∉ ιs ⇒ ∀z ∈ dom(χ(ι)) . χ(ι)(z) ∉ ιs ∧
  ∀ι ∈ ιs . reachable(χ, ι) ⊆ ιs
  where
    ιs = χ.regions(ρ).members

```

## Reference counting.

Reference counting is a no-op on `RegionGC` and `RegionArena`. It's tracked on stack allocations to ensure that no allocations on a frame that is being torn down are returned.

```rs

inc(χ, p) = χ
inc(χ, 𝕣) = dec(χ, 𝕣.object)
inc(χ, ι) =
  inc(χ, ι′) if χ.metadata(ι).location = ι′
  incref(χ, ρ, ι) if χ.metadata(ι).location = ρ ∧ ρ.type = RegionRC
  incref(χ, Φ, ι) if χ.metadata(ι).location = Φ
  χ otherwise

dec(χ, p) = χ
dec(χ, 𝕣) = dec(χ, 𝕣.object)
dec(χ, ι) =
  dec(χ, ι′) if χ.metadata(ι).location = ι′
  decref(χ, ρ, ι) if χ.metadata(ι).location = ρ ∧ ρ.type = RegionRC
  decref(χ, Φ, ι) if χ.metadata(ι).location = Φ
  χ otherwise

incref(χ, Φ, ι) =
  χ[frames(Φ)↦χ(Φ)[members(ι)↦(rc + 1)]]
  where
    rc = χ(Φ).members(ι)

incref(χ, ρ, ι) =
  χ[regions(ρ)↦χ.regions(ρ)[members(ι)↦(rc + 1)]]
  where
    rc = χ(ρ).members(ι)

decref(χ, Φ, ι) =
  χ[frames(Φ)↦χ.frames(Φ)[members(ι)↦(rc - 1)]]
  where
    rc = χ(Φ).members(ι)

decref(χ, ρ, ι) =
  free(χ, ρ, ι) if rc = 1
  χ[regions(ρ)↦χ.regions(ρ)[members(ι)↦(rc - 1)]] otherwise
  where
    rc = χ(ρ).members(ι)

free(χ, ρ, ι) = χₙ[ρ\ι] where
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
χ, σ;φ, drop x;stmt* ⇝ dec(χ, φ, φ(x)), σ;ϕ\x, stmt*

x ∉ ϕ
ϕ(y) = v
--- [dup]
χ, σ;φ, bind x (dup y);stmt* ⇝ inc(χ, v), σ;φ[x↦v], stmt*

```

## Fields

```rs

// TODO: ref can't be in a frame yet
// tricky: inc/dec, typeof, reachable take a heap but not a stack
x ∉ ϕ
y ∈ φ
𝕣 = {object: φ.id, field: z}
--- [bind stack ref]
χ, σ;ϕ, bind x (ref y);stmt* ⇝ inc(χ, ι), σ;ϕ[x↦𝕣], stmt*

// TODO: should this consume y instead of inc?
x ∉ ϕ
ι = ϕ(y)
z ∈ dom(P.types(typeof(χ, ι)).fields)
𝕣 = {object: ι, field: z}
--- [bind field ref]
χ, σ;ϕ, bind x (ref y z);stmt* ⇝ inc(χ, ι), σ;ϕ[x↦𝕣], stmt*

x ∉ ϕ
ϕ(y) = {object: ι, field: z}
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

φ(x) = false
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

This checks that:
* only the return value remains in the frame, to ensure proper reference counting, and
* that the reference count of everything allocated on this frame has dropped to zero, which ensures that no dangling references are returned.

```rs

|dom(φ₁)| = 1
ιs = {ι | χ.metadata(ι).location = φ₁}
∀ι ∈ ιs . χ.frames(φ₁.id).members(ι) = 0
--- [return]
χ, σ;φ₀;φ₁, return x;stmt* ⇝ (χ\ιs)\(φ₁.id), σ;φ₀[φ₁.ret↦φ₁(x)], ϕ₁.cont

```

## Extract

```rs

x ∉ φ
ι = φ(y)
ρ₀ = χ₀.metadata(ι).location
ρ₁ ∉ χ₀
ιs = reachable(χ, ι)
∀ι′ ∈ χ₀.regions(ρ₀).members . (ι′ ∉ ιs ⇒ ∀z ∈ dom(χ₀(ι′)) . χ₀(ι′)(z) ∉ ιs)
χ₁ = χ₀[regions(ρ₀).members\ιs]
       [regions(ρ₁)↦{type: χ₀.regions(ρ₀).type, members: ιs}]
       [∀ι′ ∈ ιs . χ₀.metadata(ι′).location↦ρ₁]
--- [extract]
χ₀, σ;φ, bind x (extract y);stmt* ⇝ χ₁, σ;φ[x↦ι]\y, stmt*

```
