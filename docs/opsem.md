# Operational Semantics

Still to do:
* Discharge (send/freeze?) when stack RC for the region and all children (recursively) is 0.
  * Could optimize by tracking the count of "busy" child regions.
* How are Arenas different from uncounted regions?
  * How should they treat changing region type?
  * How should they treat merging, freezing, extract?
* Behaviors and cowns.
* Embedded object fields?
* Arrays? Or model them as objects?

Dynamic failures:
* `store`:
  * Unsafe store:
    * Store to a finalizing object.
    * Store a finalizing object.
    * Store to an immutable object.
    * Store a region that already has a parent.
    * Store a region that would create a cycle.
    * Store a frame value in a region or in a predecessor frame.
* `merge`:
  * Trying to merge a value that isn't an object in a region.
  * Trying to merge a region that is a child of a region other than the destination region.
  * Trying to merge a region that would create a cycle.
* `freeze`:
  * Trying to freeze a value that is not an object in a region.
* `extract`:
  * Trying to extract a value that is not an object in a region.
  * Trying to extract a graph that is reachable from the region.

Error values:
* Bad target.
* Bad field.
* Bad method.
* Bad argument type.
* Bad return location.
* Bad return type.

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

    Condition = Return | Raise | Throw
ϕ ∈ Frame =
    {
      id: FrameId,
      vars: Ident ↦ Value,
      ret: Ident,
      cont: Statement*,
      condition: Condition
    }

σ ∈ Stack = Frame*

R ∈ RegionType = RegionRC | RegionGC | RegionArena

    // The size of the parents set will be at most 1.
    Region = {
      type: RegionType,
      parents: 𝒫(RegionId),
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
      frames: 𝒫(FrameId),
      pre_final: 𝒫(ObjectId),
      post_final: 𝒫(ObjectId),
      pre_final_r: 𝒫(RegionId),
      post_final_r: 𝒫(RegionId)
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
χ\ι = χ\{ι}
χ\ιs = χ[data = data\ιs, metadata = metadata\ιs]

// Regions.
ρ ∈ χ ≝ ρ ∈ dom(χ.regions)
χ[ρ↦R] = χ[regions(ρ)↦{type: R, parents: ∅, stack_rc: 1}]
χ\ρ = χ[regions\ρ]

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

reachable(χ, ∅) = ∅
reachable(χ, {v} ∪ vs) = reachable(χ, v) ∪ reachable(χ, vs)

reachable(χ, v) = reachable(χ, v, ∅)
reachable(χ, p, ιs) = ιs
reachable(χ, 𝕣, ιs) = reachable(χ, 𝕣.object, ιs)
reachable(χ, ι, ιs) =
  ιs if ι ∈ ιs
  reachable(χ, ι, {ι} ∪ ιs, dom(χ(ι))) otherwise

reachable(χ, ι, ιs, ∅) = ιs
reachable(χ, ι, ιs, {w} ∪ ws) =
  reachable(χ, ι, ιs, w) ∪ reachable(χ, ι, ιs, ws)
reachable(χ, ι, ιs, w) = reachable(χ, χ(ι)(w), ιs)

// Region.
loc(χ, p) = Immutable
loc(χ, 𝕣) = loc(χ, 𝕣.object)
loc(χ, ι) =
  loc(χ, ι′) if χ.metadata(ι).location = ι′
  χ.metadata(ι).location if ι ∈ χ
  Immutable otherwise

same_loc(χ, v₀, v₁) = (loc(χ, v₀) = loc(χ, v₁))
members(χ, ρ) = {ι | (ι ∈ χ) ∧ (loc(χ, ι) = ρ)}

// Region parents.
parents(χ, ρ) = χ.regions(ρ).parents

// Check if ρ₀ is an ancestor of ρ₁.
is_ancestor(χ, ρ₀, ρ₁) =
  ρ₀ ∈ parents(χ, ρ₁) ∨
  (∀ρ ∈ parents(χ, ρ₁) . is_ancestor(χ, ρ₀, ρ))

```

## Safety

This enforces a tree-shaped region graph, with a single reference from parent to child.

```rs

safe_store(χ, ι, v) =
  false if finalizing(ι) ⊻ finalizing(v)
  false if loc(χ, ι) = Immutable
  true if loc(χ, v) = Immutable
  true if loc(χ, ι) = 𝔽 ∧ (loc(χ, v) = ρ)
  true if loc(χ, ι) = 𝔽 ∧ (loc(χ, v) = 𝔽′) ∧ (𝔽 >= 𝔽′)
  true if same_loc(χ, ι, v)
  true if (ρ₀ = loc(χ, ι)) ∧ (ρ₁ = loc(χ, v)) ∧
          (parents(χ, ρ₁) = ∅) ∧ ¬is_ancestor(χ, ρ₁, ρ₀)
  false otherwise

finalizing(χ, p) = false
finalizing(χ, 𝕣) = finalizing(χ, 𝕣.object)
finalizing(χ, ι) = (ι ∈ χ.pre_final) ∨ (ι ∈ χ.post_final)

```

## Well-Formedness

```rs

// Deep immutability.
wf_immutable(χ) =
  ∀ι₀, ι₁ ∈ χ .
    (loc(χ, ι₀) = Immutable) ∧ (ι₁ ∈ reachable(χ, ι₀)) ⇒
    (loc(χ, ι₁) = Immutable)

// Data-race freedom.
wf_racefree(χ, σs) =
  ∀σ₀, σ₁ ∈ σs . ∀ι ∈ χ .
    (ι ∈ reachable(χ, σ₀)) ∧ (ι ∈ reachable(χ, σ₁)) ⇒
    (σ₀ = σ₁) ∨ (loc(χ, ι) = Immutable)

// Frame allocations are reachable only from that frame or antecedent frames.
wf_stacklocal(χ, σs) =
  ∀ι ∈ χ .
    (loc(χ, ι) = 𝔽) ⇒ ∀ι′ ∈ χ .
      ι ∈ reachable(χ, ι′) ⇒
        (loc(χ, ι′) = 𝔽′) ∧ (𝔽 <= 𝔽′)

// Regions are externally unique.
wf_regionunique(χ) =
  ∀ρ ∈ χ . (|ιs₂| ≤ 1) ∧ (|ρs| ≤ 1) ∧ (ρs = parents(χ, ρ))
    where
      ιs₀ = members(χ, ρ) ∧
      ιs₁ = {ι | (ι ∈ χ) ∧ (loc(χ, ι) = ρ′) ∧ (ρ ≠ ρ′)} ∧
      ιs₂ = {ι | (ι ∈ ιs₁) ∧ (w ∈ dom(χ(ι))) ∧ (χ(ι)(w) ∈ ιs₀)} ∧
      ρs = {ρ′ | (ι ∈ ιs₂) ∧ (loc(χ, ι) = ρ′)}

// The region graph is a tree.
wf_regiontree(χ) =
  ∀ρ₀, ρ₁ ∈ χ .
    (ρ₀ ∈ parents(χ, ρ₁) ⇒ (ρ₀ ≠ ρ₁) ∧ ¬is_ancestor(χ, ρ₁, ρ₀))

```

## Region Type Change

```rs

region_type_change(χ, σ, ∅, R) = χ
region_type_change(χ, σ, {ρ} ∪ ρs, R) =
  region_type_change(χ′, σ, ρs, R)
  where
    χ′ = region_type_change(χ, σ, ρ, R)

region_type_change(χ, σ, ρ, R) =
  calc_rc(χ′, σ, ρ) if (R′ ≠ RegionRC) ∧ (R = RegionRC)
  χ′ otherwise
  where
    R′ = χ.regions(ρ).type ∧
    χ′ = χ[regions(ρ)[type = R]]

calc_rc(χ, σ, ρ) = calc_rc(χ, σ, members(χ, ρ))
calc_rc(χ, σ, ∅) = χ
calc_rc(χ, σ, {ι} ∪ ιs) =
  calc_rc(χ′, σ, ιs)
  where
    χ′ = calc_rc(χ, σ, ι)
calc_rc(χ, σ, ι) =
  χ[metadata(ι)[rc = calc_stack_rc(χ, σ, ι) + calc_heap_rc(χ, ι)]]

calc_stack_rc(χ, σ, ∅) = 0
calc_stack_rc(χ, σ, {ι} ∪ ιs) =
  calc_stack_rc(χ, σ, ι) + calc_stack_rc(χ, σ, ιs)

calc_stack_rc(χ, ∅, ι) = 0
calc_stack_rc(χ, σ;φ, ι) =
  |{x | φ(x) = ι}| + calc_stack_rc(χ, σ, ι)

// The heap RC for the parent region will be zero or one.
calc_heap_rc(χ, ι) =
  calc_heap_rc(χ, {ρ} ∪ ρs, ι)
  where
    (ρ = loc(χ, ι)) ∧ (ρs = parents(χ, ρ))

calc_heap_rc(χ, ∅, ι) = 0
calc_heap_rc(χ, {ρ} ∪ ρs, ι) = calc_heap_rc(χ, ρ, ι) + calc_heap_rc(χ, ρs, ι)
calc_heap_rc(χ, ρ, ι) =
  |{(ι′, w) |
    (ι′ ∈ members(χ, ρ)) ∧
    (w ∈ dom(χ(ι′))) ∧
    ((χ(ι′)(w) = ι)) ∨ ((χ(ι′)(w) = 𝕣) ∧ (𝕣.object = ι))}|

```

## Reference Counting

Reference counting is a no-op unless the object is in a `RegionRC` or is `Immutable`.

```rs

enable-rc(χ, ι) =
  (loc(χ, ι) = ρ ∧ ρ.type = RegionRC) ∨ (loc(χ, ι) = Immutable)

region_stack_inc(χ, p) = χ
region_stack_inc(χ, 𝕣) = region_stack_inc(χ, 𝕣.object)
region_stack_inc(χ, ι) =
  χ[regions(ρ)[stack_rc += 1]] if loc(χ, ι) = ρ
  χ otherwise

region_stack_dec(χ, p) = χ
region_stack_dec(χ, 𝕣) = region_stack_dec(χ, 𝕣.object)
region_stack_dec(χ, ι) =
  χ[pre_final_r ∪= {ρ}] if
    (loc(χ, ι) = ρ) ∧
    (parents(χ, ρ) = ∅) ∧
    (χ.regions(ρ).stack_rc = 1)
  χ[regions(ρ)[stack_rc -= 1]] if loc(χ, ι) = ρ
  χ otherwise

region_add_parent(χ, ι, p) = χ
region_add_parent(χ, ι, 𝕣) = region_add_parent(χ, ι, 𝕣.object)
region_add_parent(χ, ι, ι′) =
  χ[regions(ρ′)[parents ∪= {ρ})]] if
    (loc(χ, ι) = ρ) ∧ (loc(χ, ι′) = ρ′) ∧ (ρ ≠ ρ′)
  χ[regions(ρ′)[stack_rc += 1]] if (loc(χ, ι) = 𝔽) ∧ (loc(χ, ι′) = ρ′)
  χ otherwise

region_remove_parent(χ, ι, p) = χ
region_remove_parent(χ, ι, 𝕣) = region_remove_parent(χ, ι, 𝕣.object)
region_remove_parent(χ, ι, ι′) =
  χ[regions(ρ)[parents \= {ρ′})]] if
    (loc(χ, ι) = ρ) ∧ (loc(χ, ι′) = ρ′) ∧ (ρ ≠ ρ′)
  χ[regions(ρ′)[stack_rc -= 1]] if (loc(χ, ι) = 𝔽) ∧ (loc(χ, ι′) = ρ′)
  χ otherwise

inc(χ, p) = χ
inc(χ, 𝕣) = dec(χ, 𝕣.object)
inc(χ, ι) =
  inc(χ, ι′) if χ.metadata(ι).location = ι′
  χ[metadata(ι)[rc += 1]] if enable-rc(χ, ι)
  χ otherwise

dec(χ, p) = χ
dec(χ, 𝕣) = dec(χ, 𝕣.object)
dec(χ, ι) =
  dec(χ, ι′) if χ.metadata(ι).location = ι′
  free(χ, ι) if enable-rc(χ, ι) ∧ (χ.metadata(ι).rc = 1)
  χ[metadata(ι)[rc -= 1]] if enable-rc(χ, ι)
  χ otherwise

```

## Garbage Collection

```rs

// GC on RegionRC is cycle detection.
enable-gc(χ, ρ) = χ.regions(ρ).type ∈ {RegionGC, RegionRC}

gc(χ, σ, ρ) =
  χ′[pre_final ∪= ιs₀] if enable-gc(χ₃, ρ)
  χ otherwise
  where
    ιs = members(χ₀, ρ) ∧
    ιs₀ ⊆ ιs \ reachable(χ₀, gc_roots(χ₀, σ, ρ)) ∧
    ιs₁ = ιs \ ιs₀ ∧
    χ′ = gc_dec(χ, ιs₀, ιs₁)

gc_roots(χ, σ, ρ) =
  {ι | (ι ∈ ιs) ∧ ((calc_stack_rc(χ, σ, ι) > 0) ∨ (calc_heap_rc(χ, ρs, ι) > 0))}
  where
    ρs = parents(χ, ρ) ∧ ιs = members(χ, ρ)

gc_dec(χ, ∅, ιs₁) = χ
gc_dec(χ, {ι} ∪ ιs₀, ιs₁) =
  gc_dec(χ′, ιs₀, ιs₁)
  where
    χ′ = gc_dec_fields(χ, ι, dom(χ₀(ι)), ιs₁) ∧
  
gc_dec_fields(χ, ι, ∅, ιs₁) = χ
gc_dec_fields(χ, ι, {w} ∪ ws, ιs₁) =
  gc_dec_fields(χ′, ι, ws, ιs₁)
  where
    χ′ = gc_dec_field(χ₀, ι, χ(ι)(w), ιs₁) ∧

gc_dec_field(χ, ι, p, ιs₁) = χ
gc_dec_field(χ, ι, 𝕣, ιs₁) = gc_dec_field(χ, ι, 𝕣.object)
gc_dec_field(χ, ι, ι′, ιs₁) =
  dec(χ, ι′) if (ι′ ∈ ιs₁) ∨ (loc(χ, ι′) = Immutable)
  χ otherwise

```

## Free

```rs

free(χ, ι) =
  χ′[pre_final ∪= ιs]
  where
    χ′, ιs = free_fields(χ, {ι}, ι)

free_fields(χ, ιs, ι) = free_fields(χ, ιs, ι, dom(χ(ι)))
free_fields(χ, ιs, ι, ∅) = χ, ιs
free_fields(χ, ιs, ι, {w} ∪ ws) =
  free_fields(χ′, ιs′, ι, ws)
  where
    χ₁′ ιs′ = free_field(χ, ιs, ι, w)

free_field(χ, ιs, ι, p) = χ, ιs
free_field(χ, ιs, ι, 𝕣) = free_field(χ, ιs, ι, 𝕣.object)
free_field(χ, ιs, ι, ι′) =
  χ, ιs if ι′ ∈ ιs
  free_fields(χ, {ι′} ∪ ιs, ι′), {ι′} ∪ ιs if
    (same_loc(χ, ι, ι′) ∨ (loc(χ, ι′) = Immutable)) ∧
    (χ.metadata(ι′).rc = 1)
  χ[metadata(ι′)[rc -= 1]], ιs if
    (same_loc(χ, ι, ι′) ∨ (loc(χ, ι′) = Immutable)) ∧
    (χ.metadata(ι′).rc > 1)
  free_fields(χ, {ι′} ∪ ιs, ι′), {ι} ∪ ιs if χ.metadata(ι′).location = ι
  χ, ιs, ∅ otherwise

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

## Duplicate, Drop

Local variables are consumed on use. To keep them, `dup` them first.

```rs

x ∉ ϕ
ϕ(y) = v
χ₁ = region_stack_inc(χ₀, v)
χ₂ = inc(χ₁, v)
--- [dup]
χ₀, σ;φ, bind x (dup y);stmt* ⇝ χ₂, σ;φ[x↦v], stmt*

φ(x) = v
χ₁ = region_stack_dec(χ₀, v)
χ₂ = dec(χ₁, v)
--- [drop]
χ₀, σ;φ, drop x;stmt* ⇝ χ₂, σ;ϕ\x, stmt*

```

## Fields

The `load` statement is the only operation other than `dup` or `drop` that can change the reference count of an object. The containing object in `load` and `store` is not consumed.

```rs

x ∉ ϕ
ι = ϕ(y)
w ∈ dom(P.types(typeof(χ, ι)).fields)
𝕣 = {object: ι, field: w}
--- [fieldref]
χ, σ;ϕ, bind x (ref y w);stmt* ⇝ χ, σ;ϕ[x↦𝕣]\y, stmt*

x ∉ ϕ
ϕ(y) ∉ ObjectId
v = // TODO: bad target error
--- [fieldref bad-target]
χ, σ;ϕ, bind x (ref y w);stmt* ⇝ χ, σ;ϕ[x↦v]\y, setthrow;return x

x ∉ ϕ
ι = ϕ(y)
w ∉ dom(P.types(typeof(χ, ι)).fields)
v = // TODO: bad field error
--- [fieldref bad-field]
χ, σ;ϕ, bind x (ref y w);stmt* ⇝ χ, σ;ϕ[x↦v]\y, setthrow;return x

x ∉ ϕ
ϕ(y) = {object: ι, field: w}
v = χ₀(ι)(w)
χ₁ = region_stack_inc(χ₀, v)
χ₂ = inc(χ₁, v)
--- [load]
χ₀, σ;ϕ, bind x (load y);stmt* ⇝ χ₂, σ;ϕ[x↦v], stmt*

x ∉ ϕ
ϕ(y) = {object: ι, field: w}
v₀ = χ₀(ι)(w)
v₁ = φ(z)
safe_store(χ₀, ι, v₁)
ω = χ₀(ι)[w↦v₁]
χ₁ = region_stack_inc(χ₀, v₀)
χ₂ = region_remove_parent(χ₁, ι, v₀)
χ₃ = region_add_parent(χ₂, ι, v₁)
χ₄ = region_stack_dec(χ₃, v₁)
--- [store]
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
  { id: 𝔽, vars: {F.paramsᵢ.name ↦ ϕ(yᵢ) | i ∈ 1 .. |y*|},
    ret: x, cont: stmt*, condition: Return}
  where
    (𝔽 ∉ dom(χ.frames)) ∧ (𝔽 > φ.id)

typecheck(χ, φ, F, y*) =
  |F.params| = |y*| = |{y*}| ∧
  ∀i ∈ 1 .. |y*| . typetest(χ, φ(yᵢ), F.paramsᵢ.type)

x ∉ φ₀
F = P.funcs(𝕗)
typecheck(χ, φ₀, F, y*)
φ₁ = newframe(χ, φ₀, F, x, y*, stmt*)
--- [call static]
χ, σ;φ₀, bind x (call 𝕗 y*);stmt* ⇝ χ∪(φ₁.id), σ;φ₀\{y*};φ₁, F.body

x ∉ φ
F = P.funcs(𝕗)
¬typecheck(χ, φ, F, y*)
v = // TODO: bad args error
--- [call static bad-args]
χ, σ;φ, bind x (call w y*);stmt* ⇝ χ, σ;φ[x↦v], setthrow;return x

x ∉ φ₀
τ = typeof(χ, φ₀(y₁))
F = P.funcs(P.types(τ).methods(w))
typecheck(χ, φ₀, F, y*)
φ₁ = newframe(χ, φ₀, F, x, y*, stmt*)
--- [call dynamic]
χ, σ;φ₀, bind x (call w y*);stmt* ⇝ χ∪(φ₁.id), σ;φ₀\{y*};φ₁, F.body

x ∉ φ
τ = typeof(χ, φ(y₁))
w ∉ P.types(τ).methods
v = // TODO: bad method error
--- [call dynamic bad-method]
χ, σ;φ, bind x (call w y*);stmt* ⇝ χ, σ;φ[x↦v], setthrow;return x

x ∉ φ
τ = typeof(χ, φ(y₁))
F = P.funcs(P.types(τ).methods(w))
¬typecheck(χ, φ, F, y*)
v = // TODO: bad args error
--- [call dynamic bad-args]
χ, σ;φ, bind x (call w y*);stmt* ⇝ χ, σ;φ[x↦v], setthrow;return x

```

## Return

This drops any remaining frame variables other than the return value.

```rs

dom(φ₁.vars) = {x}
v = φ₁(x)
loc(χ, v) ≠ φ₁.id
typetest(χ, v, F.result)
--- [return]
χ, σ;φ₀;φ₁, return x;stmt* ⇝ χ\(φ₁.id), σ;φ₀[φ₁.ret↦v], ϕ₁.cont

dom(φ.vars) = {x, y} ∪ zs
--- [return]
χ, σ;φ, return x;stmt* ⇝ χ, σ;φ, drop y;return x

dom(φ₁.vars) = {x}
v₀ = φ₁(x)
loc(χ, v₀) = φ₁.id
v₁ = // TODO: bad return loc error
--- [return bad-loc]
χ, σ;φ₀;φ₁, return x;stmt* ⇝ χ, σ;φ₀;φ₁[y↦v₁], drop x;setthrow;return y

dom(φ₁.vars) = {x}
v₀ = φ₁(x)
loc(χ, v₀) ≠ φ₁.id
¬typetest(χ, v₀, F.result)
v₁ = // TODO: bad return type error
--- [return bad-type]
χ, σ;φ₀;φ₁, return x;stmt* ⇝ χ, σ;φ₀;φ₁[y↦v₁], drop x;setthrow;return y

```

## Non-Local Return

Use `setreturn` before a return for a standard return. Use `setraise` for a non-local return, and `setthrow` for an error.

Use `checkblock` after a `call` from inside a Smalltalk style block, such as a Verona lambda. If it's true, return the call result to propagate a non-local return out of a collection of blocks to the calling function, i.e. the syntactically enclosing scope.

Use `checkfunc` after a `call` from inside a function. If it's true, return the call result to turn a non-local return into a local return, and to propagate an error.

To catch errors, don't check the call condition.

```rs

--- [set return]
χ, σ;φ, setreturn;stmt* ⇝ χ, σ;φ[condition = Return], stmt*

--- [set raise]
χ, σ;φ, setraise;stmt* ⇝ χ, σ;φ[condition = Raise], stmt*

--- [set throw]
χ, σ;φ, setthrow;stmt* ⇝ χ, σ;φ[condition = Throw], stmt*

x ∉ φ
--- [check block]
χ, σ;φ, bind x checkblock;stmt* ⇝ χ, σ;φ[x↦condition ≠ Return], stmt*

x ∉ φ
φ.condition = Return
--- [check function]
χ, σ;φ, bind x checkfunc;stmt* ⇝ χ, σ;φ[x↦false], stmt*

x ∉ φ
φ.condition = Raise
--- [check function]
χ, σ;φ, bind x checkfunc;stmt* ⇝ χ, σ;φ[x↦true, condition = Return], stmt*

x ∉ φ
φ.condition = Throw
--- [check function]
χ, σ;φ, bind x checkfunc;stmt* ⇝ χ, σ;φ[x↦true], stmt*

```

## Merge

This allows merging two regions. The region being merged must either have no parent, or be a child of the region it's being merged into. If there are other stack references to the region being merged, a static type system may have the wrong region information for them.

> TODO: disallow merging a region that has a parent? Disallow merging a region that has other stack references?

```rs

x ∉ φ
loc(χ₀, φ(w)) = ρ₀
loc(χ₀, φ(y)) = ρ₁
(ρ₀ ≠ ρ₁) ∧ ¬is_ancestor(χ₀, ρ₁, ρ₀) ∧ ({ρ₀} ⊇ parents(χ₀, ρ₁))
ιs = members(χ₀, ρ₁)
χ₁ = χ₀[∀ι ∈ ιs . metadata(ι)[location = ρ₀]]
       [regions(ρ₀)[stack_rc += regions(ρ₁).stack_rc)]]
--- [merge true]
χ₀, σ;φ, bind x (merge w y);stmt* ⇝ χ₁\ρ₁, σ;φ[x↦true], stmt*

x ∉ φ
(loc(χ, φ(w)) ≠ ρ₀) ∨
(loc(χ, φ(y)) ≠ ρ₁) ∨
(ρ₀ = ρ₁) ∨ is_ancestor(χ₀, ρ₁, ρ₀) ∨ ({ρ₀} ̸⊇ parents(χ, ρ₁))
--- [merge false]
χ, σ;φ, bind x (merge w y);stmt* ⇝ χ, σ;φ[x↦false], stmt*

```

## Freeze

If the region being frozen has a parent, a static type system may have the wrong type for the incoming reference. If there are other stack references to the region being frozen or any of its children, a static type system may have the wrong type for them.

> TODO: disallow freezing a region that has a parent? Disallow freezing a region that has other stack references?

```rs

x ∉ φ
ι = φ(y)
ρ = loc(χ₀, ι)
ρs = {ρ} ∪ {ρ′ | (ρ′ ∈ χ.regions) ∧ is_ancestor(χ₀, ρ, ρ′)}
χ₁ = region_type_change(χ₀, σ;φ, ρs, RegionRC)
ιs = {ι′ | loc(χ₀, ι′) ∈ ρs}
χ₂ = χ₁[∀ι′ ∈ ιs . metadata(ι′)[location = Immutable]]
--- [freeze true]
χ₀, σ;φ, bind x (freeze y);stmt* ⇝ χ₂\ρs, σ;φ[x↦true], stmt*

x ∉ φ
loc(χ, φ(y)) ≠ ρ
--- [freeze false]
χ, σ;φ, bind x (freeze y);stmt* ⇝ χ, σ;φ[x↦false], stmt*

```

## Extract

```rs

x ∉ φ
ι = φ(y)
ρ₀ = loc(χ₀, ι)
ρ₁ ∉ χ₀
ιs = reachable(χ, ι) ∩ members(χ₀, ρ₀)
|{ι | (ι ∈ members(χ₀, ρ₀)) ∧ (w ∈ dom(χ₀(ι))) ∧
      (χ₀(ι)(w) = ι′) ∧ (ι′ ∈ \ios)}| = 0
ρs = {ρ |
      (ι ∈ ιs) ∧ (w ∈ dom(χ(ι))) ∧ (χ(ι)(w) = ι′) ∧
      (ρ = loc(χ, ι′)) ∧ (ρ ≠ ρ₀)}
rc = calc_stack_rc(χ₀, σ;φ, ιs)
χ₁ = χ₀[regions(ρ₀)[stack_rc -= rc],
        regions(ρ₁)↦{type: χ.regions(ρ₀).type, parents: ∅, stack_rc: rc},
        ∀ι′ ∈ ιs . metadata(ι′)[location = ρ₁],
        ∀ρ ∈ ρs . regions(ρ)[parents = {ρ₁}]]
--- [extract true]
χ₀, σ;φ, bind x (extract y);stmt* ⇝ χ₁, σ;φ[x↦true], stmt*

x ∉ φ
ρ ≠ loc(χ, φ(y))
--- [extract false]
χ, σ;φ, bind x (extract y);stmt* ⇝ χ, σ;φ[x↦false], stmt*

```

## Finalization

These steps can be taken regardless of what statement is pending.

```rs

region_fields(χ, ι) =
  χ[∀ρ′ ∈ ρs . regions(ρ′)[parents \= {ρ}], pre_final_r ∪= ρs′]
  where
    ρ = loc(χ, ι) ∧
    ws = dom(χ(ι)) ∧
    ρs = {ρ′ | w ∈ ws ∧ (χ(ι)(w) = ι′) ∧ (ρ′ = loc(χ, ι′)) ∧ (ρ ≠ ρ′)} ∧
    ρs′ = {ρ′ | ρ′ ∈ ρs ∧ χ.regions(ρ′).stack_rc = 0}

χ₀.pre_final = {ι} ∪ ιs
τ = typeof(χ, ι)
F = P.funcs(P.types(τ).methods(final))
|F.params| = 1
typetest(χ, ι, F.params₀.type)
𝔽 ∉ dom(χ.frames)
𝔽 > φ₀.id
φ₁ = { id: 𝔽, vars: {F.paramsᵢ.name ↦ ι},
       ret: final, cont: (drop final;stmt*), condition: Return}
χ₁ = region_fields(χ₀, ι)
χ₂ = χ₁[frames ∪= 𝔽, pre_final = ιs, post_final ∪= {ι}]
--- [finalize true]
χ₀, σ;φ₀, stmt* ⇝ χ₂, σ;φ₀;φ₁, F.body

χ₀.pre_final = {ι} ∪ ιs
τ = typeof(χ, ι)
final ∉ dom(P.types(τ).methods)
χ₁ = region_fields(χ₀, ι)
χ₂ = χ₁[pre_final = ιs, post_final ∪= {ι}]
--- [finalize false]
χ₀, σ;φ, stmt* ⇝ χ₂, σ;φ, stmt*

χ.pre_final = ∅
χ.post_final = {ι} ∪ ιs
--- [collect object]
χ, σ;φ, stmt* ⇝ χ[post_final = ιs]\ι, σ;φ, stmt*

χ.pre_final = ∅
χ.pre_final_r = {ρ} ∪ {ρs}
χ′ = χ[pre_final = members(χ, ρ), pre_final_r \= {ρ}, post_final_r ∪= ρ]
--- [finalize region]
χ, σ;φ, stmt* ⇝ χ′, σ;φ, stmt*

χ.pre_final = ∅
χ.post_final_r = {ρ} ∪ {ρs}
--- [collect region]
χ, σ;φ, stmt* ⇝ χ[post_final_r = ρs]\ρ, σ;φ, stmt*

```
