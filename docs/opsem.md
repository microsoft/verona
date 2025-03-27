# Operational Semantics

Still to do:
* WIP: Extract.
* Add finalizers explicitly in the semantics.
  * Set the objects being finalized to `Immutable`?
* Discharge (send/free/freeze?) when stack RC for the region and all children (recursively) is 0.
  * Can free even if child regions have stack RC.
  * Could optimize by tracking the count of "busy" child regions.
  * Can we still have an Arena per frame?
    * Frame Arenas can't be discharged anyway.
* Efficient frame teardown.
  * Each frame could be an Arena.
  * References from a frame Arena to another region (including another frame Arena) would be tracked as stack RC.
  * A frame Arena must reach external RC 0 at `return`.
  * Prevent heap or "older frame" regions from referencing a frame Arena?
    * Any region other than the current frame Arena can't reference the current frame Arena?
* How are Arenas different from uncounted regions?
  * How should they treat changing region type?
  * How should they treat merging, freezing, extract?
* Behaviors and cowns.
* Embedded object fields?
* Arrays? Or model them as objects?
* Non-local returns.

Dynamic failures:
* `store`:
  * `w` is not a field of the target.
  * `store` to an immutable object.
  * `store` a region that already has a parent.
  * `store` a region that would create a cycle.
  * TODO: frame references?
* `call dynamic`:
  * `w` is not a method of the target.
* `merge`:
  * Trying to merge a value that isn't an object in a region.
  * Trying to merge a region that is a child of a region other than the destination region.
  * Trying to merge a region that would create a cycle.
* `freeze`:
  * Trying to freeze a value that is not an object in a region.

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
χ\ι = χ\{ι}
χ\ιs = χ[data = data\ιs, metadata = metadata\ιs]

// Regions.
ρ ∈ χ ≝ ρ ∈ dom(χ.regions)
χ[ρ↦R] = χ[regions(ρ)↦(R, ∅)]
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
  false if loc(χ, ι) = Immutable
  true if loc(χ, v) = Immutable
  // TODO: more precise frame references?
  true if loc(χ, ι) = 𝔽
  true if same_loc(χ, ι, v)
  true if (ρ₀ = loc(χ, ι)) ∧ (ρ₁ = loc(χ, v)) ∧
          (parents(χ, ρ₁) = ∅) ∧ ¬is_ancestor(χ, ρ₁, ρ₀)
  false otherwise

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

// Stack allocations are reachable only from that stack.
// TODO:
wf_stacklocal(χ, σs) =
  ∀σ₀, σ₁ ∈ σs . ∀φ ∈ σ₀ . (reachable(χ, σ₁) ∩ ιs = ∅)
  where
    ιs = {ι | loc(χ, ι) = φ.id}

// The region graph is a tree.
// TODO: examine all references
wf_regiontree(χ) =
  ∀ρ₀, ρ₁ ∈ χ .
    (|parents(χ, ρ₀)| ≤ 1) ∧
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
    χ′ = χ[regions(ρ)[type↦R]]

calc_rc(χ, σ, ρ) = calc_rc(χ, σ, members(χ, ρ))
calc_rc(χ, σ, ∅) = χ
calc_rc(χ, σ, {ι} ∪ ιs) =
  calc_rc(χ′, σ, ιs)
  where
    χ′ = calc_rc(χ, σ, ι)
calc_rc(χ, σ, ι) =
  χ[metadata(ι)[rc = calc_stack_rc(χ, σ, ι) + calc_heap_rc(χ, ι)]]

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
  χ[regions(ρ)[stack_rc = stack_rc + 1]] if loc(χ, ι) = ρ
  χ otherwise

region_stack_dec(χ, p) = χ
region_stack_dec(χ, 𝕣) = region_stack_dec(χ, 𝕣.object)
region_stack_dec(χ, ι) =
  free_region(χ, ρ) if
    (loc(χ, ι) = ρ) ∧
    (parents(χ, ρ) = ∅) ∧
    (χ.regions(ρ).stack_rc = 1)
  χ[regions(ρ)[stack_rc = stack_rc - 1]] if loc(χ, ι) = ρ
  χ otherwise

region_add_parent(χ, ι, p) = χ
region_add_parent(χ, ι, 𝕣) = region_add_parent(χ, ι, 𝕣.object)
region_add_parent(χ, ι, ι′) =
  χ[regions(ρ)[parents = parents ∪ {ρ′})]] if
    (loc(χ, ι) = ρ) ∧ (loc(χ, ι′) = ρ′) ∧ (ρ ≠ ρ′)
  χ otherwise

region_remove_parent(χ, ι, p) = χ
region_remove_parent(χ, ι, 𝕣) = region_remove_parent(χ, ι, 𝕣.object)
region_remove_parent(χ, ι, ι′) =
  χ[regions(ρ)[parents = parents \ {ρ′})]] if
    (loc(χ, ι) = ρ) ∧ (loc(χ, ι′) = ρ′) ∧ (ρ ≠ ρ′)
  χ otherwise

inc(χ, p) = χ
inc(χ, 𝕣) = dec(χ, 𝕣.object)
inc(χ, ι) =
  inc(χ, ι′) if χ.metadata(ι).location = ι′
  χ[metadata(ι)[rc = rc + 1]] if enable-rc(χ, ι)
  χ otherwise

dec(χ, p) = χ
dec(χ, 𝕣) = dec(χ, 𝕣.object)
dec(χ, ι) =
  dec(χ, ι′) if χ.metadata(ι).location = ι′
  free(χ, ι) if enable-rc(χ, ι) ∧ (χ.metadata(ι).rc = 1)
  χ[metadata(ι)[rc = rc - 1]] if enable-rc(χ, ι)
  χ otherwise

```

## Garbage Collection

```rs

// GC on RegionRC is cycle detection.
enable-gc(χ, ρ) = χ.regions(ρ).type ∈ {RegionGC, RegionRC}

gc(χ₀, σ, ρ) =
  χ₃ \ ιs₀ if enable-gc(χ₃, ρ)
  χ₀ otherwise
  where
    ιs = members(χ₀, ρ) ∧
    ιs₀ = ιs \ reachable(χ₀, gc_roots(χ₀, σ, ρ)) ∧
    ιs₁ = ιs \ ιs₀ ∧
    χ₁, ρs = gc_dec(χ₀, ιs₀, ιs₁) ∧
    χ₂ = finalize(χ₁, ιs₀) ∧
    χ₃ = free_regions(χ₂, ρs)

gc_roots(χ, σ, ρ) =
  {ι | (ι ∈ ιs) ∧ ((calc_stack_rc(χ, σ, ι) > 0) ∨ (calc_heap_rc(χ, ρs, ι) > 0))}
  where
    ρs = parents(χ, ρ) ∧ ιs = members(χ, ρ)

gc_dec(χ, ∅, ιs₁) = χ, ∅
gc_dec(χ₀, {ι} ∪ ιs₀, ιs₁) =
  χ₂, ρ₁ ∪ ρs₂
  where
    χ₁, ρs₁ = gc_dec_fields(χ₀, ι, dom(χ₀(ι)), ιs₁) ∧
    χ₂, ρs₂ = gc_dec(χ₁, ιs₀, ιs₁)
  
gc_dec_fields(χ, ι, ∅, ιs₁) = χ, ∅
gc_dec_fields(χ₀, ι, {w} ∪ ws, ιs₁) =
  χ₂, ρs₁ ∪ ρs₂
  where
    χ₁, ρs₁ = gc_dec_field(χ₀, ι, χ(ι)(w), ιs₁) ∧
    χ₂, ρs₂ = gc_dec_fields(χ₁, ι, ws, ιs₁)

gc_dec_field(χ, ι, p, ιs₁) = χ
gc_dec_field(χ, ι, 𝕣, ιs₁) = gc_dec_field(χ, ι, 𝕣.object)
gc_dec_field(χ, ι, ι′, ιs₁) =
  dec(χ, ι′), ∅ if (ι′ ∈ ιs₁) ∨ (loc(χ, ι′) = Immutable)
  χ, {ρ} if
    (loc(χ, ι′) = ρ) ∧ ¬same_loc(χ, ι, ι′) ∧ (χ.regions(ρ).stack_rc = 0)
  region_remove_parent(χ, ι, ι′), ∅ if
    (loc(χ, ι′) = ρ) ∧ ¬same_loc(χ, ι, ι′) ∧ (χ.regions(ρ).stack_rc > 0)
  χ, ∅ otherwise

```

## Free

```rs

free_regions(χ, ∅) = χ
free_regions(χ, {ρ} ∪ ρs) =
  free_regions(χ′, ρs)
  where
    χ′ = free_region(χ, ρ)

free_region(χ₀, ρ) =
  χ₂ \ ιs \ ρ
  where
    ρs = {ρ′ | (ρ′ ∈ χ) ∧ is_ancestor(χ₀, ρ, ρ′)} ∧
    ιs = members(χ₀, ρ)
    χ₁ = finalize(χ₀, ιs) ∧
    χ₂ = free_regions(χ₁, ρs)

free(χ₀, ι) =
  χ₃ \ ιs
  where
    χ₁, ιs, ρs = free_fields(χ₀, {ι}, ι) ∧
    χ₂ = finalize(χ₁, ιs) ∧
    χ₃ = free_regions(χ₂, ρs)

free_fields(χ, ιs, ι) = free_fields(χ, ιs, ι, dom(χ(ι)))
free_fields(χ, ιs, ι, ∅) = χ, ιs, ∅
free_fields(χ₀, ιs₀, ι, {w} ∪ ws) =
  χ₂, ιs₂, ρs₁ ∪ ρs₂
  where
    χ₁, ιs₁, ρs₁ = free_field(χ₀, ιs₀, ι, w) ∧
    χ₂, ιs₂, ρs₂ = free_fields(χ₁, ιs₁, ι, ws)

free_field(χ, ιs, ι, p) = χ, ιs, ∅
free_field(χ, ιs, ι, 𝕣) = free_field(χ, ιs, ι, 𝕣.object)
free_field(χ, ιs, ι, ι′) =
  χ, ιs, ∅ if ι′ ∈ ιs
  free_fields(χ, {ι′} ∪ ιs, ι′), {ι′} ∪ ιs, ∅ if
    (same_loc(χ, ι, ι′) ∨ (loc(χ, ι′) = Immutable)) ∧
    (χ.metadata(ι′).rc = 1)
  χ[metadata(ι′)[rc = rc - 1]], ιs, ∅ if
    (same_loc(χ, ι, ι′) ∨ (loc(χ, ι′) = Immutable)) ∧
    (χ.metadata(ι′).rc > 1)
  free_fields(χ, {ι′} ∪ ιs, ι′), {ι} ∪ ιs, ∅ if χ.metadata(ι′).location = ι
  χ, ιs, {ρ} if
    (loc(χ, ι′) = ρ) ∧ ¬same_loc(χ, ι, ι′) ∧ (χ.regions(ρ).stack_rc = 0)
  region_remove_parent(χ, ι, ι′), ιs, ∅ if
    (loc(χ, ι′) = ρ) ∧ ¬same_loc(χ, ι, ι′) ∧ (χ.regions(ρ).stack_rc > 0)
  χ, ιs, ∅ otherwise

```

## Finalization

```rs

finalize(χ, ∅) = χ
finalize(χ₀, {ι} ∪ ιs) =
  finalize(χ₁, ιs)
  where
    χ₁ = finalize(χ₀, ι)
finalize(χ, ι) =
  // TODO: make sure this is read-only to be resurrection-free.

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
--- [field ref]
χ, σ;ϕ, bind x (ref y w);stmt* ⇝ χ, σ;ϕ[x↦𝕣]\y, stmt*

x ∉ ϕ
ϕ(y) = {object: ι, field: w}
w ∈ dom(P.types(typeof(χ₀, ι)).fields)
v = χ₀(ι)(w)
χ₁ = region_stack_inc(χ₀, v)
χ₂ = inc(χ₁, v)
--- [load]
χ₀, σ;ϕ, bind x (load y);stmt* ⇝ χ₂, σ;ϕ[x↦v], stmt*

// TODO: what happens if safe_store is false?
x ∉ ϕ
ϕ(y) = {object: ι, field: w}
w ∈ dom(P.types(typeof(χ₀, ι)).fields)
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

## Merge

This allows merging two regions. The region being merged must either have no parent, or be a child of the region it's being merged into. If there are other stack references to the region being merged, a static type system may have the wrong region information for them.

> TODO: disallow merging a region that has a parent? Disallow merging a region that has other stack references?

```rs

x ∉ φ
loc(χ₀, φ(w)) = ρ₀
loc(χ₀, φ(y)) = ρ₁
(ρ₀ ≠ ρ₁) ∧ ¬is_ancestor(χ₀, ρ₁, ρ₀) ∧ ({ρ₀} ⊇ parents(χ₀, ρ₁))
ιs = members(χ₀, ρ₁)
χ₁ = χ₀[∀ι ∈ ιs . metadata(ι)[location↦ρ₀]]
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
χ₂ = χ₁[∀ι′ ∈ ιs . metadata(ι′)[location↦Immutable]]
--- [freeze true]
χ₀, σ;φ, bind x (freeze y);stmt* ⇝ χ₂\ρs, σ;φ[x↦true], stmt*

x ∉ φ
loc(χ, φ(y)) ≠ ρ
--- [freeze false]
χ, σ;φ, bind x (freeze y);stmt* ⇝ χ, σ;φ[x↦false], stmt*

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
