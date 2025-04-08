# Operational Semantics

Still to do:
* Should type checks be entirely removed from the semantics?
  * Or make them more powerful, able to check mutability and more?
* Discharge (send/freeze?) when stack RC for the region and all children (recursively) is 0.
  * Could optimize by tracking the count of "busy" child regions.
* How are Arenas different from uncounted regions?
  * How should they treat changing region type?
  * How should they treat merging, freezing, extract?
* Embedded object fields?
* Arrays? Or model them as objects?

## Type Checker as Optimizer

Dynamic failures that aren't trivial to eliminate with a type checker:
* `BadStore`.
* `BadReturnLoc`.
* Merge, freeze, extract failures.

## Shape

```rs

n ∈ ℕ
w, x, y, z ∈ Ident
ws, xs, ys, zs ∈ 𝒫(Ident)
τ ∈ TypeId
𝕗 ∈ FunctionId
ρ ∈ RegionId
𝔽 ∈ FrameId
ι ∈ ObjectId
ιs ∈ 𝒫(ObjectId)
π ∈ CownId
𝛽 ∈ BehaviorId
θ ∈ ThreadId

T ∈ Type = Bool | Signed × ℕ | Unsigned × ℕ | Float × ℕ | TypeId
         | Cown Type | Ref Type
         | Union Type Type | Raise Type | Throw Type

𝕥 ∈ TypeDesc =
    {
      supertypes: 𝒫(TypeId),
      fields: Ident ↦ Type,
      methods: Ident ↦ FuncId
    }

F ∈ Function =
    {
      params: {name: Ident, type: Type}*,
      result: Type,
      body: Stmt*
    }

P ∈ Program =
    {
      primitives: Type ↦ TypeId,
      types: TypeId ↦ TypeDesc,
      functions: FunctionId ↦ Function,
      globals: Ident ↦ Value
    }

// mjp: What is a reference to a Cown usage?
𝕣 ∈ Reference = {target: ObjectId | CownId, field: Ident}
    Error = BadType | BadTarget | BadField | BadStore | BadMethod | BadArgs
          | BadReturnLoc | BadReturnType

// mjp: Is this primitive values, or the type names?
// There is a confusion here.  Does `Bool` mean the type or the set of 
// values of the type?  Earlier is was the type, but here I think it is the values?
// Perhaps use NoneV, BoolV, ... instead of None, Bool, ...?
// mjp: Signed × ℕ is this a product type or the inhabitants of the type.
// /Not sure what a good syntax here is
p ∈ Primitive = None | Bool | Signed × ℕ | Unsigned × ℕ | Float × ℕ | Error
   
v ∈ Value = ObjectId | Primitive | Reference | CownId
ω ∈ Object = Ident ↦ Value

    Condition = Return | Raise | Throw
ϕ ∈ Frame =
    {
      id: FrameId,
      vars: Ident ↦ Value,
      ret: Ident,
      type: Type,
      cont: Statement*,
      condition: Condition
    }

σ ∈ Stack = Frame*

R ∈ RegionType = RegionRC | RegionGC | RegionArena

    Region = {
      type: RegionType,
      parent: RegionId | CownId | BehaviorId | None,
      stack_rc: ℕ,
      readonly: Bool
    }

    // mjp: Factored this out as a lot of the helpers use this.
    // mjp:  Should this have CownId?
    Location = RegionId | FrameId | ObjectId | Immutable

    // An object located in another object is an embedded field.
    Metadata =
    {
      type: TypeId,
      location: Location,
      rc: ℕ
    }

χ ∈ Heap =
    {
      data: ObjectId ↦ Object,
      metadata: ObjectId ↦ Metadata,
      regions: RegionId ↦ Region,
      cowns: CownId ↦ Cown,
      behaviors: BehaviorId ↦ Behavior,
      threads: ThreadId ↦ Thread,
      frames: 𝒫(FrameId),
      pre_final: 𝒫(ObjectId),
      post_final: 𝒫(ObjectId),
      pre_final_r: 𝒫(RegionId),
      post_final_r: 𝒫(RegionId)
    }

Π ∈ Cown =
    {
      type: Type,
      content: Value,
      queue: BehaviorId*,
      read: ℕ,
      write: ℕ,
      rc: ℕ
    }

B ∈ Behavior =
    {
      read: Ident ↦ CownId,
      write: Ident ↦ CownId,
      capture: Ident ↦ Value,
      body: Statement*,
      result: CownId
    }

Θ ∈ Thread =
    {
      stack: Frame*,
      cont: Statement*,
      read: 𝒫(CownId),
      write: 𝒫(CownId),
      result: CownId
    }

Heap, Stack, Statement* ⇝ Heap, Stack, Statement*

```

## Helpers
We use `.` notation for accessing fields of a record:
```rs
{label0↦value0, ... labeln↦valuen}.labeli = valuei
```

We use `[.. ↦ ..]` notation for updating a record:
```rs
r[labeli  ↦ value] = {label0 ↦ r(label0), ..., labeli ↦ value, ..., labeln ↦ r(labeln)}
  where r = {label0↦value0, ... labeln↦valuen}
```

We use `[..(..) ↦ ..]` notation for updating an element of a function component of a record:
```rs
r[label(idx) ↦ v] = r[label ↦ r(label)[idx ↦ v]]
```

We use `[.. op= ..]` notation for updating a component of a record with a specific operation:
```rs
r[labeli op= value] = r[labeli ↦ r(labeli) op value]
```

We compose updates with `[.., ..]`:
```rs
r[upd1, upd2] = r[upd1][upd2]
```

mjp: You use ϕ for frames, but here I think you are using φ for frames.  Should we just use one?
```rs
// Frames.
x ∈ φ ≝ x ∈ dom(φ.vars)
φ(x) = φ.vars(x)
φ[x↦v] = φ[vars(x)↦v]
φ\xs = φ[vars \= xs]
φ\x = φ\{x}

𝔽 ∈ χ ≝ φ ∈ dom(χ.frames)
χ∪𝔽 = χ[frames ∪= {𝔽}]
χ\𝔽 = χ[frames \= {𝔽}]

// Heap objects.
ι ∈ χ ≝ ι ∈ dom(χ.data)
χ(ι) = χ.data(ι)
χ[ι↦ω] = χ[data(ι)↦ω]
χ[ι↦(ω, τ, 𝔽)] = χ[data(ι)↦ω, metadata(ι)↦{type: τ, location: 𝔽, rc: 1}]
χ[ι↦(ω, τ, ρ)] = χ[data(ι)↦ω,
                   metadata(ι)↦{type: τ, location: ρ, rc: 1},
                   regions(ρ)[stack_rc += 1]]
χ\ι = χ\{ι}
χ\ιs = χ[data \= ιs, metadata \= ιs]

// Regions.
ρ ∈ χ ≝ ρ ∈ dom(χ.regions)
χ[ρ↦R] = χ[regions(ρ)↦{type: R, parent: None, stack_rc: 0, readonly: false}]
χ\ρs = χ[regions \= ρs]
χ\ρ = χ\{ρ}

// Cowns.
π ∈ χ ≝ π ∈ dom(χ.cowns)
χ(π) = χ.cowns(π)
χ[π↦P] = χ[cowns(π)↦P]
χ\π = χ[cowns \= {π}]

// Behaviors.
𝛽 ∈ χ ≝ 𝛽 ∈ dom(χ.behaviors)
χ(𝛽) = χ.behaviors(𝛽)
χ[𝛽↦B] = χ[behaviors(𝛽)↦B]
χ\𝛽 = χ[behaviors \= {𝛽}]

// Threads.
θ ∈ χ ≝ θ ∈ dom(χ.threads)
χ(θ) = χ.threads(θ)
χ[θ↦Θ] = χ[threads(θ)↦Θ]
χ\θ = χ[threads \= {θ}]

```

## Dynamic Types
The following definition are all implicitly passed the current program, `P`.

```rs

// Dynamic type of a value.
typeof(χ, v) =
  // mjp: Perhaps BoolV based on comment above of dual use of Bool here.
  P.primitives(Bool) if v ∈ Bool
  // mjp: Signed × ℕ shouldn't this be an element of the type? (Signed,n) rather than the product itself?
  P.primitives(Signed × ℕ) if v ∈ Signed × ℕ
  P.primitives(Unsigned × ℕ) if v ∈ Unsigned × ℕ
  P.primitives(Float × ℕ) if v ∈ Float × ℕ
  P.primitives(Error) if v ∈ Error
  χ.metadata(ι).type if ι = v
  Cown χ(π).type if π = v
  // mjp: Is the recursion well-founded here?  The argument is not clear to me, the semantics allows references in fields?
  // This definition might need to be coinductive if that is the case.
  Ref P.types(typeof(χ, ι).field(𝕣.field).type) if (𝕣 = v) ∧ (𝕣.target = ι)
  Ref χ(π).type if (𝕣 = v) ∧ (𝕣.target = π)

typetest(T₀, T₁) =
  typetest(T₂, T₁) ∧ typetest(T₃, T₁) if T₀ = Union T₂ T₃
  typetest(T₀, T₂) ∨ typetest(T₀, T₃) if T₁ = Union T₂ T₃
  T₀ = T₁ if (T₀ ∈ Ref T) ∨ (T₀ ∈ CownId T) ∨ (T₁ ∈ Ref T) ∨ (T₁ ∈ CownId T)
  // mjp: Is supertypes transitive? Does it need to be here?
  T₁ ∈ P.types(τ).supertypes if T₀ = τ
  false otherwise

typetest(χ, v, T) = typetest(typeof(χ, v), T)

```

## Reachability

```rs

// Transitive closure.
// mjp: Do you really mean forall here?  Is this a predicate?  Or is this really:
//     ⋃_(σ ∈ σs) {reachable(χ, σ)}
//     ...
reachable(χ, σs) = ∀σ ∈ σs . ⋃{reachable(χ, σ)}
reachable(χ, σ) = ∀φ ∈ σ . ⋃{reachable(χ, φ)}
reachable(χ, φ) = ∀x ∈ φ . ⋃{reachable(χ, φ(x), ∅)}

reachable(χ, ∅, ιs) = ιs
// mjp: Technically need to show this is a function as the choice of v is arbitrary.  Would need to show order is irrelevant.
reachable(χ, {v} ⊎ vs, ιs) = reachable(χ, vs, reachable(χ, v, ιs))

reachable(χ, p, ιs) = ιs
reachable(χ, π, ιs) = ιs
reachable(χ, 𝕣, ιs) = reachable(χ, 𝕣.target, ιs)
reachable(χ, ι, ιs) =
  ιs if ι ∈ ιs
  reachable(χ, ⋃_(w ∈ dom(χ(ι))) χ(ι)(w), {ι} ∪ ιs) otherwise
reachable(χ, π, ιs) = ιs

// mjp: I wonder if location is the right name here?  Location has a lot of connotations of address to me, which is not what you mean.
// I wonder if `owner` is a better term: "the owner of primitive values is the Immutable region", "the owner of a nested object is the containing object", etc.
// Location of a value.
loc(χ, p) = Immutable
// mjp:  Cown being Immutable, makes me think we should call this Sharable.
loc(χ, π) = Immutable
loc(χ, 𝕣) =
  loc(χ, 𝕣.target) if ι = 𝕣.target
  π if π = 𝕣.target
loc(χ, ι) =
  loc(χ, ι′) if χ.metadata(ι).location = ι′
  χ.metadata(ι).location if ι ∈ χ
  Immutable otherwise

same_loc(χ, v₀, v₁) = (loc(χ, v₀) = loc(χ, v₁))
members(χ, ρ) = {ι | (ι ∈ χ) ∧ (loc(χ, ι) = ρ)}

// Region parent.
parent(χ, ρ) = χ.regions(ρ).parent

// Check if ρ₀ is an ancestor of ρ₁.
is_ancestor(χ, ρ₀, ρ₁) =
  (ρ₀ = parent(χ, ρ₁)) ∨
  ((ρ = parent(χ, ρ₁) ∧ is_ancestor(χ, ρ₀, ρ)))

```

## Safety

This enforces a tree-shaped region graph, with a single reference from parent to child.

```rs
// mjp:  Does this need to account for nested/embedded objects?
safe_store(χ, Immutable, v) = false
safe_store(χ, 𝔽, v) =
  true if loc(χ, v) = Immutable
  true if loc(χ, v) = π
  true if loc(χ, v) = ρ
  true if (loc(χ, v) = 𝔽′) ∧ (𝔽 >= 𝔽′)  // MJP: What is the order on FrameIds? Is this coming from the stack in χ?
  false otherwise
safe_store(χ, ρ, v) =
  false if χ(ρ).readonly
  false if (loc(χ, v) = ρ′) ∧ χ(ρ′).readonly
  false if finalizing(χ, v)
  true if loc(χ, v) = Immutable
  true if (loc(χ, v) = ρ)
  true if (loc(χ, v) = ρ′) ∧ (parent(χ, ρ′) = None) ∧ ¬is_ancestor(χ, ρ′, ρ)
  false otherwise
safe_store(χ, π, v) =
  true if loc(χ, v) = Immutable
  true if (loc(χ, v) = ρ) ∧ (parent(χ, ρ) = None) ∧
          ¬finalizing(χ, v) ∧ ¬χ(ρ).readonly
  false otherwise
safe_store(χ, 𝛽, v) =
  true if loc(χ, v) = Immutable
  true if (loc(χ, v) = ρ) ∧ (parent(χ, ρ) = None) ∧
          ¬finalizing(χ, v) ∧ ¬χ(ρ).readonly
  false otherwise

finalizing(χ, p) = false
finalizing(χ, 𝕣) = finalizing(χ, 𝕣.target)
finalizing(χ, ι) = (ι ∈ χ.pre_final) ∨ (ι ∈ χ.post_final)
finalizing(χ, π) = false

```

## Well-Formedness

```rs

// Globals are immutable.
wf_globals(χ) =
  ∀w ∈ dom(P.globals) . (loc(χ, P.globals(w)) = Immutable)

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
  ∀ρ ∈ χ .
    (|ιs₂| ≤ 1) ∧ (|ρs| ≤ 1) ∧
    ((ρs = {ρ′}) ⇒ (ρ′ = parent(χ, ρ))) ∧
    ((ρs = ∅) ⇒ (parent(χ, ρ) ∉ RegionId))
    where
      ιs₀ = members(χ, ρ) ∧
      ιs₁ = {ι | (ι ∈ χ) ∧ (loc(χ, ι) = ρ′) ∧ (ρ ≠ ρ′)} ∧
      ιs₂ = {ι | (ι ∈ ιs₁) ∧ (w ∈ dom(χ(ι))) ∧ (χ(ι)(w) ∈ ιs₀)} ∧
      ρs = {ρ′ | (ι ∈ ιs₂) ∧ (loc(χ, ι) = ρ′)}

// The region graph is a tree.
wf_regiontree(χ) =
  ∀ρ₀, ρ₁ ∈ χ .
    (ρ₀ = parent(χ, ρ₁)) ⇒ (ρ₀ ≠ ρ₁) ∧ ¬is_ancestor(χ, ρ₁, ρ₀)

// A cown contains an immutable value or a region bound to that cown.
wf_cownvalue(χ) =
  ∀π ∈ χ .
    (loc(χ(π).value) = Immutable) ∨
    ((loc(χ(π).value) = ρ) ∧ (parent(χ, ρ) = π))

// Fields don't have Raise or Throw types.
wf_fieldtypes(P) =
  ∀τ ∈ dom(P.types) .
    ∀w ∈ dom(P.types(τ).fields) .
      P.types(τ).fields(w) ∉ {Raise T, Throw T}

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

// The heap RC from the parent region will be zero or one.
calc_heap_rc(χ, ι) =
  calc_heap_rc(χ, ρ, ι) + calc_heap_rc(χ, parent(χ, ρ), ι)
  where
    ρ = loc(χ, ι)

calc_heap_rc(χ, None, ι) = 0
calc_heap_rc(χ, 𝛽, ι) = 0
calc_heap_rc(χ, π, ι) = 0
calc_heap_rc(χ, ρ, ι) =
  |{(ι′, w) |
    (ι′ ∈ members(χ, ρ)) ∧
    (w ∈ dom(χ(ι′))) ∧
    ((χ(ι′)(w) = ι)) ∨ ((χ(ι′)(w) = 𝕣) ∧ (𝕣.target = ι))}|

```

## Reference Counting

Reference counting is a no-op unless the object is in a `RegionRC` or is `Immutable`.

```rs

enable-rc(χ, ι) =
  ((loc(χ, ι)) = ρ ∧ (ρ.readonly = false) ∧ (ρ.type = RegionRC)) ∨
  (loc(χ, ι) = Immutable)

region_stack_inc(χ, p) = χ
region_stack_inc(χ, π) = χ
region_stack_inc(χ, 𝕣) = region_stack_inc(χ, 𝕣.target)
region_stack_inc(χ, ι) =
  χ[regions(ρ)[stack_rc += 1]] if (loc(χ, ι) = ρ)
  χ otherwise

region_stack_dec(χ, p) = χ
region_stack_dec(χ, π) = χ
region_stack_dec(χ, 𝕣) = region_stack_dec(χ, 𝕣.target)
region_stack_dec(χ, ι) =
  χ[pre_final_r ∪= {ρ}] if
    (loc(χ, ι) = ρ) ∧
    (parent(χ, ρ) = None) ∧
    (χ.regions(ρ).stack_rc = 1)
  χ[regions(ρ)[stack_rc -= 1]] if (loc(χ, ι) = ρ)
  χ otherwise

region_add_parent(χ, ι, p) = χ
region_add_parent(χ, ι, π) = χ
region_add_parent(χ, ι, 𝕣) = region_add_parent(χ, ι, 𝕣.target)
region_add_parent(χ, ι, ι′) =
  χ[regions(ρ′)[parent = ρ]] if
    (loc(χ, ι) = ρ) ∧ (loc(χ, ι′) = ρ′) ∧ (ρ ≠ ρ′)
  χ[regions(ρ′)[stack_rc += 1]] if (loc(χ, ι) = 𝔽) ∧ (loc(χ, ι′) = ρ′)
  χ otherwise

region_add_parent(χ, π, p) = χ
region_add_parent(χ, π, π′) = χ
region_add_parent(χ, π, 𝕣) = region_add_parent(χ, ι, 𝕣.target)
region_add_parent(χ, π, ι) =
  χ[regions(ρ)[parent = π]] if loc(χ, ι) = ρ
  χ otherwise

region_remove_parent(χ, ι, p) = χ
region_remove_parent(χ, ι, π) = χ
region_remove_parent(χ, ι, 𝕣) = region_remove_parent(χ, ι, 𝕣.target)
region_remove_parent(χ, ι, ι′) =
  χ[regions(ρ′)[parent = None]] if
    (loc(χ, ι) = ρ) ∧ (loc(χ, ι′) = ρ′) ∧ (ρ ≠ ρ′)
  χ[regions(ρ′)[stack_rc -= 1]] if (loc(χ, ι) = 𝔽) ∧ (loc(χ, ι′) = ρ′)
  χ otherwise

region_remove_parent(χ, π, p) = χ
region_remove_parent(χ, π, π′) = χ
region_remove_parent(χ, π, 𝕣) = region_remove_parent(χ, ι, 𝕣.target)
region_remove_parent(χ, π, ι) =
  χ[regions(ρ)[parent = None]] if loc(χ, ι) = ρ
  χ otherwise

inc(χ, p) = χ
inc(χ, π) = χ[cowns(π)[rc += 1]]
inc(χ, 𝕣) =
  inc(χ, 𝕣.target) if ι = 𝕣.target
  χ if π = 𝕣.target
inc(χ, ι) =
  inc(χ, ι′) if χ.metadata(ι).location = ι′
  χ[metadata(ι)[rc += 1]] if enable-rc(χ, ι)
  χ otherwise

dec(χ, p) = χ
dec(χ, π) = χ[cowns(π)[rc -= 1]] // TODO: free
dec(χ, 𝕣) =
  dec(χ, 𝕣.target) if ι = 𝕣.target
  χ if π = 𝕣.target
dec(χ, ι) =
  dec(χ, ι′) if χ.metadata(ι).location = ι′
  χ[pre_final ∪= {ι}] if enable-rc(χ, ι) ∧ (χ.metadata(ι).rc = 1)
  χ[metadata(ι)[rc -= 1]] if enable-rc(χ, ι)
  χ otherwise

```

## Garbage Collection

```rs

// mjp: Should we consider behaviours as roots too?
// mjp: Should we consider cowns as roots, or should we track there reachability?

// GC on RegionRC is cycle detection.
enable-gc(χ, ρ) = χ.regions(ρ).type ∈ {RegionGC, RegionRC}

gc(χ, σ, ρ) =
  χ′[pre_final ∪= ιs₀] if enable-gc(χ₃, ρ)
  χ otherwise
  where
    ιs = members(χ₀, ρ) ∧
    // TODO: doesn't work with finalization.
    // need to keep everything we might look at during finalization alive.
    // if A can reach B, and we select B but not A, then we can't finalize A.
    ιs₀ ⊆ ιs \ reachable(χ₀, gc_roots(χ₀, σ, ρ)) ∧
    ιs₁ = ιs \ ιs₀ ∧
    χ′ = gc_dec(χ, ιs₀, ιs₁)

gc_roots(χ, σ, ρ) =
  {ι | (ι ∈ members(χ, ρ)) ∧
       ((calc_stack_rc(χ, σ, ι) > 0) ∨
        (calc_heap_rc(χ, parent(χ, ρ), ι) > 0))}

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
gc_dec_field(χ, ι, π, ιs₁) = χ
gc_dec_field(χ, ι, 𝕣, ιs₁) = gc_dec_field(χ, ι, 𝕣.target)
gc_dec_field(χ, ι, ι′, ιs₁) =
  dec(χ, ι′) if (ι′ ∈ ιs₁) ∨ (loc(χ, ι′) = Immutable)
  χ otherwise

```

## Free

```rs

// TODO: only call free_fields after finalizing the object.
// remove this, put object into pre_final directly from `dec`.
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
free_field(χ, ιs, ι, π) = χ, ιs
free_field(χ, ιs, ι, 𝕣) = free_field(χ, ιs, ι, 𝕣.target)
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

## Global Values

```rs

x ∉ φ
v = P.globals(y)
χ₁ = inc(χ₀, v)
--- [global]
χ₀, σ;φ, bind x (global y);stmt* ⇝ χ₁, σ;φ[x↦v], stmt*

```

## New

For an "address-taken" local variable, i.e. a `var` as opposed to a `let`, allocate an object in the frame with a single field to hold the value.

```rs

once(x*) = |{x | x ∈ x*}| = |x*|
once((x, y)*) = |{y | y ∈ (x, y)*}| = |(x, y)*|

newobject(φ, (y, z)*) = {y ↦ φ(z) | y ∈ (y, z)*}

typecheck(χ, τ, ω) =
  (dom(P.types(τ).fields) = dom(ω)) ∧
  ∀w ∈ dom(ω) . typetest(χ, ω(w), P.types(τ).fields(w))

x ∉ φ
--- [new primitive]
χ, σ;φ, bind x (new p);stmt* ⇝ χ, σ;φ[x↦p], stmt*

x ∉ φ
ι ∉ χ
once((y, z)*)
∀z ∈ (y, z)* . safe_store(χ, φ.id, φ(z))
ω = newobject(φ, (y, z)*)
typecheck(χ, τ, ω)
--- [new stack]
χ, σ;φ, bind x (new τ (y, z)*);stmt* ⇝ χ[ι↦(ω, τ, φ.id)], σ;φ[x↦ι]\zs, stmt*

x ∉ φ
∃z ∈ (y, z)* . ¬safe_store(χ, φ.id, φ(z))
--- [new stack bad-store]
χ, σ;φ, bind x (new τ (y, z)*);stmt* ⇝ χ, σ;φ[x↦BadStore], throw;return x

x ∉ φ
ω = newobject(φ, (y, z)*)
¬once((y, z)*) ∨ ¬typecheck(χ, τ, ω)
--- [new stack bad-type]
χ, σ;φ, bind x (new τ (y, z)*);stmt* ⇝ χ, σ;φ[x↦BadType], throw;return x

x ∉ φ
ι ∉ χ
ι′ = φ(w)
ρ = loc(χ, ι′)
once((y, z)*)
ω = newobject(φ, (y, z)*)
typecheck(χ, τ, ω)
--- [new heap]
χ, σ;φ, bind x (new w τ (y, z)*);stmt* ⇝ χ[ι↦(ω, τ, ρ)], σ;φ[x↦ι]\zs, stmt*

x ∉ φ
(ι′ ≠ φ(w)) ∨ (ρ ≠ loc(χ, ι′))
--- [new heap bad-target]
χ, σ;φ, bind x (new w τ (y, z)*);stmt* ⇝ χ, σ;φ[x↦BadTarget], throw;return x

x ∉ φ
ι′ = φ(w)
ρ = loc(χ, ι′)
∃z ∈ (y, z)* . ¬safe_store(χ, ρ, φ(z))
--- [new heap bad-store]
χ, σ;φ, bind x (new τ (y, z)*);stmt* ⇝ χ, σ;φ[x↦BadStore], throw;return x

x ∉ φ
ι′ = φ(w)
ρ = loc(χ, ι′)
ω = newobject(φ, (y, z)*)
¬once((y, z)*) ∨ ¬typecheck(χ, τ, ω)
--- [new heap bad-type]
χ, σ;φ, bind x (new w τ (y, z)*);stmt* ⇝ χ, σ;φ[x↦BadType], throw;return x

x ∉ φ
ι ∉ χ
ρ ∉ χ
once((y, z)*)
ω = newobject(φ, (y, z)*)
typecheck(χ, τ, ω)
--- [new region]
χ, σ;φ, bind x (new R τ (y, z)*);stmt* ⇝ χ[ρ↦R][ι↦(ω, τ, ρ)], σ;φ[x↦ι]\zs, stmt*

x ∉ φ
ρ ∉ χ
∃z ∈ (y, z)* . ¬safe_store(χ, ρ, φ(z))
--- [new heap bad-store]
χ, σ;φ, bind x (new R τ (y, z)*);stmt* ⇝ χ, σ;φ[x↦BadStore], throw;return x

x ∉ φ
ω = newobject(φ, (y, z)*)
¬once((y, z)*) ∨ ¬typecheck(χ, τ, ω)
--- [new region bad-type]
χ, σ;φ, bind x (new R τ (y, z)*);stmt* ⇝ χ, σ;φ[x↦BadType], throw;return x

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
𝕣 = {target: ϕ(y), field: w}
--- [ref]
χ, σ;ϕ, bind x (ref y w);stmt* ⇝ χ, σ;ϕ[x↦𝕣]\y, stmt*

x ∉ ϕ
ϕ(y) ∉ ObjectId
--- [ref bad-target]
χ, σ;ϕ, bind x (ref y w);stmt* ⇝ χ, σ;ϕ[x↦BadTarget]\y, throw;return x

x ∉ ϕ
ι = ϕ(y)
w ∉ dom(P.types(typeof(χ, ι)).fields)
--- [ref bad-field]
χ, σ;ϕ, bind x (ref y w);stmt* ⇝ χ, σ;ϕ[x↦BadField]\y, throw;return x

x ∉ ϕ
𝕣 = φ(y)
v = χ₀(ι)(w) if 𝕣 = {target: ι, field: w}
    χ₀(π).value if 𝕣 = {target: π, field: w}
χ₁ = region_stack_inc(χ₀, v)
χ₂ = inc(χ₁, v)
--- [load]
χ₀, σ;ϕ, bind x (load y);stmt* ⇝ χ₂, σ;ϕ[x↦v], stmt*

x ∉ ϕ
ϕ(y) ∉ Reference
--- [load bad-target]
χ, σ;ϕ, bind x (load y);stmt* ⇝ χ, σ;ϕ[x↦BadTarget], throw;return x

x ∉ ϕ
𝕣 = φ(y)
v₀ = φ(z)
safe_store(χ₀, loc(χ₀, 𝕣.target), v₀)
v₁, χ₁ = ω(w), χ₀[ι↦ω[w↦v₀]] if
            (𝕣 = {target: ι, field: w}) ∧ (ω = χ₀(ι)) ∧
            typetest(χ₀, v₀, P.types(typeof(χ₀, ι)).fields(w))
         Π.value, χ₀[π↦Π[value↦v₀]] if
            (𝕣 = {target: π, field: w}) ∧ (Π = χ₀(π)) ∧
            typetest(χ₀, v₀, Π.type)
χ₂ = region_stack_inc(χ₁, v₁)
χ₃ = region_remove_parent(χ₃, 𝕣.target, v₁)
χ₄ = region_add_parent(χ₃, 𝕣.target, v₀)
χ₅ = region_stack_dec(χ₄, v₀)
--- [store]
χ₀, σ;ϕ, bind x (store y z);stmt* ⇝ χ₅, σ;ϕ[x↦v₁]\z, stmt*

x ∉ ϕ
ϕ(y) ∉ Reference
--- [store bad-target]
χ, σ;ϕ, bind x (store y z);stmt* ⇝ χ, σ;ϕ[x↦BadTarget], throw;return x

x ∉ ϕ
𝕣 = φ(y)
v = φ(z)
¬safe_store(χ₀, loc(χ, 𝕣.target), v₁)
--- [store bad-store]
χ, σ;ϕ, bind x (store y z);stmt* ⇝ χ, σ;ϕ[x↦BadStore], throw;return x

x ∉ ϕ
𝕣 = φ(y)
v = φ(z)
((𝕣 = {target: ι, field: w}) ∧
  ¬typetest(χ₀, v, P.types(typeof(χ₀, 𝕣.target)).fields(w))) ∨
((𝕣 = {target: π, field: w}) ∧
  ¬typetest(χ₀, v₀, Π.type))
--- [store bad-type]
χ, σ;ϕ, bind x (store y z);stmt* ⇝ χ, σ;ϕ[x↦BadType], throw;return x

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
    ret: x, type: F.result, cont: stmt*, condition: Return}
  where
    (𝔽 ∉ dom(χ.frames)) ∧ (𝔽 > φ.id)

typecheck(χ, φ, F, y*) =
  |F.params| = |y*| ∧
  ∀i ∈ 1 .. |y*| . typetest(χ, φ(yᵢ), F.paramsᵢ.type)

x ∉ φ₀
once(y*)
F = P.functions(𝕗)
typecheck(χ, φ₀, F, y*)
φ₁ = newframe(χ, φ₀, F, x, y*, stmt*)
--- [call static]
χ, σ;φ₀, bind x (call 𝕗 y*);stmt* ⇝ χ∪(φ₁.id), σ;φ₀\{y*};φ₁, F.body

x ∉ φ
once(y*)
F = P.functions(𝕗)
¬typecheck(χ, φ, F, y*)
--- [call static bad-args]
χ, σ;φ, bind x (call w y*);stmt* ⇝ χ, σ;φ[x↦BadArgs], throw;return x

x ∉ φ₀
once(y*)
τ = typeof(χ, φ₀(y₁))
F = P.functions(P.types(τ).methods(w))
typecheck(χ, φ₀, F, y*)
φ₁ = newframe(χ, φ₀, F, x, y*, stmt*)
--- [call dynamic]
χ, σ;φ₀, bind x (call w y*);stmt* ⇝ χ∪(φ₁.id), σ;φ₀\{y*};φ₁, F.body

x ∉ φ
once(y*)
τ ≠ typeof(χ, φ(y₁))
--- [call dynamic bad-target]
χ, σ;φ, bind x (call w y*);stmt* ⇝ χ, σ;φ[x↦BadTarget], throw;return x

x ∉ φ
once(y*)
τ = typeof(χ, φ(y₁))
w ∉ P.types(τ).methods
--- [call dynamic bad-method]
χ, σ;φ, bind x (call w y*);stmt* ⇝ χ, σ;φ[x↦BadMethod], throw;return x

x ∉ φ
once(y*)
τ = typeof(χ, φ(y₁))
F = P.functions(P.types(τ).methods(w))
¬typecheck(χ, φ, F, y*)
--- [call dynamic bad-args]
χ, σ;φ, bind x (call w y*);stmt* ⇝ χ, σ;φ[x↦BadArgs], throw;return x

```

## Return

This drops any remaining frame variables other than the return value.

```rs

dom(φ₁.vars) = {x}
v = φ₁(x)
loc(χ, v) ≠ φ₁.id
T = typeof(χ, v) if φ₁.condition = Return
    Raise typeof(χ, v) if φ₁.condition = Raise
    Throw typeof(χ, v) if φ₁.condition = Throw
typetest(T, φ.type)
φ₂ = φ₀[φ₁.ret↦v, condition = φ₁.condition]
--- [return]
χ, σ;φ₀;φ₁, return x;stmt* ⇝ χ\(φ₁.id), σ;φ₂, ϕ₁.cont

dom(φ.vars) = {x}
v = φ(x)
loc(χ₀, v) ≠ φ.id
T = typeof(χ, v) if φ₁.condition = Return
    Raise typeof(χ, v) if φ₁.condition = Raise
    Throw typeof(χ, v) if φ₁.condition = Throw
typetest(T, φ.type)
π = φ(final)
safe_store(χ₀, π, v)
χ₁ = χ₀[cowns(π)[value↦v]]
χ₂ = region_add_parent(χ₁, π, v)
χ₃ = region_stack_dec(χ₂, v)
--- [return]
χ₀, φ, return x;stmt* ⇝ χ₃\φ.id, ∅, ∅

dom(φ.vars) = {x, y} ∪ zs
--- [return]
χ, σ;φ, return x;stmt* ⇝ χ, σ;φ, drop y;return x

dom(φ.vars) = {x}
v = φ(x)
(loc(χ, v) = φ.id) ∨ ((π = φ(final)) ∧ ¬safe_store(χ, π, v))
--- [return bad-loc]
χ, σ;φ, return x;stmt* ⇝ χ, σ;φ[y↦BadReturnLoc], throw;return y

dom(φ.vars) = {x}
v = φ(x)
loc(χ, v) ≠ φ.id
T = typeof(χ, v) if φ₁.condition = Return
    Raise typeof(χ, v) if φ₁.condition = Raise
    Throw typeof(χ, v) if φ₁.condition = Throw
¬typetest(T, φ.type)
--- [return bad-type]
χ, σ;φ, return x;stmt* ⇝ χ, σ;φ[y↦BadReturnType], throw;return y

```

## Non-Local Return

Use `raise` before a return for a non-local return, and `throw` for an error.

Use `reraise` after a `call` from inside a Smalltalk style block, such as a Verona lambda. This propagates both non-local returns and errors. Use `rethrow` after a `call` from inside a function. This returns a non-local return as local, and propagates errors. Use `catch` instead of either to capture a non-local return or error without propagating it.

```rs

--- [raise]
χ, σ;φ, raise;stmt* ⇝ χ, σ;φ[condition = Raise], stmt*

--- [throw]
χ, σ;φ, throw;stmt* ⇝ χ, σ;φ[condition = Throw], stmt*

--- [catch]
χ, σ;φ, catch;stmt* ⇝ χ, σ;φ[condition = Return], stmt*

x ∈ φ
φ.condition = Return
--- [reraise]
χ, σ;φ, reraise x;stmt* ⇝ χ, σ;φ, stmt*

x ∈ φ
φ.condition ≠ Return
--- [reraise]
χ, σ;φ, reraise x;stmt* ⇝ χ, σ;φ, return x

x ∉ φ
φ.condition = Return
--- [rethrow]
χ, σ;φ, rethrow x;stmt* ⇝ χ, σ;φ, stmt*

x ∉ φ
φ.condition = Raise
--- [rethrow]
χ, σ;φ, rethrow x;stmt* ⇝ χ, σ;φ[condition = Return], return x

x ∉ φ
φ.condition = Throw
--- [rethrow]
χ, σ;φ, rethrow x;stmt* ⇝ χ, σ;φ, return x

```

## Merge

This allows merging two regions. The region being merged must either have no parent, or be a child of the region it's being merged into. If there are other stack references to the region being merged, a static type system may have the wrong region information for them.

> TODO: Disallow merging a region that has other stack references?

```rs

x ∉ φ
ι₀ = φ(w)
ι₁ = φ(y)
loc(χ₀, ι₀) = ρ₀
loc(χ₀, ι₁) = ρ₁
(ρ₀ ≠ ρ₁) ∧ (parent(χ₀, ρ₁) = None)
ιs = members(χ₀, ρ₁)
χ₁ = χ₀[∀ι ∈ ιs . metadata(ι)[location = ρ₀]]
       [regions(ρ₀)[stack_rc += regions(ρ₁).stack_rc]]
--- [merge]
χ₀, σ;φ, bind x (merge w y);stmt* ⇝ χ₁\ρ₁, σ;φ[x↦φ(y)], stmt*

x ∉ φ
(ι₀ ≠ φ(w)) ∨ (ι₁ ≠ φ(y)) ∨
(loc(χ, φ(w)) ≠ ρ₀) ∨ (loc(χ, φ(y)) ≠ ρ₁) ∨
(ρ₀ = ρ₁) ∨ (parent(χ, ρ₁) ≠ None)
--- [merge bad-target]
χ, σ;φ, bind x (merge w y);stmt* ⇝ χ, σ;φ[x↦BadTarget], throw;return x

```

## Freeze

If the region being frozen has a parent, a static type system may have the wrong type for the incoming reference. If there are other stack references to the region being frozen or any of its children, a static type system may have the wrong type for them.

> TODO: Disallow freezing a region that has other stack references?

```rs

x ∉ φ
ι = φ(y)
ρ = loc(χ₀, ι)
parent(χ₀, ρ) = None
ρs = {ρ} ∪ {ρ′ | (ρ′ ∈ χ.regions) ∧ is_ancestor(χ₀, ρ, ρ′)}
χ₁ = region_type_change(χ₀, σ;φ, ρs, RegionRC)
ιs = {ι′ | loc(χ₀, ι′) ∈ ρs}
χ₂ = χ₁[∀ι′ ∈ ιs . metadata(ι′)[location = Immutable]]
--- [freeze true]
χ₀, σ;φ, bind x (freeze y);stmt* ⇝ χ₂\ρs, σ;φ[x↦ι]\y, stmt*

x ∉ φ
(ι ≠ φ(y)) ∨ (loc(χ, ι) ≠ ρ) ∨ (parent(χ, ρ) ≠ None)
--- [freeze false]
χ, σ;φ, bind x (freeze y);stmt* ⇝ χ, σ;φ[x↦BadTarget], throw;return x

```

## Extract

```rs

x ∉ φ
ι = φ(y)
ρ₀ = loc(χ₀, ι)
ρ₁ ∉ χ₀
ιs = reachable(χ, ι) ∩ members(χ₀, ρ₀)
|{ι | (ι ∈ members(χ₀, ρ₀)) ∧ (w ∈ dom(χ₀(ι))) ∧
      (χ₀(ι)(w) = ι′) ∧ (ι′ ∈ ιs)}| = 0
ρs = {ρ |
      (ι ∈ ιs) ∧ (w ∈ dom(χ(ι))) ∧ (χ(ι)(w) = ι′) ∧
      (ρ = loc(χ, ι′)) ∧ (ρ ≠ ρ₀)}
rc = calc_stack_rc(χ₀, σ;φ, ιs)
χ₁ = χ₀[regions(ρ₀)[stack_rc -= rc],
        regions(ρ₁)↦{ type: χ.regions(ρ₀).type, parent: None,
                       stack_rc: rc, readonly: false },
        ∀ι′ ∈ ιs . metadata(ι′)[location = ρ₁],
        ∀ρ ∈ ρs . regions(ρ)[parent = ρ₁]]
--- [extract]
χ₀, σ;φ, bind x (extract y);stmt* ⇝ χ₁, σ;φ[x↦ι]\y, stmt*

x ∉ φ
(ι ≠ φ(y)) ∨ (ρ ≠ loc(χ, φ(y)))
--- [extract bad-target]
χ, σ;φ, bind x (extract y);stmt* ⇝ χ, σ;φ[x↦BadTarget], throw;return x

x ∉ φ
ι = φ(y)
ρ = loc(χ, ι)
ιs = reachable(χ, ι) ∩ members(χ, ρ)
|{ι | (ι ∈ members(χ, ρ)) ∧ (w ∈ dom(χ(ι))) ∧
      (χ(ι)(w) = ι′) ∧ (ι′ ∈ ιs)}| > 0
--- [extract bad-target]
χ, σ;φ, bind x (extract y);stmt* ⇝ χ, σ;φ[x↦BadTarget], throw;return x

```

## Finalization

These steps can be taken regardless of what statement is pending.

```rs

region_fields(χ, ι) =
  χ[∀ρ′ ∈ ρs . regions(ρ′)[parent = None], pre_final_r ∪= ρs′]
  where
    ρ = loc(χ, ι) ∧
    ws = dom(χ(ι)) ∧
    ρs = {ρ′ | w ∈ ws ∧ (χ(ι)(w) = ι′) ∧ (ρ′ = loc(χ, ι′)) ∧ (ρ ≠ ρ′)} ∧
    ρs′ = {ρ′ | ρ′ ∈ ρs ∧ χ.regions(ρ′).stack_rc = 0}

χ₀.pre_final = {ι} ∪ ιs
τ = typeof(χ, ι)
F = P.functions(P.types(τ).methods(final))
|F.params| = 1
typetest(χ, ι, F.params₀.type)
𝔽 ∉ dom(χ.frames)
𝔽 > φ₀.id
φ₁ = { id: 𝔽, vars: {F.paramsᵢ.name ↦ ι},
       ret: final, type: F.result, cont: (drop final;stmt*), condition: Return}
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

## Behaviors

```rs

ready(χ, 𝛽) =
  (∀π ∈ πs . (χ(π).queue = 𝛽;𝛽*) ∧ χ(π).write = 0) ∧
  (∀π ∈ χ(𝛽).write . χ(π).read = 0) ∧
  (∀ρ ∈ ρs′ . χ(ρ).stack_rc = 0)
  where
    (πs = {π | π ∈ (χ(𝛽).read ∪ χ(𝛽).write ∪ {χ(𝛽).result})}) ∧
    (ρs = {ρ | (ι ∈ χ(𝛽).capture) ∧ (loc(χ, ι) = ρ)}) ∧
    (ρs′ = {ρ′| (ρ ∈ ρs) ∧ (ρ′ ∈ χ) ∧ is_ancestor(χ, ρ, ρ′)})

mark-readonly(χ, π) =
  mark-readonly(χ, {ρ} ∪ {ρ′ | ρ′ ∈ χ ∧ is_ancestor(ρ, ρ′)}) if
    ρ = loc(χ, χ(π).value)
  χ otherwise
mark-readonly(χ, {ρ} ∪ ρs) =
  mark-readonly(χ′, ρs)
  where
    χ′ = mark-readonly(χ, ρ)
mark-readonly(χ, ρ) = χ[regions(ρ)[readonly = true]]

unmark-readonly(χ, π) =
  unmark-readonly(χ, {ρ} ∪ {ρ′ | ρ′ ∈ χ ∧ is_ancestor(ρ, ρ′)}) if
    ρ = loc(χ, χ(π).value)
  χ otherwise
unmark-readonly(χ, {ρ} ∪ ρs) =
  unmark-readonly(χ′, ρs)
  where
    χ′ = unmark-readonly(χ, ρ)
unmark-readonly(χ, ρ) = χ[regions(ρ)[readonly = false]]

read-inc(χ, ∅) = χ
read-inc(χ, {π} ∪ πs) =
  read-inc(χ′, πs)
  where
    χ′ = read-inc(χ, π)
read-inc(χ, π) =
  χ′[cowns(π)[queue = 𝛽*, read += 1]]
  where
    χ(π).queue = 𝛽;𝛽 ∧
    χ′ = mark-readonly(χ, π)

write-inc(χ, ∅) = χ
write-inc(χ, {π} ∪ πs) =
  write-inc(χ′, πs)
  where
    χ′ = write-inc(χ, π)
write-inc(χ, π) =
  χ[cowns(π)[queue = 𝛽*, write += 1]]
  where
    χ(π).queue = 𝛽;𝛽*

read-dec(χ, ∅) = χ
read-dec(χ, {π} ∪ πs) =
  read-dec(χ′, πs)
  where
    χ′ = read-dec(χ, π)
read-dec(χ, π) =
  χ′[cowns(π)[rc -= 1, read -= 1]] // TODO: free
  where
    χ′ = unmark-readonly(χ, π)

write-dec(χ, ∅) = χ
write-dec(χ, {π} ∪ πs) =
  write-dec(χ′, πs)
  where
    χ′ = write-dec(χ, π)
write-dec(χ, π) = χ[cowns(π)[rc -= 1, write -= 1]] // TODO: free

read-acquire(φ, ∅) = φ
read-acquire(φ, ω) =
  read-acquire(φ′, ω\x)
  where
    x ∈ dom(ω) ∧
    π = ω(x) ∧
    φ′ = φ[x↦χ(π).value]

write-acquire(φ, ∅) = φ
write-acquire(φ, ω) =
  write-acquire(φ′, ω\x)
  where
    x ∈ dom(ω) ∧
    π = ω(x) ∧
    φ′ = φ[x↦{target: π, field: final}]

// TODO: regions put in a behavior need to set a parent to prevent them being put anywhere else.
// what if z* contains multiple objects in the same region, and that region has no parent? is that ok?
// stack_rc isn't going to 0 here, as it's being moved to the new thread.
x ∉ φ
𝛽 ∉ χ
π ∉ χ
once(w*;y*;z*)
∀w ∈ w* . φ(w) ∈ CownId
∀y ∈ y* . φ(y) ∈ CownId
∀z ∈ z* . safe_store(χ, 𝛽, φ(z))
πs = {φ(x′) | (x′ ∈ w*;y*)} ∪ {π}
χ′ = χ[∀π′ ∈ πs . cowns(π′)[queue ++ 𝛽]]
Π = { type: T, value: None, queue: 𝛽 }
B = { read: {w ↦ φ(w) | w ∈ w*},
      write: {y ↦ φ(y) | y ∈ y*},
      capture: {z ↦ φ(z) | z ∈ z*},
      body: stmt₀*,
      result: π }
--- [when]
χ, σ;φ, bind x (when T (read w*) (write y*) (capture z*) stmt₀*);stmt₁* ⇝
  χ′[π↦Π, 𝛽↦B]∪𝔽, σ;φ[x↦π]\(w*;y*;z*), stmt₁*

𝛽 ∈ χ₀
θ ∉ χ₀
𝔽 ∉ χ₀
ready(χ₀, 𝛽)
πs₀ = {π′ | π′ ∈ χ₀(𝛽).read} \ πs₁
πs₁ = {π′ | π′ ∈ χ₀(𝛽).write}
π = χ₀(𝛽).result
φ₀ = { id: 𝔽,
       vars: {x ↦ χ₀(𝛽).capture(x) | x ∈ dom(χ₀(𝛽).capture)},
       ret: final,
       type: χ₀(π).type,
       cont: ∅,
       condition: Return }
Θ = { stack: φ₂[final↦π],
      cont: χ₀(𝛽).body,
      read: πs₀
      write: πs₁
      result: π }
φ₁ = read-acquire(φ₀, χ₀(𝛽).read)
φ₂ = write-acquire(φ₁, χ₀(𝛽).write)
χ₁ = read-inc(χ₀, Θ.read)
χ₂ = write-inc(χ₁, Θ.write ∪ {π})
--- [start thread]
χ₀ ⇝ χ₂[θ↦Θ]\𝛽

θ ∈ χ
χ, χ(θ).stack, χ(θ).cont ⇝ χ′, σ′, stmt′*
--- [step thread]
χ ⇝ χ′[threads(θ)[stack = σ′, cont = stmt′*]]

θ ∈ χ
χ(θ) = {stack: ∅, cont: ∅, read: πs₀, write: πs₁, result: π}
χ₁ = read-dec(χ₀, πs₀)
χ₂ = write-dec(χ₁, πs₁ ∪ {π})
--- [end thread]
χ₀ ⇝ χ₂\θ

```
