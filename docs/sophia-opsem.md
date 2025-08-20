# Operational Semantics 
No finalization right now
Immediate TODOs:
* Get required definitions from opsem doc
* Rewrite Call semantics to split lookup and call 
* Probably want some other blocks so that non local return actually makes sense
* Non Local Return


## Definitions 
Beginning with a sequential (no regions, no behaviors etc semantics for non local return only)
```rs 

Ident := //Give some definition of what an identifier actually looks like (regex string)
TypeId := //Give some definition of what these look like 
ObjectId := //Give some definition of what these look like
FunctionId := //Give some definition fo what these look like
FrameId := //Give some definition of what these look like
RegionId := 
n ∈ ℕ
w, x, y, z ∈ Ident
// An element of the powerset over identifiers, so a set of identifiers
ws, xs, ys, zs ∈ 𝒫(Ident)
τ ∈ TypeId
𝕗 ∈ FunctionId
ι ∈ ObjectId
ρ ∈ RegionId
𝔽 ∈ FrameId
ιs ∈ 𝒫(ObjectId)
Stmt := //Give the list of statements here


Type := TNone | Bool | TypeId | Fun (Type * Type * ... Type) * Type // No refs for now 
T ∈ Type
// User Defined Types
TypeDesc := 
{
    supertypes : 𝒫(TypeId) 
    fields : Ident ↦ Type 
    methods: Ident ↦ FunctionId
}
// Each function output type is given as a record: the type the function can return, the type the function could raise, and the type the function can throw. All are optional (if a function only returns and never raises, it should only be given a return type, and None for the others etc)
Function_Body := 
    {
        params : {name : Ident, type : T}* //input params x:T
        result : {return : Opt T, raise : Opt T, throw : Opt T} // return, raise, and throw types for the function (None or Some T, where None and Some are on the meta level, not on the type level)
        body : Stmt* //body of function
    }
Function_Type := 
    params : T*
    result :  {return : Opt T, raise : Opt T, throw : Opt T} // Not quite a union type? 
F ∈ Function

Program := 
    {
        primitives :  Type ↦ TypeId 
        types : TypeId ↦ TypeDesc // Type Defs
        function_types : FunctionId ↦ 
        functions: FunctionId ↦ Function // Function Defs
        globals : Ident ↦ Value // Global vars
    }

P ∈ Program



Primitive := PNone | PTrue | PFalse  // Drop None from here?
p ∈ Primitive

Values := ObjectId | Primitive | FunctionId 
v ∈ Value




RegionType := RegionRC | RegionArena
RT ∈ RegionType

// None not really a location

Location := None | RegionId | Immutable | FrameId // Maybe find better name, still don't like location
L ∈ Locatio



Region := 
{
    type: RegionType
    parent: RegionId | None | FrameId // (Frame id if local region only) 
    stack_rc : ℕ // if type is not RegionRC (can be arbitrary if not a ref counted region)
}
R ∈ Region




// This is basically a dictionary (fields in an object {x = A, y = B} etc)
Object := Ident ↦ Value 
ω ∈ Object

ObjectInformation = 
    {type : TypeID
    location : Location
    rc : ℕ}

ωi ∈ ObjectInformation

CallType = Call | Subcall | Catch

Frame :=
    {
      id: FrameId, 
      vars: Ident ↦ Value,
      ret: Ident,
      type: Type,
      cont: Statement*,
      calltype: CallType
      region : RegionId
    }
φ ∈ Frame

Stack := Frame*
σ ∈ Stack

Heap := 
    {
        // Each ObjectId maps to an object 
        data : ObjectId ↦ Object 
        // Each ObjectId is also associated with some information about that object
        metadata_obj : ObjectId ↦ ObjectInformation
        frames : 𝒫(FrameId)
        regions : RegionId ↦ Region
    }
Χ ∈ Heap

```
## Typing 
Implicit program P
```rs

typeof(χ, PTrue) = P.primitives(Bool)
typeof(χ, PFalse) = P.primitives(Bool)
typeof(χ, ι) = χ.metadata(ι).type
typeof(χ, 𝕗) =  
typeof(χ, PNone) = P.primitives(TNone)
//typetest (T₀,T₁) Checks whether T₀ is of type T₁
typetest(T₀,None) = False //Need these to deal with function types that could be empty
typetest(T₀, Some T₁) = typetest(T₀,T₁)
typetest(τ₀,T₁) = T₁ ∈ P.types(τ₀).supertypes


typetest(χ,v,τ) = (typeof(χ,v),τ)
```

## Call


```rs
Params := Move Ident | Copy Ident
pr ∈ Params

typecheck(χ, φ, F, y*) =
  |F.params| = |y*| ∧
  ∀i ∈ 1 .. |y*| . typetest(χ, φ(yᵢ), F.paramsᵢ.type) 




newframe_init(χ, φ, F, x, stmt*,calltype,ρ) =
    {id: 𝔽, vars:{},
    ret: x, type: F.result, cont:stmt*, calltype: calltype, region : ρ}
     where
    (𝔽 ∉ dom(χ.frames)) ∧ (𝔽 > φ.id) // Fresh frame id gen



newframe_populate(χ,φₒ,φₙ,Move y;pr*,F,i) = 
    φₒ,φₙ = move_fun_arg(φₒ,φₙ,y,F.paramsᵢ.name)
    newframe_populate(χ,φₒ,φₙ,pr*,F,i + 1)

newframe_populate(χ,φₒ,φₙ,Copy y;pr*,F,i) = 
    (χ₁,φₙ) = copy_fun_arg(χ,φₒ,φₙ,y,F.paramsᵢ.name)
    newframe_populate(χ₁,φₒ,φₙ,pr*,F,i + 1)

newframe_populate(χ,φₒ,φₙ,[],F,i) = 
    χ,φₒ,φₙ
    
newframe(χ, φₒ, F, x, pr*, stmt*,calltype) = 
    φₙ = newframe_init(χ, φ, F, x, stmt*,calltype)
    newframe_populate(χ,φₒ,φₙ,pr*,F,1)


move_fun_arg (φₒ,φₙ,y,z) = 
    (φₒ\{y},φₙ[z ↦ φₒ(y)])

copy_fun_arg (χ,φₒ,φₙ,y,z) = 
    χ₁ = inc(χ,φ₀(y))
    χ₂ = stack_inc(χ₁,loc(χ,φ₀(y)),1)
    φₙ[z ↦ φₒ(y)] 



//Get the identifiers out of input params to pass to new frame
get_idents(Move y; pr*) = y;get_names(pr*)
get_idents(Copy y; pr*) = y;get_names(pr*)
get_idents([]) = []



// There are three ways to call: 
// bind x (call f pr*) will return anything raised by f, throw anything thrown by f, and bind x to the return value of f if f returns
// bind x (subcall f pr*) will raise anything raised by f, throw anything thrown by f, and bind x to the return value of f if f returns
// bind x (catch f pr*) will treat throws and raises by f as returns, and so will bind the value thrown, raised, or returned to x.


CallTerm = call | subcall | catch
ct ∈ CallTerm

call_term_to_call_type(call) = Call
call_term_to_call_type(subcall) = Subcall
call_term_to_call_type(catch) = Catch

x ∉ φ₀
ρ ∉ χ
F = P.functions(φ₀(f))
y* = get_idents(pr*)
typecheck (Χ,φ₀,F,y*)

φ₂,φ₁ = newframe(χ, φ₀, F, x, pr*, stmt*, call_term_to_call_type(ct),ρ) 

----------------------------------------------------------------------------------------------------------------------[call/subcall/catch]
χ, σ;φ₀, (bind x (ct f pr*));stmt* ⇝ (χ ∪ (φ₁.id))[ρ ↦ {type : RegionRC, parent = φ₁.id, rc : 0}], σ;φ₂ ;φ₁, F.body



```
## Return

```rs
// Three forms of return: return, raise, and throw. Return is a local return. Raise is a non-local return. It will return at a place where the caller used standard call. It can also be captured into a binder if the caller used catch. Throw should be used in the case of an error, and will propogate upwards unless captured by a catch.


ReturnTerm = return | NonLocal
rt ∈ ReturnTerm
NonLocal = raise | throw
nl ∈ NonLocal


heap_after_return(χ,ι,𝔽,φ) = 
    ρ = χ.metadata_obj(ι) 
    drag(χ,ι,φ.region) if islocal(χ,ρ) ∧ ρ.parent = 𝔽

heap_after_return(χ,_,𝔽,φ) = Some χ
// Capture Rule (regular return and any catch)
// TODO: what if this is the last frame on the stack?

dom(φ₁.vars) = {x}
v = φ₁(x)
typetest(typeof(χ,v),φ₁.type.rt)
(rt = return) ∨ (φ₁.calltype = Catch)
heap_after_return(χ,v,φ₁.id,φ₀) = Some χ₁
φ₂ = φ₀[φ₁.ret ↦ v] 
---------------------------------------------------------------[return/catch] 
χ, σ;φ₀;φ₁,rt x;stmt* ⇝ χ₁\(φ₁.id)\(φ₁.region), σ;φ₂, φ₁.cont


// All the other rules unwrap (based on the combination of the call type and return type)
unwrap(call,throw) = throw
unwrap(call,raise) = return
unwrap(subcall,throw) = throw
unwrap(subcall,raise) = raise 


dom(φ₁.vars) = {x}
v = φ₁(x)
typetest(typeof(χ,v),φ₁.type.nl)
rt = unwrap(φ₁.calltype,nl)
heap_after_return(χ,v,φ₁.id,φ₀) = Some χ₁
φ₂ = φ₀[φ₁.ret ↦ v]
---------------------------------------------------------------[non-local]
χ, σ;φ₀;φ₁, nl x; stmt* ⇝ χ₁\(φ₁.id)\(φ₁.region), σ;φ₂, rt φ₁.ret



// Drop other frame variables
dom(φ.vars) = {x,y} ∪ zs
----------------------------------------------------------[frame-exit-drop]
χ, σ;φ, rt x;stmt* ⇝ χ, σ;φ, drop y;rt x
```

## Lookup-FunctionPtr 
```rs
x ∉ φ
τ = typeof(χ, φ(y))
𝐟 = (P.types(τ).methods(w))
----------------------------------------------------------------[lookup-dynamic]
Χ,σ;φ bind x (lookup w y);stmt* ⇝ Χ,σ,φ[x ↦ 𝐟],stmt*

```
## Drop 

```rs
φ(x) = v
χ₁ = region_stack_dec(χ₀, v)
χ₂ = dec(χ₁, v)
--- [drop]
χ₀, σ;φ, drop x;stmt* ⇝ χ₂, σ;φ\x, stmt*
```

## Write Barrier
```rs 
// HELPERS //

// gives new heap with ι now in ρ 
move_object(χ,ι,ρ) = χ.metadata_obj(ι)[location ↦ ρ]

move_objects(χ,ι;ιs,ρ) = move_objects(move_object(χ,ι,ρ), ιs, ρ)
move_objects(χ,∅,ρ) = χ

stack_inc(χ,ρ,n) = 
    stack_inc(χ,χ(ρ).parent,1)(ρ)[stack_rc += n] if χ(ρ).stack_rc == 0 // If we are increasing from 0 (this region now has references to it), add 1 to parent
    χ(ρ)[stack_rc += n] otherwise
stack_inc(χ,_,n) = χ // Anything other than a region, do nothing

stack_dec_1(χ,ρ) = 
    stack_dec_1(χ,χ(ρ).parent)(ρ)[stack_rc --] if χ(ρ).stack_rc == 1 // If this region no longer has any references to it, decrease the parent ref count by 1
    χ(ρ)[stack_rc --] otherwise

stach_dec_1(χ,_) = χ //Anything other than a region, do nothing

stack_dec(χ,ρ,n) = 
    χ if n == 0
    stack_dec(stack_dec_1(χ,ρ),ρ,n-1)

// - Be consistent about sets or lists, which is better? how to do lists if lists better? if sets, need some understanding of picking an element/order doesn't matter since do things per region/per object, all sets finite

islocal(χ,L) = 
χ(ρ).type = RegionRC ∧ χ(ρ).parent ∈ FrameId if ρ = L 
false otherwise



//n = internal ref count
frame_locals_nl(χ,ι,ρ,ιsₜ,n) = 
    frame_locals_nl(χ,ι,xs,ρ,ιs ∪ {ιₓ}, n + 1) if ιₓ = χ(ι)(x) ∧ ρₓ = loc(χ,ιₓ) ∧ islocal(χ,ρ₁)

frame_locals_nl(χ,ι,⋅,ρ,ιs,n) = (ιs,n)


// all the fields of the object that are in real regions, cannot be a set, because if an object appears twice, we need to process it twice
region_fields(χ,ω,ρ) = 
    {x | x ∈ dom(ω) ∧ ω(x) = ι ∧ ρₓ = loc(χ,ι) ∧ ~islocal(χ,ρₓ) ∧ ρₓ ≠ ρ}



// precondition: all χ(ι)(x) are in real regions, and not in ρ (so ρ ≠ ρ₁) // need to go object by object, to ensure that no region apeears twice, so maintain list of field ids rather than object identifiers 
parent_regions_ok(χ,ι,x;xs,ρs,ρ) = 
    ω = χ(ι)
    ρ₁ = loc(χ,ω(x)) ∉ ρs ∧ (χ(ρ₁).parent = None) ∧ ~is_ancestor_of(χ,ρ₁,ρ) ∧ parent_regions_ok(χ,ι,xs,ρs ∪ {ρ₁}, ρ)

parent_regions_ok(χ,ι,⋅,ρs,ρ) = true

//regions that the objects ιs live in precondition: all χ(ι)(x) are in real regions
get_regions(χ,ι,x;xs) = 
    {loc(χ,χ(ι)(x))} ∪ get_regions(χ,ι,xs)

get_regions(χ,ι,⋅) = ∅ 

    

add_regions(χ,ι,xs,ρs,ρ) = 
    Some (get_regions(χ,ι,xs) ∪ ρs) if parent_regions_ok(χ,ι,xs,ρs,ρ)
    None otherwise






is_ancestor_of(χ,ρ₀,ρ) =
    true if ρ₀ = ρ
    false if χ[ρ].parent ∉ RegionId 
    is_ancestor_of(χ,ρ₀,χ[ρ].parent) otherwise


//ιsₕ = objects that still need to be handeled 
//ιsₜ = objects that are already tracked 
//ρ = the region we want to drag objects into
//χ = the heap
//ρs = the real regions that we visit during the traversal
get_all_draggables(χ,ι;ιsₕ,ρ,n,ιsₜ,ρs) = 
    ω = χ(ι) // the source object we want to drag
    ρ₀ = loc(χ,ι) // the location of the object
    (ιsₜ₀,n₀) = frame_locals_nl(χ,ι,dom(ω),ρ,n,ιsₜ)// total set of objects now tracked, new internal ref count

    ιsₙ = ιsₜ₀ \ ιsₜ //new objects found in above pass

    xsᵣ = region_fields(χ,ι,ρ) //fields of ι which are in real regions that are not ρ this needs to be a list, because if a region appears more than once, we need to fail

    get_all_draggables(χ,ιsₕ ∪ ιsₙ,n₀, ρ, ιsₜ₀,ρs₁)  if add_regions(χ,ι,xsᵣ,ρs,ρ) = Some ρs₁
    None otherwise
   
get_all_draggables(χ,∅,ρ,n,ιsₜ,ρs) = Some (ιsₜ,ρs,n)





// precondition: ρ₁ and ρ are not frame local
parent_region(χ,ρ₁,ρ) = 
    χ(ρ₁)[parent ↦ ρ] if χ(ρ₁).stack_rc = 0 
    stack_inc(χ(ρ₁)[parent ↦ ρ],ρ,1) otherwise

// precondition: ρ₁;ρs, ρ are not frame local
parent_regions(χ,ρ₁;ρs,ρ) = 
    χ₁ = stack_dec(parent_region(χ,ρ₁,ρ),ρ₁,1)
    
parent_regions(χ,∅, ρ) = χ

update_ref_counts(χ,ρ,n,ι;ιs) = 
 update_ref_counts(stack_inc(χ,ρ,χ.metadata_obj(ι).rc),ρ,n,ιs)
    
update_ref_counts(χ,ρ,n,∅) = stack_dec(χ,ρ,n)
    
//precondition: ι is in frame local region, ρ is a non local region
drag_non_local(χ,ι,ρ) = 
    χ₁ = move_objects(χ,ιs,ρ)
    χ₂ = parent_regions(χ₁,ρs,ρ)   
    Some (update_ref_counts(χ₂,ρ,n,ιs))
        if get_all_draggables(χ,{ι},ρ,0,{ι},∅) = Some (ιs,ρs,n) // things to drag, regions to parent, internal ref count of things getting dragged
    None otherwise


// precondition: ρ is frame local
frame_local_l(χ,ι,ρ) = 
    {ι₁ | x ∈ dom(χ(ι)) ∧ ρ₁ = loc(χ,ι(x)) ∧ islocal(χ,ρ₁) ∧ ρ₁.parent > ρ.parent}


// precondition: ρ is frame local
drag_local(χ,ι;ιs,ρ) = 

    drag_local(χ,ιs,ρ) if loc(χ,ι) = ρ // this object was already moved (seen before)
    drag_local(move_object(χ,ι,ρ), frame_local_l(χ,ι,dom(χ(ι)),ρ) ∪ ιs,ρ) otherwise // if we haven't seen this object yet, move it and get its fields


drag_local(χ,∅,ρ) = χ


drag(χ,ι,ρ) = 
    drag_local(χ,{ι},ρ) if islocal(χ,ρ)
    drag_non_local(χ,ι,ρ) otherwise

  
  

clear_parent(χ,L) = 
    stack_dec(χ,χ(ρ).parent,1)(ρ).parent = None if L = ρ ∧ χ(ρ).parent ∈ RegionId 
    χ otherwise


safe_from_frame(χ,𝔽,ι₁) = 
    true if ρ₁ = loc(χ,ι₁)  ∧ ~islocal(χ,ρ₁) 
    true if ρ₁ = loc(χ,ι₁) ∧ islocal(χ,ρ₁) ∧ 𝔽 ≥ χ(ρ₁).parent
    true if 𝔽₁ = loc(χ,ι₁) ∧ 𝔽 ≥ 𝔽₁ 
    true if Immutable = loc(χ,ι₁)
    false otherwise

write_barrier_frame(χ,𝔽,ι₁) = 
    Some χ if safe_from_frame(χ,𝔽,ι₁)
    None otherwise

// precondition: ρ is local, ι₁ is not on the stack (true means safe with no changes, false means must do drag)
safe_from_local(χ,ρ,ι₁) = 
    true if ~islocal(χ,loc(χ,ι₁))  
    true if ρ₁ = loc(χ,ι₁) ∧ islocal(χ,ρ₁) ∧ χ(ρ₁).parent ≤ χ(ρ).parent // covers if ρ₁ = ρ₂
    false otherwise

// precondition: ρ is frame local
write_barrier_local(χ,ρ,ι₁) = 
    None if 𝔽 = loc(χ,ι₁) 
    Some χ if ρ₁| Immutable = loc(χ,ι₁) ∧ safe_from_local(χ,ρ,ι₁)
    Some drag(χ,ι,ρ) otherwise



// precondition: ρ is a real region 
write_barrier_real_region(χ,Lₚ,ρ,ι₁) = 
    None if 𝔽 = loc(χ,ι₁)
  
    Some χ if Immutable = loc(χ,ι₁) ∧ islocal(Lₚ)

    Some stack_inc(χ,Lₚ,1) if Immutable = loc(χ,ι₁) ∧ ~islocal(Lₚ)

    Some χ if ρ₁ = loc(χ,ι₁) ∧ ρ₁ = Lₚ 
    
    Some stack_dec(clear_parent(stack_inc(χ,Lₚ,1),Lₚ),ρ₁,1) if ρ₁ = loc(χ,ι₁) ∧ ρ = ρ₁ ∧ ρ ≠ Lₚ

    Some stack_dec(parent_region(stack_inc(χ,Lₚ,1),ρ₁,ρ),ρ₁,1) if ρ₁ = loc(χ,ι₁)  ∧ ~islocal(χ,ρ₁) ∧ ρ ≠ ρ₁ ∧ ρ = Lₚ ∧ χ.ρ₁.parent = None ∧ ~is_ancestor(ρ₁,ρ)

    Some stack_dec(parent_region(clear_parent(stack_inc(χ,Lₚ,1),Lₚ),ρ₁,ρ),ρ₁,1) if ρ₁ = loc(χ,ι₁) ∧ ~islocal(χ,ρ₁) ∧ ρ₁ ≠ ρ ≠ Lₚ ∧ χ.ρ₁.parent = None ∧ ~is_ancestor(ρ₁,ρ)

    Some stack_inc(χ₁,Lₚ,1) if ρ₁ = loc(χ,ι₁) ∧ islocal(χ,ρ₁) ∧ Some χ₁ = drag(χ,ι₁,ρ) 



write_barrier(χ,Lₚ,L,ι₁) = 
    None if Immutable = L 
    write_from_frame(χ,𝔽,ι₁) if 𝔽 = L 
    write_barrier_local(χ,ρ,ι₁) if ρ = L ∧ islocal(χ,ρ)
    write_barrier_real_region(χ,Lₚ,L,ι₁) if ρ = L ∧ ~islocal(χ,ρ)
    
   
 write_barrier_fields(χ,L,ι,x;xs) = 
    
    write_barrier_fields(χ₁,L, ι,xs) if ι₁ = χ(ι)(x) ∧ write_barrier(χ,None,L,ι₁) = Some χ₁
    None if ι₁ = χ(ι)(x)  ∧ write_barrier(χ,None,L,ι₁) = None
    write_barrier_fields(χ,L,ι,xs) if χ(ι)(x) ∉ ObjectId

write_barrier_fields(χ,_,ι,∅) = χ

write_barrier_ref(χ,𝕣,ι) = 
    Lₚ = loc(χ,𝕣.field) // previous location 
    L = loc(χ,𝕣.target) // location of object this is a reference into 
    write_barrier(χ,Lₚ,L,ι)

write_barrier_ref_cown(χ,)


write_cown(χ,π,)
```

## New Objects
```rs
// a new frame with things invalidated if moves, and remaining if copies,along with the new object ω
newobject(χ₁,φ,(y₁,move z);(y,pr)*) =  
        ω₁ = {y₁ ↦ φ(z)}
        (χ₁,φ₂,ω₂) = newobject(χ,φ₁\z,(y,pr)*) 
        (χ₁,φ₂,ω₁ ∪ ω₂)

newobject(χ,φ,⋅) = (χ,φ,∅)

newobject(χ,φ,(y₁,copy z);(y,pr)*) =  
        χ₁ = inc(χ,φ(z))
        χ₂ = stack_inc(χ,loc(χ,φ(z)),1)
        ω₁ = {y₁ ↦ φ(z)}
        (χ₃,φ₂,ω₂) = newobject(χ₂,φ₁,(y,pr)*) 
        (χ₃,φ₂,ω₁ ∪ ω₂)


ι ∉ χ
ρ ∉ χ
φ₁,ω = newobject(φ, (y, pr)*) 
R = {type : RegionRC, stack_rc : 1, parent : None} // Region init
χ₁ = χ[ρ ↦ R] // bind region
typecheck(χ, τ, ω) // make sure object actually has the right type
χ₂ = χ₁[ι ↦ ω] 
χ₃ = χ₂.metadata_obj[ι] = {type:τ, location:ρ,rc : 1} 
χ₄ = write_barrier_fields(χ₃,ρ,ι) //can't just do a for all on the args because heap updated for every field
----------------------------------------------------------------- [regionrc] // Creates an object in a new region 
χ,σ;φ,bind x (region rc τ (y,pr)*) stmt* ⇝ χ₄,σ;φ₁, stmt*


ι ∉ χ
ι₁ = φ(w)
ρ = χ.metadata_obj(ι₁).location 
(φ₁,ω) = newobject(φ, (y, pr)*) 
typecheck(χ, τ, ω)
χ₁ = χ[ι ↦ ω] 
χ₂ = χ.metadata_obj[ι] = {type:τ, location:ρ,rc : 1} 
χ₃ = write_barrier_fields(χ₂,ρ,ι) //can't just do a for all on the args because heap updated for every field
-----------------------------------------------------------------
χ,σ;φ,bind x (heap w τ (y,pr)*) stmt* ⇝ χ₃,σ;φ₁, stmt*



ι ∉ χ
φ₁,ω = newobject(φ, (y, pr)*) 
typecheck(χ, τ, ω)
χ₁ = χ[ι ↦ ω] 
χ₂ = χ.metadata_obj[ι] = {type:τ, location:ρ,rc : 1} 
χ₃ = write_barrier_fields(χ₂,ρ,ι) //can't just do a for all on the args because heap updated for every field
-----------------------------------------------------------------
χ,σ;φ,bind x (new τ (y,pr)*) stmt* ⇝ χ₃,σ;φ₁, stmt*


```
## References 
```rs
x ∉ φ
𝕣 = φ(y)


------------------------------------------------------------------ store 
χ,σ;φ, (bind x (store y z));stmt* ⇝ 
```
## Ref Counting Helpers 
```rs

loc(χ, p) = Immutable
loc(χ, ι) =
  χ.metadata(ι).location if ι ∈ χ
  Immutable otherwise

inc(χ, p) = χ
inc(χ, ι) =
  χ[metadata_obj(ι)[rc += 1]] // even if not an rc region, ok to have this because we don't rely on ref count? 

dec(χ, p) = χ
dec(χ, ι) =
  χ[metadata(ι)[rc -= 1]] 



## Examples for non-local return: 
(These are currently written not in bytecode, will rewrite in a bytecode style)

```rs
//Return early from iterator
def fold_left f acc l = 
    match l with
    |[] -> return (acc)
    |h ::t -> x = call f acc h
              y = call fold_left f x t
              return y
    

def div acc x = 
  if x == 0 then throw (None) 
  else 
    match acc with
    |None -> throw (None)
    |Some n -> return (Some (n/x))

sequence_of_divs_ok =  catch fold_left div (Some 600) [10;5;4;3] //This should go through the whole thing
sequence_of_divs_early_return = catch fold_left div (Some 600) [10;0;4;3]//This should raise 0 after hitting the second element, forcing fold_left to exit early (returning none)

// Behavior of call vs subcall vs catch
def daz y = 
    raise 0

// foo1 calls bar1 (if bar1 raises, foo1 returns)
// bar1 calls daz as a subcall (if daz raises, bar1 raises)
// daz1 raises 0
// bar1 raises 0
// foo1 returns 0
def foo1 x = 
    def bar1 y = 
        w = subcall (call daz y)
        return 1
    z = call bar1 x
    return 2



// foo2 calls bar2 (if bar2 raises, foo2 returns)
// bar2 calls daz (if daz raises, bar2 returns)
// daz raises 0
// bar2 returns 0
// z is set to 0
// foo2 returns 2
def foo2 x = 
    def bar2 y = 
        w = call daz y 
        return 1

    z = call bar2 x 
    return 2
 
 
// foo3 calls bar3 (if bar3 raises, foo3 returns)
// bar3 calls daz but in a catch (if daz raises, w will be bound to the value that daz raises)
// daz raises 0
// w is bound to 0 
// bar3 returns 1
// z is bound to 1 
// foo3 returns 2 
def foo3 x = 
    def bar3 y = 
        w = catch daz y 
        return 1

    z = call bar3 x 
    return 2




// Use of subcall example [find position in list that x is in, raises if not found]
def in_list x cur_pos l acc = 
    match l with 
    | [] -> raise Not_Found 
    | hd :: tl -> if hd == x then 
                    result = cur_pos::acc
                    return result
                  else
                    result = subcall in_list x (cur_pos + 1) tl acc
                    return result


def fold_right f l acc = 
    match l with 
    | [] -> return acc 
    | hd :: tl -> 
            new_acc = subcall fold_right f tl acc 
            result = subcall f hd new_acc 
            return result

l = [[1;2;5];[3;2;5];[5;7;4]]                
loc_of_5 = catch fold_right (in_list 5 0) l [] // should return from every frame
loc_of_2 = catch fold_right (in_list 2 0) l [] // 
  
// find in 2D list 

def find_in_2d x ls = 
    def find_in_l list_index index x l = 
        match l with 
        | [] -> return None
        | hd::tl -> if hd == x: 
                        raise Some (list_index,index)
                    else: 
                        result = subcall find_in_l list_index (index + 1) x tl
                        return result
    
    def helper list_index x ls = 
        match ls with 
        | [] -> raise None
        | l::tl -> in_l = subcall find_in_l list_index 0 x l
                   in_tl = subcall helper (list_index + 1) x tl 
                   

    result = call helper 0 x ls 


    // binary search with insert if not found 
    Node A = Leaf | Node A * A * Node A 

    def find x tree = 
        match tree with 
        | Leaf -> 
        | 









// early return if something found, try another path if something not found? 

def find x l name= 
    match l with 
    | [] -> throw "not found"
    | hd :: tl -> 
         if hd == x: 
            return ("found in " + name)
         else subcall (find x l name)  

def find_in_columns x sheet = 

    

    


```