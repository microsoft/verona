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
n ∈ ℕ
w, x, y, z ∈ Ident
// An element of the powerset over identifiers, so a set of identifiers
ws, xs, ys, zs ∈ 𝒫(Ident)
τ ∈ TypeId
𝕗 ∈ FunctionId
ι ∈ ObjectId
ιs ∈ 𝒫(ObjectId)
Stmt := //Give the list of statements here


Type := TNone | Bool | TypeId  // No refs for now, may add
T ∈ Type
// User Defined Types
TypeDesc := 
{
    supertypes : 𝒫(TypeId) 
    fields : Ident ↦ Type 
    methods: Ident ↦ FunctionId
}
// Each function output type is given as a record: the type the function can return, the type the function could raise, and the type the function can throw. All are optional (if a function only returns and never raises, it should only be given a return type, and None for the others etc)
Function := 
    {
        params : {name : Ident, type : T}* //input params x:T
        result : {return : Opt T, raise : Opt T, throw : Opt T} // return, raise, and throw types for the function (None or Some T, where None and Some are on the meta level, not on the type level)
        body : Stmt* //body of function
    }

F ∈ Function

Program := 
    {
        primitives :  Type ↦ TypeId 
        types : TypeId ↦ TypeDesc // Type Defs
        functions: FunctionId ↦ Function // Function Defs
        globals : Ident ↦ Value // Global vars
    }

P ∈ Program



Primitive := PNone | PTrue | PFalse  // Drop None from here?
p ∈ Primitive

Values := ObjectId | Primitive | FunctionID 
v ∈ Value


// This is basically a dictionary (fields in an object {x = A, y = B} etc)
Object := Ident ↦ Value 
ω ∈ Object
// Data associated with an object that isn't the fields, for now just its type, but may be extended, thus defined this way insted of directly as a type id 
ObjectInformation = 
    {type : TypeID}

CallType = Call | Subcall | Try

Frame :=
    {
      id: FrameId, 
      vars: Ident ↦ Value,
      ret: Ident,
      type: Type,
      cont: Statement*,
      calltype: CallType
    }
φ ∈ Frame

Stack := Frame*
σ ∈ Stack

Heap := 
    {
        // Each ObjectId maps to an object 
        data : ObjectId ↦ Object 
        // Each ObjectId is also associated with some information about that object
        metadata : ObjectId ↦ ObjectInformation
        frames : 𝒫(FrameId)
    }
Χ ∈ Heap

```
## Typing 
Implicit program P
```rs

typeof(χ, PTrue) = P.primitives(Bool)
typeof(χ, PFalse) = P.primitives(Bool)
typeof(χ, ι) = χ.metadata(ι).type
typeof(χ, 𝕗) = // ?? what is type of function? do we need to actually have a function type [t*] -> {return : , raise: , throw :}?, or should this error? 
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

newframe_init(χ, φ, F, x, stmt*,calltype) =
    {id: 𝔽, vars:{},
    ret: x, type: F.result, cont:stmt*, calltype: calltype}
     where
    (𝔽 ∉ dom(χ.frames)) ∧ (𝔽 > φ.id) // Fresh frame id gen



newframe_populate(φₒ,φₙ,Move y;pr*,F,i) = 
    φₒ,φₙ = move_fun_arg(φₒ,φₙ,y,F.paramsᵢ.name)
    newframe_populate(φₒ,φₙ,pr*,F,i + 1)

newframe_populate(φₒ,φₙ,Copy y;pr*,F,i) = 
    φₙ = copy_fun_arg(φₒ,φₙ,y,F.paramsᵢ.name)
    newframe_populate(φₒ,φₙ,pr*,F,i + 1)

newframe_populate(φₒ,φₙ,[],F,i) = 
    φₒ,φₙ
    
newframe(χ, φₒ, F, x, pr*, stmt*,calltype) = 
    φₙ = newframe_init(χ, φ, F, x, stmt*,calltype)
    newframe_populate(φₒ,φₙ,pr*,F,1)


move_fun_arg (φₒ,φₙ,y,z) = 
    (φₒ\{y},φₙ[z ↦ φₒ(y)])

copy_fun_arg (φₒ,φₙ,y,z) = 
    φₙ[z ↦ φₒ(y)] // This should increase ref count


//Get the identifiers out of input params to pass to new frame
get_idents(Move y; pr*) = y;get_names(pr*)
get_idents(Copy y; pr*) = y;get_names(pr*)
get_idents([]) = []



// There are three ways to call: 
// bind x (call f pr*) will return anything raised by f, throw anything thrown by f, and bind x to the return value of f if f returns
// bind x (subcall f pr*) will raise anything raised by f, throw anything thrown by f, and bind x to the return value of f if f returns
// bind x (try f pr*) will treat throws and raises by f as returns, and so will bind the value thrown, raised, or returned to x.


CallTerm = call | subcall | try 


call_term_to_call_type(call) = Call
call_term_to_call_type(subcall) = Subcall
call_term_to_call_type(try) = Try
x ∉ φ₀
F = P.functions(φ₀(f))
y* = get_idents(pr*)
typecheck (Χ,φ₀,F,y*)
φ₂,φ₁ = newframe(χ, φ₀, F, x, y*, stmt*, call_term_to_call_type(CallTerm)) 
--------------------------------------------------------------------------[call/subcall/try]
χ, σ;φ₀, (bind x (CallTerm f pr*));stmt* ⇝ χ ∪ (φ₁.id), σ;φ₂ ;φ₁, F.body



```
## Return

```rs
// Three forms of return: return, raise, and throw. Return is a local return. Raise is a non-local return. It will return at a place where the caller used standard call. It can also be captured into a binder if the caller used try. Throw should be used in the case of an error, and will propogate upwards unless captured by a try.


ReturnTerm = return | NonLocal
NonLocal = raise | Throw


// REGULAR RETURN (behaves the same way regardless of how we were called)

dom(φ₁.vars) = {x}
v = φ₁(x)
typetest(typeof(χ,v),φ₁.type.return)
φ₂ = φ₀[φ₁.ret ↦ v] 
----------------------------------------------------------[return/raise/throw] 
χ, σ;φ₀;φ₁,return x;stmt* ⇝ χ\(φ₁.id), σ;φ₂, φ₁.cont



// Called as Try (behaves the same way regardless of how we are returning) 

dom(φ₁.vars) = {x}
v = φ₁(x)
typetest(typof(χ,v),φ₁.type.ReturnTerm) // A bit overloaded here
φ₁.calltype = Try 
φ₂ = φ₀[φ₁.ret ↦ v] 
-------------------------------------------------------[return/raise/raise]
χ, σ;φ₀;φ₁,ReturnTerm x;stmt* ⇝ χ\(φ₁.id), σ;φ₂, φ₁.cont


// Called as Subcall with either raise or throw (return covered by the first rule)
dom(φ₁.vars) = {x}
v = φ₁(x)
typetest(typof(χ,v),φ₁.type.NonLocal) 
φ₁.calltype = Subcall
φ₂ = φ₀[φ₁.ret ↦ v] 
--------------------------------------------------------------[return/raise/throw]
χ, σ;φ₀;φ₁, NonLocal x; stmt* ⇝ χ\(φ₁.id), σ;φ₂, NonLocal φ₁.ret



// Called as regular Call with raise
dom(φ₁.vars) = {x}
v = φ₁(x)
typetest(typof(χ,v),φ₁.type.Raise) 
φ₀.calltype = Call 
φ₂ = φ₀[φ₁.ret ↦ v] 
--------------------------------------------------------------[return/raise/throw] 
χ, σ;φ₀;φ₁, raise x; stmt* ⇝ χ\(φ₁.id), σ;φ₂, return φ₁.ret

// Called as regular Call with throw
dom(φ₁.vars) = {x}
v = φ₁(x)
typetest(typof(χ,v),φ₁.type.Throw) 
φ₀.calltype = Call 
φ₂ = φ₀[φ₁.ret ↦ v] 
--------------------------------------------------------------[return/raise/throw]
χ, σ;φ₀;φ₁, throw x; stmt* ⇝ χ\(φ₁.id), σ;φ₂, throw φ₁.ret



// Drop other frame variables
dom(φ.vars) = {x,y} ∪ zs
----------------------------------------------------------[return/raise/throw]
χ, σ;φ, ReturnTerm x;stmt* ⇝ χ, σ;φ, drop y;ReturnTerm x
```

## Lookup-FunctionPtr 
```rs
x ∉ φ
-----------------------------------------------------------------[lookup-static]
Χ,σ;φ bind x (lookup 𝐟);stmt* ⇝ Χ,σ,φ[x ↦ 𝐟],stmt*

x ∉ φ
τ = typeof(χ, φ(y))
𝐟 = (P.types(τ).methods(w))
----------------------------------------------------------------[lookup-dynamic]
Χ,σ;φ bind x (lookup w y);stmt* ⇝ Χ,σ,φ[x ↦ F],stmt*



```


## Examples for non-local return: 
(These are currently written not in bytecode, will rewrite in a bytecode style)

```rs
//Return early from iterator
def fold_left f acc l = 
    match l with
    |[] -> return (acc)
    |h ::t -> return (fold_left f (f acc h) t)

def div acc x = 
  if x == 0 then raise (None) 
  else 
    match acc with
    |None -> raise (None)
    |Some n -> return (Some (n/x))

sequence_of_divs_ok = fold_left div (Some 600) [10;5;4;3] //This should go through the whole thing
sequence_of_divs_early_return = fold_left div (Some 600) [10;0;4;3]//This should raise 0 after hitting the second element, forcing fold_left to exit early (returning none)

// Behavior of call vs subcall vs try
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
// bar3 calls daz but in a try (if daz raises, w will be bound to the value that daz raises)
// daz raises 0
// w is bound to 0 
// bar3 returns 1
// z is bound to 1 
// foo3 returns 2 
def foo3 x = 
    def bar3 y = 
        w = try daz y 
        return 1

    z = call bar3 x 
    return 2

```