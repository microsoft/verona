# Todo

builtins
  if...else
  typetest

if...else and short-circuit evaluation
  we have to build the lambdas before we decide which lambdas are executed
  the same iso free variable can't be in more than one lambda

list inside TypeParams or TypeArgs along with groups or other lists
= in an initializer
lookup
- isect: lookup in lhs and rhs?
- lookups in typetraits
- error on too many typeargs

public/private
`new` to create an instance of the enclosing class
object literals

package schemes
dependent types

CallLHS
- separate implementation
- `fun f()` vs `fun ref f()`
- if a `ref` function has no non-ref implementation, autogenerate one that calls the `ref` function and does `load` on the result

## QuickCheck

if symbol table lookup has multiple definitions
  and any of them isn't a flag::multidef node
  then error

type assertions are accidentally allowed as types
type assertions on operators are awkward
catch unexpected bindings as well
  or generate bindings from wf conditions

functions on types (functors, higher-kinded types)
  kind is functor arity

## Ellipsis

`expr...` "unpacks" the tuple produced by `expr`
- needed when putting `expr...` in another tuple
- or passing a tuple as a sequence of arguments to a function
`id...` on the lhs of an assignment accepts "remaining" (possibly 0 length) tuple elements
`T...` is a tuple of unknown arity (0 or more) where every element is a `T`
- bounding a type list bounds the elements, not the list itself
- `T...` in a tuple "unpacks" the type into the tuple
- `T...` in a function type "unpacks" the type into the function arguments

```ts
// multiply a tuple of things by a thing
mul[n: type {*(n, n): n}, a: n...](x: n, y: a): a
{
  match y
  {
    { () => () }
    { y, ys... => x * y, mul(x, ys...)... }
  }
}
```

## ANF

rhs ref -> dup ?

Γ := id -> typevar

block: expr* term

expr:
  $0 = literal ...
    (let $0 (int 4))
    Γ($0) :> literal
  $0 = tuple $1..$N
    Γ($0) :> Γ($1..$N)
  $0 = new $1..$N // stack or region
  $0 = lambda ...
  $0 = call (selector f) $1..$N
    Γ($1) <: { f($T1..$TN): $T0 } & $T1
    Γ($0) :> $T0
  $0 = call reffunction $1..$N
  $0 = calllhs selector $1..$N
  $0 = calllhs reffunction $1..$N
  $0 = typetest T $1
    Γ($0) :> Bool

  (let $0 (if $1 (block ...) (block ...)))

  $0 = dup $1 ?
  drop $0 ?

term:
  br <label>
  condbr $0 <label> <label>
  ret $0
  throw $0

## If...Else

pre (let x (if $1 block1 block2)) post
->
  pre:
    condbr $1 block1 block2

  block1:
    ...
    br post

  block2:
    ...
    br post

  post:
    $x = phi [block1 $0, block2 $0]
    ...

## Try...Catch

if there's no catch then insert one that drops the exception
if there's no try around a call then insert one that rethrows the exception

pre (let x (try block1 catch block2)) post
->
  pre:
    br block1.1

  block1.1:
    $0 = call f1 ...
    // intrinsic for carry flag checking?
    %ok = intrinsic.carry_flag()
    condbr $ok block1.2 cleanup1.1

  block1.2:
    $0 = call f2 ...
    %ok = intrinsic.carry_flag()
    condbr $ok post cleanup1.2

  cleanup1.1:
    // cleanup from failed block1.1
    ...
    br block2

  cleanup1.2:
    ...
    br block2

  block2:
    $0 = ...
    br post

  post:
    $x = phi [block1.2 $0, cleanup1.2 $0]
    ...

## Lookup

lookup in union and intersection types

## type checker

can probably do global inference

finding selectors
  don't bother: treat as a `{ selector }` constraint?
  what about arity?
    arg count gives minimum arity
    a selector with higher arity is a subtype of a selector with lower arity
    given `f(A, B, C): D`, type it as `A->B->C->D` not `(A, B, C)->D`

selecting by reffunc arity
  select the shortest, not the longest
  use tuple notation to force a longer arity

## DNF

does `throw` need to have a separate DNF level?
- in between union and isect?

## param: values as parameters for pattern matching

named parameters
  (group ident type)
  (equals (group ident type) group*)
pattern match on type
  (type)
pattern match on value
  (expr)

## Lambda typing

how to get a "call once" lambda
  apply(self: iso)
    (var ...) = destruct self
  this extracts the fields
  what if the lambda can't be iso?

## Unique vs Linear?

linear is use once
what does the existence of a linear closure deny?
  denies any local or global aliases
  but linear can still reach the "enclosing" region
  which means it's truly unique, not externally unique

can we separate region and unique?
  unique[T] has only one reference, but T isn't a different region
  unique[region[T]] has one reference, and there's a new region
  region[T] on it's own - blah, what's that?

for an object O in region R1
and a reference in region R2
  mut =>
    R1 = R2
    no mut references to O outside of R1
    no imm references to O in any region
  imm =>
    no mut or iso references to O in any region
  iso =>
    R1 != R2
    no other iso reference to O in any region
    no imm references to O in any region
    no mut references to O outside of R1
  lin => `unique` or `linear`
    R1 = R2
    no other references to O in any region

{x => ... } : ((self: Self & lin, T)->U) & lin

what about "one of these"?
  ie will only call one of a set of linear lambdas

## Type Language

adjacency for intersection?
write functions of type->type explicitly?
  treat any type alias as a function of type->type?

```ts
f(x: T iso): imm T
{
  freeze(consume x)
}

type ?[T] = Some[T] | None

x: ?[T] // awkward
x: ?T // better - implies adjacency is application, not intersection
x: Array U32 // awkward or good?

type |[T, U] = Union[T, U] // no, unions are fundamental

```

## Flow Typing Contexts

```ts

True::if[T, U](self: True ^ ⌊U⌋, f: (() ^ ⌊U⌋) -> T): Taken[T] =
  // Taken(f(() with self))
  Taken(f())
False::if[T, U](self: False ^ ⌊U⌋, f: (() ^ ⌊U⌋) -> T): NotTaken[T] ^ ⌊U⌋ =
  // NotTaken[T] with self
  NotTaken

Taken[T]::elseif[DC](self, cond: () -> Bool, f: DC -> T): Taken[T] = self
NotTaken[T]::elseif[U](
  self: NotTaken[T] ^ ⌊U⌋, cond: () -> Bool, f: (() ^ ⌊U⌋) -> T):
  Taken[T] | (NotTaken[T] ^ ⌊U⌋) =
  // cond().if(f with self)
  if(cond()) f

Taken[T]::else[DC](self, f: DC -> T): T = self()
NotTaken[T]::else[U](self: NotTaken[T] ^ ⌊U⌋, f: (() ^ ⌊U⌋) -> T): T =
  // f(() with self)
  f()

```

## Kappa

```ts

iso "entry point to another region"
  denies {iso, mut, imm, paused}
mut "from my region"
  denies {iso, imm, paused}
imm "from no region"
  denies {iso, mut, paused}
paused "from another region"
  denies {iso, mut, imm}

iso▹T = ⊥
mut▹T = T
imm▹T = imm
paused▹iso = iso
paused▹mut = paused
paused▹imm = imm

ref[T]::load : (ref[T] & K) -> K▹T
ref[T]::store : (ref[T] & mut) -> T -> T

Γ ⊢ x 
T = T1 in Γ[borrow/mut] ⊢ e : T1 iff ¬open(region(x))
    throw AlreadyOpen otherwise
--- [ENTER]
Γ ⊢ enter x e : T

```
