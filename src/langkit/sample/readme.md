# Todo

https://devblogs.microsoft.com/cppblog/cpp23-deducing-this/

print errors!

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

# DontCare for Partial Application

use `_` as a call argument to build a lambda that needs that value filled in
  `f(x, _)` -> `{ $0 => f(x, $0) }`, same as `f(x)` with partial application
  `f(_, x)` -> `{ $0 => f($0, x) }`
  `f _` -> `{ $0 => f($0) }`

## Ellipsis

`expr...` flattens the tuple produced by `expr`
- only needed when putting `expr...` in another tuple
`T...` is a tuple of unknown arity (0 or more) where every element is a `T`
- bounding a type list bounds the elements, not the list itself
- `T...` in a tuple flattens the type into the tuple
- `T...` in a function type flattens the type into the function arguments

```ts
// multiply a tuple of things by a thing
mul[n: type {*(n, n): n}, a: n...](x: n, y: a): a
{
  match y
  {
    { () => () }
    { y, ys => x * y, mul(x, ys)... }
  }
}

let xs = mul(2, (1, 2, 3)) // xs = (2, 4, 6)
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

## param: values as parameters for pattern matching

named parameters
  (group ident type)
  (equals (group ident type) group*)
pattern match on type
  (type)
pattern match on value
  (expr)

## Type Language

adjacency for intersection?
write functions of type->type explicitly?
  treat any type alias as a function of type->type?

use DontCare to create type lambdas?
  `T: Array[_]`, `T[U]` -> `Array[U]`

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
