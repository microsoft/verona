# Todo

minimum 1 thing in Expr?

builtins
  if...else
  typetest
  match

list inside TypeParams or TypeArgs along with groups or other lists

lookup
- isect: lookup in lhs and rhs?
- lookups in typetraits
- error on too many typeargs

`new` to create an instance of the enclosing class
public/private
object literals
package schemes
type assertions are accidentally allowed as types

CallLHS
- separate implementation
- `fun f()` vs `fun ref f()`
- if a `ref` function has no non-ref implementation, autogenerate one that calls the `ref` function and does `load` on the result

## Symbol Tables

if symbol table lookup has multiple definitions
  and any of them isn't a flag::multidef node
  then error

catch unexpected bindings as well
  or generate bindings from wf conditions

## Ellipsis

`expr...` flattens the tuple produced by `expr`
- only needed when putting `expr...` in another tuple
`T...` is a tuple of unknown arity (0 or more) where every element is a `T`
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

functions on types (functors, higher-kinded types)
  kind is functor arity

write functions of type->type explicitly?
  treat any type alias as a function of type->type?

use DontCare to create type lambdas?
  `T: Array[_]`, `T[U]` -> `Array[U]`

```ts
type ?[T] = Some[T] | None
type ->[T, U] = Fun[T, U]

type Fun[T, U] =
{
  (self, x: T): U
}

x: ?[T] // awkward
x: ?T // better - implies adjacency is application
x: T.? // reverse application? prefer . as viewpoint
x: Array U32 // awkward or good?

type |[T, U] = Union[T, U] // no, unions are fundamental

```
