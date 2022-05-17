# Todo

builtins
  if...else
  typetest

if...else and short-circuit evaluation
  we have to build the lambdas before we decide which lambdas are executed
  the same iso free variable can't be in more than one lambda

list inside Typeparams or Typeargs along with groups or other lists
= in an initializer
lookup
- isect: lookup in lhs and rhs?
well-formedness for errors
- error on too many typeargs

public/private
`new` to create an instance of the enclosing class
object literals

package schemes
dependent types

applying typeargs to a typeparam
- T[C1]
- this is filling in the bounds in some way?

## ANF

destructure tuples in assignment lhs?

((a, b), c) = e

(assign
  (tuple-lhs (tuple-lhs (refvar-lhs a) (refvar-lhs b)) (refvar-lhs c))
  e
  )

(let $a (refvar-lhs a))
(let $v_a (load $a))
(let $b (refvar-lhs b))
(let $v_b (load $b))
(let $t1 (tuple $v_a $v_b))
(let $c (refvar-lhs c))
(let $v_c (load $c))
(let $t2 (tuple $t1 $v_c))
(let $e e)
(let $e_1 (call _1 $e))
(let $e_1_1 (call _1 $e_1))
(let $e_1_2 (call _2 $e_1))
(let $e_2 (call _2 $e))
(store $a $e_1_1)
(store $b $e_1_2)
(store $c $e_2)
(reflet $t2)

type assertions on:
  reflet, refvar, refparam
`let x = ...`
  currently trying to store to `x` instead of bind to it

```ts

(assign e0 e1 e2)

(let $0 = lhs e0)
(let $1 = load $0)
(let $2 = lhs e1)
(let $3 = load $2)
(store $0 $3)
(assign $2 e2)
  (let $0 = lhs $2)
  (let $1 = load $0) // $3
  (let $2 = e2) // not lhs+load!
  (store $0 $2)
(reflet $1)

tuple-lhs?
  then rewrite (store tuple-lhs e0)

e0 = e1 = e2

$0 = e2
$1 = lhs e1
$2 = load $1
store $1 $0
$3 = lhs e0
$4 = load $3
store $3 $2

```

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
