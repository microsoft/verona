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
- lookups in typetraits
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

refvar
  var x -> let x = <cell>
  refvar-lhs -> reflet?
  refvar -> load?

type assertions on:
  reflet, refvar, refparam, literal

## Lookup

not nested: return the target
nested
  let/var/param/function: fail
  class::X
    lookup X in the class
    if X is
      class: done
      typeparam: done
      function: done
      typealias: done
  typeparam::X
    lookup X in our bounds
    the typeparam and our bounds may already have been subbed
    `A[B[C]::E, B[D]::E]::F`
      find A
      find B, no subs
      find C, no subs
      find D, no subs
      bind B[C]::E to A::_0
      bind B[D]::E to A::_1
        problem: two different bindings for the same typeparam B::_0
      find F
      drop binding

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
