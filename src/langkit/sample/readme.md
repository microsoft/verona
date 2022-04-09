# Todo

list inside Typeparams or Typeargs along with groups or other lists
= in an initializer
lookup
- isect: lookup in lhs and rhs?
well-formedness for errors
- error on too many typeargs

type checker

public/private
`new` to create an instance of the enclosing class
object literals

package schemes
dependent types

type assertions for operators
- is it ok to just append the type to the operator node?

applying typeargs to a typeparam
- T[C1]
- this is filling in the bounds in some way?

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
