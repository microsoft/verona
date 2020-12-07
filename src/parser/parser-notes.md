# Parser Notes

* multiple definitions of `apply` produces a poor error message
* tests need rewriting for the new syntax rules
* transform prefix and infix to apply
  this needs to know if operators are free functions or not
* try/catch
* create sugar
* update sugar
  should we instead have Ref[T] or Ref[T, U]?
* distinguishing value parameters from type parameters
* constant expressions

## Open Questions

* lambdas and object literals: are they ever block expressions?
* transform an object literal into an anonymous class and a create call
  needs free variable analysis
  could do the same thing for lambdas
  might not want to do either as they have type checking implications
* `where` might need to have non-type constraints

## Public/Private

* Public access only if the imported module path is not a prefix of the current module path (private access from submodules).
* Perhaps also allow private access if the current module path is a prefix of the imported module path (private access to submodules).
* do we want to be able to have private methods with symbol names?

## Anonymous Types

* anonymous interface types
* allow class/interface definition where a type is expected
  and also where an expr is expected?
* `'type' tuple` where a type is expected
  meaning the type of the expression
* `'interface' tupletype` where a type is expected
  meaning extract an interface from the type

## Strings

trim indent
  look at multiple elements of an apply?

strings are whitespace sensitive
  indent trimming is about reducing that sensitivity

## Resolving Names

No rewrites if a ref resolves to a localref.
Ignore fields and methods when resolving.

typeref
  type or error
staticref
  type, function, or error
ref
  type, function, or unknown
symref
  function or unknown

(prefix ? expr)
  type -> (call ?::create expr)
  function -> (call ? expr)
  unknown -> (apply (select expr ?) ())
(infix ? expr0 expr1)
  type -> (apply (apply expr0 (call ?::create ())) expr1)
  function -> (call ? (tuple expr0 <unpack>expr1))
  unknown -> (apply (select expr0 ?) expr1)
(select ? op)
  type -> (select (call ?::create ()) op)
  > or an error?
  function -> (select (call ? ()) op)
  > or an error?
  unknown -> ERROR
  > if this is always an error, only allow select on a localref
(apply ? expr)
  type -> (call ?::create expr)
  function -> (call ? expr)
  unknown -> (apply (select expr ?) ())
(specialise ? typeargs)
  type ->
  function ->
  unknown ->

TODO:
* bubble up rewrites?
* (apply e0 e1) -> (call (select e0 apply) (tuple e0 <unpack>e1))  ?
