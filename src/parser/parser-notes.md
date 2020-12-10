# Parser Notes

* special types: iso, mut, imm, Self
* multiple definitions of `apply` produces a poor error message
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

> TODO: need to resolve through type aliases
~> <~ : right-most
-> : fail
, : fail
& : lookup on all entity types, succeed if one and only one success
| : fail
for typeparams, check the bounds

eliminates: infix, prefix, staticref
introduces: function (or use staticref for this)

staticref
  type
  function
  error

staticref can appear here:
  alone
  (select X op)
  (apply X tuple) [no-rewrite]
  (prefix X X)
  (infix X X X)

expr
  type ->
    lookup ...::create
    if it is 0 args
      (apply (function ...::create) ())
    else
      (function ...::create)
  function -> (function ...)

(select function member)
  ->
  (select (apply function ()) member)

(prefix unknown expr)
  ->
  (apply (select expr unknown) ())
(prefix _ expr)
  ->
  (apply _ expr)

(prefix op function)
  ->
  (prefix op function)
(prefix op unknown)
  -> ERROR

(infix function expr1 expr2)
  ->
  (apply function (tuple <unpack>expr1 <unpack>expr2))
(infix unknown expr1 expr2)
  ->
  (apply (select expr1 unknown) expr2)

TODO: from here
(infix op expr1 expr2)
  op = type -> (apply (function type::create) (tuple expr1 expr 2))
    x T y -> T::create(x, y)
  op = function -> (apply function (tuple expr1 expr2))
  op = unknown -> (apply (select expr1 op) expr2)

(infix op expr1 expr2)
  expr1 = type -> (infix op (apply (function type::create) ()) expr2)
  expr1 = function -> (infix op function expr2)
  expr1 = unknown -> ERROR

(infix op expr1 expr2)
  expr2 = type -> (infix op expr1 (apply (function type::create) ()))
  expr2 = function -> (infix op expr1 function)
  expr2 = unknown -> ERROR
