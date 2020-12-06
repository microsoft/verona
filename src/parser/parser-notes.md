# Parser Notes

* multiple definitions of `apply` produces a poor error message
* tests need rewriting for the new syntax rules
* transform prefix and infix to apply
  this needs to know if operators are free functions or not
* try/catch
* create sugar
* update sugar
  should we instead have Ref[T]?
* distinguishing value parameters from type parameters
* constant expressions

## Open Questions

* lambdas and object literals: are they ever block expressions?
* transform an object literal into an anonymous class and a create call
  needs free variable analysis
  could do the same thing for lambdas
  might not want to do either as they have type checking implications

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

## Operators

if op is not a static function in scope
  (prefix op expr) -> (apply (select expr op) ())
  (infix op expr0 expr1) -> (apply (select expr0 op) (tuple <unpack>expr1))

if op is a static function in scope
  (prefix op expr) -> (call op (tuple expr))
  (infix op expr0 expr1) -> (call op (tuple expr0 <unpack>expr1))
