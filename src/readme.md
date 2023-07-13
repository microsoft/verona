# TODO

Error messages:
- Too many repetitions from implicit methods.
  - `defaultargs`, `partialapp`, `autocreate`, `autofields`, etc.

Code reuse:
- Code reuse must be intersections of classes and traits only, recursively through type aliases.
- Logical order: (1) defaultargs, (2) inheritance, (3) partialapp.
  - Can actually do inheritance first, treating defaultargs as blocking multiple arities.
- Do textual inclusion of any member or method that isn't already defined.
  - Need to do type substitution on the included code.

- Automatically insert `use std::builtin`.
- Better system for including parts of `std`.
- Check that default types for type parameters satisfy predicates.

Tuples are traits:
```ts
type Tuple[T, U] =
{
  head(self): self.T
  rest(self): self.U
}

// make unit a 0-arity tuple
class Unit: Tuple[(), ()]
{
  head(self): () = ()
  rest(self): () = ()
}

Unit: Tuple[Unit, Unit]
T1: Tuple[T1, Unit]
(T1, T2): Tuple[T1, Tuple[T2, Unit]]
(T1, T2, T3): Tuple[T1, Tuple[T2, Tuple[T3, Unit]]]
(T1, T2, T3, T4): Tuple[T1, Tuple[T2, Tuple[T3, Tuple[T4, Unit]]]]

match w
{
  // matches Unit
  { () => e0 }
  // matches {} (x = w)
  { x => e1 }
  // matches tuple_1 (x = w._0, y = w._1plus)
  // problem: for w: (T1, T2), we want y: T2, not y: Tuple[T2, Unit]
  { x, y => e2 }
  // matches tuple_2 (x = _0, y = _1, z = _2plus)
  { x, y, z => e3 }
  // explicity indicate a w._1plus match?
  { x, y... => e2plus }
}

// experiment: tuple types
class tuple_1[T1]
{
  size(self): Size = 1
  apply(self, n: Size): T1 | Unit = if (n == 0) { self._0 }
  _0(self): T1
}

class tuple_2[T1, T2]
{
  size(self): Size = 2
  apply(self, n: Size): T1 | T2 | Unit =
    if (n == 0) { self._0 }
    else if (n == 1) { self._1 }

  _0(self): T1
  _1(self): T2
}

class tuple_3[T1, T2, T3]
{
  size(self): Size = 2
  apply(self, n: Size): T1 | T2 | T3 | Unit =
    if (n == 0) { self._0 }
    else if (n == 1) { self._1 }
    else if (n == 2) { self._2 }

  _0(self): T1
  _1(self): T2
  _2(self): T3
}

type typelist[T] =
{
  size(self): Size
  apply(self, n: Size): T | Unit
}

```

Associated types

Mangling
- need reachability to do precise flattening
- for dynamic execution:
  - use polymorphic versions of types and functions
  - encode type arguments as fields (classes) or arguments (functions)

Pattern Matching
- values as arguments
- exhaustiveness
- backtracking?

Late loads of code that's been through some passes
- delay name lookup
- only tricky part is `create` sugar

Type Descriptor
- sizeof: encode it as a function?
```c
%1 = getementptr [0 x %T], ptr null, i64 1
%2 = ptrtoint ptr %1 to i64
```
- sizeofptr: could do this for primitive types
  - 8 (i64) for most things, 1 (i8) for I8, etc
- trace: could be "fields that might be pointers", or encoded as a function
- finalizer: a function
- `typetest`: could be a function
- with sizeof, trace, finalizer, and typetest encoded as functions, they could have well-known vtable indices, and the type descriptor is then only a vtable
- vtable: could use linear/binary search when there's no selector coloring

LLVM lowering
- types-as-values?
  - encode class type arguments as fields?
  - pass function type arguments as dynamic arguments?
    - use the default if the typearg isn't specified, or the upper bounds if there's no default
  - insert type tests (for both args and typeargs) as function prologues?
- mangling
  - flatten all names, use fully-qualified names
- `fieldref`
- Ptr, Ref[T], primitive types need a way to find their type descriptor
- `typetest`
  - every type needs an entry for every `typetest` type
- dynamic function lookup
  - find all `selector` nodes
  - every type needs an entry for every `selector` name
- only `Ref[T]::store` does llvm `store` - it's the only thing that needs to check for immutability?
- literals: integer (including char), float, string, bool
- `copy` and `drop` on `Ptr` and `Ref[T]`
  - implementation depends on the region type
- strings can't be arrays without type-checking
- region types, cowns, `when`
- could parse LLVM literals late, allowing expr that lift to reflet and not just ident
- destructuring bind where a variable gets "the rest" or "nothing"
  - ie lhs and rhs arity don't match
  - include destructuring selectors on every `class`?
  - make them RHS only? this still breaks encapsulation
  - `destruct` method, default impl returns the class fields as a tuple

- `Package` in a scoped name, typeargs on packages
- free variables in object literals
- mixins
- match
- lazy[T]
- weak[T]?
- public/private
- package schemes
- list inside TypeParams, Params, TypeArgs along with groups or other lists

## Key Words

get rid of capabilities as keywords
- make them types in `std`?
- or just handle those `typename` nodes specially in the typechecker?

## Lambdas

type of the lambda:
- no captures, or all captures are `const` = `const`, `self: const`
- any `lin` captures = `lin`, `self: lin`
- 0 `lin`, 1 or more `in`, 0 or more `const` = `lin`, `self: in`
- don't know if any `out` captures

## Lowering

- mangled names for all types and functions
- struct for every concrete type
- static and virtual dispatch
- heap to stack with escape analysis
- refcount op elimination

## Ellipsis

`expr...` flattens the tuple produced by `expr`
- only needed when putting `expr...` in another tuple

`T...` is a tuple of unknown arity (0 or more) where every element is a `T`
- `T...` in a tuple flattens the type into the tuple
- `T...` in a function type flattens the type into the function arguments

```ts
// multiply a tuple of things by a thing
mul[n: {*(n, n): n}, a: n...](x: n, y: a): a
{
  match y
  {
    { _: () => () }
    { y, ys => x * y, mul(x, ys)... }
  }
}

let xs = mul(2, (1, 2, 3)) // xs = (2, 4, 6)
```

## Lookup

lookup in union and intersection types

may need to check typealias bounds during lookup
- `type foo: T` means a subtype must have a type alias `foo` that's a subtype of `T`.

## param: values as parameters for pattern matching

named parameters
- (group ident type)
- (equals (group ident type) group*)
pattern match on type
- (type)
pattern match on value
- (expr)
