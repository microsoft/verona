# To Do

- `partialapp` needs to use FQN when extracting type parameters.
- Blocks.
  - Enforce `Return | Move | LLVM` at the end?
- Type inference.
  - Uses of `typevar`:
    - Return type of `new`, auto-create, lambda create, lambda apply, partial functions, partial app class create, partial app class apply.
    - Lambda free variable types.
    - `DontCare` syntactically in a type (remove this?).
    - Unspecified field types, parameter types, function return types.
- Check that `var` is stored to before it's loaded.
- Free variables in object literals.
- Tuples as traits.
- Pattern matching.
- Type lists.
- `lazy[T]`
- `weak[T]`
- Lookup in a `TypeParam`, e.g., `create` or an associated type.
- Public/private.
- Package schemes.
  - Better system for including parts of `std`.
  - `Package` in a scoped name, `typeargs` on packages.
- List inside `TypeParams`, `Params`, `TypeArgs` along with groups or other lists.
- Lowering.
  - Reachability.
  - Selector coloring.

## Conditional Compilation

Use `where` for conditional compilation:
- As an expression block.
- Over fields.
- Over chunks of a class body?
- Will it all nicely type check for all platforms?

## Tuples as Traits

Tuples as traits:
```ts
match w
{
  // matches Unit
  { () => e }
  // matches Any (x = w)
  { x => e }
  // matches tuple_2 (x = w._0, y = w._1)
  { x, y => e }
  // matches tuple_3 (x = w._0, y = w._1, z = w._3)
  { x, y, z => e }
  // explicity indicate a tuple_1+ match
  // x = w._0, y = w._1plus
  { x, y... => e }
}

x, y... = e // e must be tuple_1
x, y = e // e must be tuple_2
x, y, z... = e // e must be tuple_2
x, y, z = e // e must be tuple_3

// experiment: tuple types
class unit
{
  _0(self): unit = self
  _1plus(self): unit = self
}

type ituple_1 =
{
  size(self): Size = 1
  apply(self, n: Size): Self | () = if (n == 0) { self }
  _0(self): Self = self
  _1plus(self): ()
}

type ituple_2[T1, T2] =
{
  let _0: T1
  let _1: T2
  size(self): Size = 2
  apply(self, n: Size): T1 | T2 | () =
    if (n == 0) { self._0 }
    else if (n == 1) { self._1 }
  _1plus(self): (T2, ())
  _2plus(self): ()
}

class tuple_2[T1, T2]: ituple[T1, T2] {}

type ituple_3[T1, T2, T3] =
{
  let _0: T1
  let _1: T2
  let _2: T3
  size(self): Size = 3
  apply(self, n: Size): T1 | T2 | T3 | () =
    if (n == 0) { self._0 }
    else if (n == 1) { self._1 }
    else if (n == 2) { self._2 }
  _1plus(self): (T2, T3, ())
  _2plus(self): (T3, ())
  _3plus(self): ()
}

class tuple_3[T1, T2, T3]: ituple[T1, T2, T3] {}

type typelist[T] =
{
  size(self): Size
  apply(self, n: Size): T | ()
  rest(self, n: Size): typelist[T]
}

```

## Associated types

Needs discussion.

## Mangling

- need reachability to do precise flattening
- for dynamic execution:
  - use polymorphic versions of types and functions
  - encode type arguments as fields (classes) or arguments (functions)

## Pattern Matching

- values as arguments
- exhaustiveness
- backtracking?

## Incremental Compilation

Late loads of code that's been through some passes
- delay name lookup
- only tricky part is `create` sugar

## Lowering

- mangled names for all types and functions
- struct for every concrete type
- static and virtual dispatch
- heap to stack with escape analysis
- refcount op elimination

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
