# Unified Call Syntax

Verona does not distinguish between functions, static functions, or methods. All functions are defined within some class (possibly the module class), and all functions can be used in any of the three roles.

There are multiple ways to refer to functions:

* Dynamic unbound selection
* Dynamic bound selection
* Fully-qualified reference
* Unqualified reference
* Fully-qualified infix
* Unqualified infix

## Dynamic Unbound Selection

The syntactic form `object::identifier` looks up a function on a dynamic value. The identifier can have type arguments. The function is looked up on the dynamic (concrete) type of the of the object, which may be more precise than the static type. No arguments are bound.

The static type of `object` must indicate that look-up on the dynamic type will succeed.

```ts
// This looks up `foo` with no type arguments on the dynamic type of `x` and
// stores the resulting function in `f`.
let f = x::foo;

// This calls `x::foo` with the argument `4`.
f 4;

// This also calls `x::foo` with the argument `4`.
x::foo(4);
```

## Dynamic Bound Selection

The syntactic form `object.identifier` works like dynamic unbound selection, but with one additional step: the `object` is bound as the first argument to the function. That is, after look-up, the function is partially applied.

```ts
// This looks up `bar` on the dynamic type of `x` and binds `x` as the first
// argument.
let f = x.bar;

// This calls x::bar(x, "hi")
f "hi";

// This also calls x::bar(x, "hi")
x.bar("hi");
```

## Fully-Qualified Reference

The syntactic form `type::identifier` is a direct reference to a function defined in a type. The type can itself be fully qualified, and the function may optionally have type arguments. No arguments are bound, and the function must be defined in that type, not just declared. That is, if the type is an interface, it must provide a default implementation of the function.

The function reference can be stored, or can be used in application in the same way as any other value.

```ts
// This stores a statically known function in `f`.
let f = Foo::bar;

// This calls Foo::bar(4).
f 4;

// This also calls Foo::bar(4).
Foo::bar(4);
```

## Unqualified Reference

An unqualified function name `identifier` (as distinct from a qualified name such as `ident1::ident2`) can be used. If there is a function in the current scope (either defined in the current class or made available without qualification with a `using` directive), this will resolve as if it were a fully-qualified reference.

If there is no static resolution to a function, this will instead resolve as if it were a dynamic unbound selection on the first argument. It's a compile-time error to try to use an undefined function with no arguments.

```ts
// The function `foo` must be in scope.
let f = foo;

// If `foo` is in scope, this calls `foo(x)`. If `foo` is not in scope, this
// calls `x::foo(x)`.
foo x;
```

## Fully-Qualified Infix

A fully-qualified reference used in an infix position receives arguments from both the left-hand side and the right-hand side. If either or both arguments are tuples, the tuples are concatenated into a single tuple.

```ts
// These are all equivalent.
Foo::bar(w, x, y, z);
w Foo::bar (x, y, z);
(w, x) Foo::bar (y, z);
(w, x, y) Foo::bar z;
```

## Unqualified Infix

An unqualified reference used in an infix position receives arguments from both sides in the same way as a fully-qualified reference used in an infix position. The unqualified reference is then either resolved in the current scope or looked up on the first argument.

```ts
// These are all equivalent. If `foo` is in scope, it is used. Otherwise,
// `w::foo` must exist and is used.
foo(w, x, y, z);
w foo (x, y, z);
(w, x) foo (y, z);
(w, x, y) foo z;
```

## Tuples as Arguments

A tuple as an argument to a function is unpacked into a sequence of arguments. In order to pass a tuple in a specific argument position, nest the tuple.

```ts
// These are equivalent
f a b c;
f(a, b) c;
f a (b, c);
a f (b, c)
(a, b) f c;
a f b c;

// If the second argument is a tuple, nest it.
f(a, (b, c));
```

## Default Arguments

> TODO
