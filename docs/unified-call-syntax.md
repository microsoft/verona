# Unified Call Syntax

Verona does not distinguish between functions, static functions, and methods. All functions are defined within some class (possibly the module class), and all functions can be used in any of the three roles.

There are multiple ways to refer to functions:

* Dot notation
* Application-like sequences
* Infix-like sequences

## Dot Notation

The syntactic form `e0.name` looks up `name` on the dynamic value of `e0`, and `e0` is the "left-hand side" of the expression. If it's followed by a tuple, i.e. `e0.name(e1, e2)`, then the tuple is the "right-hand side" of the expression. If there is no tuple, then the right-hand side is empty. Dot notation is the tightest bounds.

## Application-like Sequences

The syntactic form `e0 e1` is syntactic sugar for `e0.apply(e1)`. It binds less tightly than dot notation, so `e0 e1.f(e2)` treats `e1.f(e2)` as the right-hand side, i.e. `e0.apply(e1.f(e2))`.

## Infix-like Sequences

If an element in a sequence is a selector, rather than a value, it's treated differently. A selector is an optional sequence of `::` separated type references, followed by a function reference, for example `Foo[Bar]::Quex::some_function[U64]`, or simply `f` where `f` is not a local identifier.

Such a selector acts as loosely bound dot notation. If there is no preceding value, then the left-hand side is empty, and if there's no following value, then the right-hand side is empty. For example, `e0 selector e1` is `e0.selector(e1)`, `selector e0` is `().selector(e0)`, and `e0 selector` is `e0.selector()`. This binds at the same tightness as application, i.e. more loosely than dot notation, so `e0.f selector e1.g(e2)` is `e0.f.selector(e1.g(e2))`.

## Overload Resolution

Functions are first-round candidates if:
1. They have an arity that matches the concatenation of the left-hand and right-hand side arguments.
2. The static types of all arguments match.

Functions are second-round candidates if:
1. They have an arity that matches the right-hand side arguments.
2. The static types of all arguments match.

In each round, a static function is preferred if one is available, otherwise a dynamic function is selected. If no function is available in a round, resolution moves on to the next round. Dynamic look up type checks based on the static type of the first argument and resolves at run-time based on the dynamic type of the first argument.

Open question: if more than one function is available in a round, is this always an error, or do we provide a resolution order for the "most specific" static types?

## Named Arguments

Open question: should named arguments be supported?

## Default Arguments

Default arguments are equivalent to arity-based overloading. For example, with two parameters, one of which has a default, a function can be used as arity/1 or arity/2. Similarly, a function with 8 parameters, all of which have defaults, can be used for any arity between 0 and 8.

## Variable Length Arguments

Use type lists to write functions with a variable number of arguments.
