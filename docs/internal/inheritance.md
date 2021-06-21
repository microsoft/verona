# Inheritance

In Verona, class inheritance is not the same as sub-typing.
A type is a sub-type of another if explicitly declared by the programmer.

Example:
```
  interface Foo {
    f(...);
  };
  class Bar : Foo {
    f(...) { ... }
  };
```

Here, `Bar` is a sub-type of `foo`, explicit in the declaration `Bar : Foo`.
However, `Bar` also has implementation inheritance, because it implements `Foo`'s `f` function.

## Capabilities

Like classes, capabilities can be sub-types of each other.
This is used to allow objects with different capabilities to reference each other.

Example:
```
  let x : Foo & mut = ...;

  // This is fine, as `readonly` is a sub-type of `mut`
  let y : Foo & readonly = x;

  // This is not
  let z : Foo & imm = x; // ERROR
```

Some capabilities can be converted to each other in special ways, even if they're not a sub-type of each other.

Example:
```
  let foo : Foo & iso = ...;

  // This works because access to `foo` as `iso` means you own it and can freeze it
  let frozen : Foo & imm = Builtin::freeze(foo);

  // Here, `foo` is no longer valid
```
