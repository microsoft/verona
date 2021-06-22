# Object's memory model

In Verona, there are no implicit deep copies of objects and no implicit initialisation.
Methods used to construct a type (aka. factory methods, constructors) need to be called explicitly.
Any static function that returns an object of a certain type is a factory method for that type.

Example:
```
  class Foo {
    // "Foo()" is sugar for "Foo::create()"
    create() : Foo { ... }

    // Some other factory method, not sugared but always explicit
    build(a: Bar, b: Baz) : Foo { ... }

    ...

    // If we allow overload, this can also work from sugared "Foo(SomeType)"?
    create(x: SomeType) : Foo { ... }

    ...
  };

  func() {
    // Explicitly call the factory method
    let a = Foo::create();

    // This is sugar for `Foo::create()`
    let b = Foo();

    // Careful, this is sugar for `Foo::apply()`
    let c = b();

    // If we allow overload, is this valid?
    let x : SomeType = ...;
    let c = Foo(x); // Does type match propagate through sugar?
  }
```

## Variables as references

All Verona variables are pointers. So assignment is done by pointer copy and the types must be compatible (sub-type).
There are no implicit deep copies, no implicit allocations.
There is, however, an `update` sugar, which allows users to programatically add deep-copy (or any other) logic to assignments.

Example:
```
  class Foo {
    value : I32;

    // Implement the sugar factory method
    create() { ... };

    // Implement the sugar copy factory method
    update(I32 arg) { value = arg; };
  };

  // This is sugar to `Foo::create()`
  let a : Foo & mut = Foo();

  // Now, both `a` and `b` are pointers to the same object.
  let b : Foo & mut = a;

  // Change all
  a.field = 42;
  print(b.field); // This yields `42`.

  // This creates a new pointer to Foo, uninitialised
  let c : Foo & mut;

  // The object `c` doesn't exist yet!
  c.field = 42; // ERROR

  // This is sugar for `c.update(b)`
  c() = b;

  // Now `c` is a copy of `b`
  print(c.field); // This yields `42`.
```

## Freezing

Freezing must be an explicit call (no sugar for now).
Frozen (immutable) regions cannot be thawed.

Example:
```
  let a : Foo & iso = Foo();
  let b : Foo & imm = Builtin::freeze(a);

  // No guarantees some random behaviour is using our immutable data
  let c : Foo & iso = b; // ERROR
```

Verona access memory via regions.
You either have access to the whole region or none of it.
Therefore, you cannot freeze a single object inside a region, but must freeze the entire region.
The freeze function receives a `T & iso` type, so freezing a mutable object is a type error.

## Changing region types

As stated, there are no automatic conversion, not even between region types.
Creating an object in one region type and then moving into another requires users to code that explicitly.
There isn't a syntax for selecting regions types yet, but the code must deep copy by hand.
