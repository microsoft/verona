# Object's memory model

In Verona, there are no implicit deep copies of objects and no implicit initialisation.
Constructors need to be called explicitly.
Any static function that returns an object of a certain type is a constructor for that type.

Example:
```
  class Foo {
    // The "standard" constructor
    create() : Foo { ... }
    // Some other constructor
    build(a: Bar, b: Baz) : Foo { ... }
    ...
    // If we allow overload, this can also work
    create(x: SomeType) : Foo { ... }
    ...
  };

  func() {
    // Explicitly call the constructor
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

All copies (assignment) are done by reference, and so the types must be compatible (sub-type).
There are no implicit deep copies, no implicit allocations.

Example:
```
  let a : Foo & mut = Foo();

  // Now, both `a` and `b` are references to the same object.
  let b : Foo & mut = a;

  // Change all
  a.field = 42;
  print(b.field); // This yields `42`.
```

## Freezing

Freezing must be an explicit call (no sugar).
Frozen (immutable) regions cannot be thawed.

Example:
```
  let a : Foo & iso = Foo();
  let b : Foo & imm = Builtin::freeze(a);

  // No guarantees some random behaviour is using our immutable data
  let c : Foo & iso = b; // ERROR
```

Freezing a mutable object is an error, as we are not using the entry point (isolated reference).
To freeze one mutable object, one needs to find the `iso` and freeze that.
Freezing an `iso` freezes **all** objects reachable (dominated by) that `iso` reference.

## Changing region types

As stated, there are no automatic conversion, not even between region types.
Creating an object in one region type and then moving into another requires users to code that explicitly.
There isn't a syntax for selecting regions types yet, but the code must deep copy by hand.

The idea is to create a new region, `when` both, and then copy element by element.
This is tedious and error-prone, but can be made better by coding deep-copies in the classes themselves.
