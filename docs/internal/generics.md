# Generics

## WARNING

This is a preliminary document under heavy discussion.
The ideas here are mostly right, the details, not necessarily.
There will be a tutorial document that will be correct in all counts, but this isn't it.

## Overview

Generics are a way to use type parameters to define or restrict other types.
For example, a generic class can use its type parameters to define the type of fields, and a generic method can use its parameters to restrict what can be passed as an argument.

Example:
```ts
class MyGenericType[MyType, SomeInterface]
{
  // MyType is the type of this field
  var field: MyType;

  // SomeInterface is the super type of anything that can be passed to this method
  method(arg: SomeInterface)
}

// Some concrete types that implements Hashable
class MyClass: Hashable { ... }
class MyOtherClass: Hashable { ... }

// This creates a concrete type
let x = MyGenericType[U32, Hashable]::create();

x.field = 42; // OK: The type of the field (U32) is compatible with integral literals

let c = MyClass::create();
x.method(c); // OK: MyClass implements Hashable
let o = MyOtherClass::create();
x.method(o); // OK: MyOtherClass also implements Hashable

// This creates another concrete type, now with a more restrictive argument
let y = MyGenericType[F64, MyClass]::create();

y.field = "hi"; // ERROR: No known conversion between string literals and float types

y.method(c); // OK: This is the expected type
y.method(o); // ERROR: Even though MyOtherType implements the same interface, it's not the same type
```

In the example above, there is one generic class called `MyGenericClass` and two implementations (or specialisations) with types `[U32, Hashable]` and `[F64, MyClass]`.
If all types are used correctly (ex. removing the ERROR lines above), the compiler will create two concrete classes from the one generic class: one for each specialisation.

## Type Parameters

Type parameters are the named types that can be used inside a generic declaration in place of actual types.
In the example above, the generic class `MyGenericClass` has two type parameters: `MyType` and `Hashable`.
Those names are used inside the class in place of existing (declared) types.
In each specialisation, the type arguments (ex. `[U32, Hashable]`) replace the named arguments and the specialisation types are matched 1:1.

Type parameters have the following properties:
* A name: The name that will be used in the generic class as a substitute for the specialisation's passed declared name.
* An optional super-type restriction: What requirements (e.g., interfaces or capabilities) must hold of the type being passed in each specialisation.
* An optional default type: In case the type is omitted from the specialisation declaration, a default type is assumed.
  If no default is provided, a type must be provided at specialisation.

The syntax is: `[Name: Restriction = Default Value]`.

On the example above, only the names were provided.
Even though the names were `MyType` and `SomeInterface`, they could have been initialised with anything, included concrete types, interfaces and capabilities.
It's up to the compiler's type checks to make sure the types passed by each specialisation make sense.
For example, a field with just a capability type cannot be properly initialised, so that's a type error.

Example declaration:
```ts
// A can be any type with any capability
// B can be any type, but must have a mut capability
// C can also be any type, but if none is passed, it's U32
class Generic[A, B: mut, C = U32] { ... }
```
Example uses:
```ts
// A is unrestricted, and B requires it to be mut, while C has a default value
let a = Generic[MyClass & mut, OtherClass & mut]::create();

// Specifying C works too
let b = Generic[MyClass & mut, OtherClass & mut, SomeOther & imm]::create();

// ERROR: B has no default value
let error = Generic[MyClass]::create();

// ERROR: B has to be mut
let error = Generic[MyClass, Otherclass & iso]::create();
```

In the example above, `C = U32` is a potentially misleading declaration.
Having a concrete default type on a unrestricted generic type may lead the implementation to assume users will only pass numeric types.

To correct that mistake, a complete declaration would help:
```ts
type Unsigned = (U8 | U16 | U32 | U64 | U128) & imm;
class Generic[A, B: mut, C: Unsigned & imm = U32 & imm] { ... }
```

However, this is still misleading because `C` could be a union of any combination of the allowed types, for example, `(U8 | U64)` and methods like `+` wouldn't apply, because neither `U8`'s nor `U64`'s `+` method returns `(U8 | U64)`.

To fix that, we can use interfaces to mean _any class that provides those characteristics or behaviours_.
If we have an interface `Numeric` that all numeric types (integral and floating point) implement, and another (ex. `Integer`) that only integral ones do, and another (ex. `Unsigned`) that only unsigned integral types implement, then any type passed, including union types, will have to implement the interface and therefore the call will be a dynamic dispatch and we don't need a `match` on the receiver.

```ts
interface Numeric { }
interface Integer: Numeric { }
interface Unsigned: Integer { }
...
class U8: Unsigned { ... }
...
class U64: Unsigned { ... }
...
class Generic[A, B: mut, C: Unsigned = U32 & imm] { ... }
```

Now, in the `Generic` class above, the parameter `C` can only be one concrete type that implements `Unsigned` (and also `Integer` and also `Numeric`).

Another way to fix this is to use `Self` types, and it shows how powerful the Verona type system is.
If we create an interface that restricts what the `create` method can return to a `Self` type, then it restricts to only concrete types, as those types don't return union types in their constructors.

```ts
// Restricts constructors to only those that return the self concrete type
interface OneOf
{
  create(): Self;
}
type OneUnsigned = OneOf & Unsigned;
...
class Generic[A, B: mut, C: OneUnsigned = U32 & imm] { ... }
```

This works because no `Unsigned` concrete class implements `create` methods which return type is a union, so `C` can only be a single concrete type.

It does, however, have a limitation: The `create` method has no parameters.
If the concrete type you want to use passes arguments to the constructor, it won't implement the `OneOf` interface.

### Type Parameter Values

It is still under discussion how we're going to implement values in type parameters.

For example:
```ts
// A list of a particular type and a compile-time constant size (ex. a vector)
class Vector[T, let size: U32]
{
  ...
}

// A dynamicaly allocated matrix of any size
class Matrix[Type] { ... }

// A function that takes in only lists of a particular type and size
convolution3x3(image: Matrix[F64], filter: Vector[F64, 9]): Matrix[F64] { ... }

// Initialise a matrix and a filter
let matrix = Matrix[F64]::create(...);
let vector = Vector[F64, 3]::create(1, 2, 3);
let vector2 = Vector(1.0, 2.0, 3.0); // equivalent

// ERROR: Types are different because the compile-time sizes are not the same
convolution3x3(matrix, vector); // vector size is 3, expected size is 9
```

## Generic Methods

Methods can declare additional type parameters to further specialise functionality.
These are independent from the class specialisation and must be passed every time the method is called.

Example:
```ts
class Generic[A, B: mut, C: Unsigned = U32 & imm]
{
  ...
  // Converts some Unsigned to D
  to[D: Unsigned](arg: C): D { ... }
}
...
// Create a specialisation with default C = U32
let obj = Generic[SomeType & imm, OtherType & mut]::create();

let x: U8 & imm = 42;
let y: (U8 | U16) & imm = ...;

let z: U32 & imm = obj.to[U32](x); // OK: x is U8, return is U32

let z: U16 & imm = obj.to[U8](x); // ERROR: return not U16

let z: U32 & imm = obj.to(x); // OK: x is U8, return type inferred as U32

let z: U32 & imm = obj.to(y); // ERROR: y is U8 | U16, not U8
```

## Specialisation

Type parameters make classes generic to share implementation for a range of different types ([parametric polymorphism](https://en.wikipedia.org/wiki/Parametric_polymorphism)).
But concrete classes need monomorphic types to be defined at compile time, so that we can have a concrete implementation of a parametric type.
This conversion is called [reification](https://en.wikipedia.org/wiki/Reification_(computer_science)) and is done via a reachability analysis of all possible definitions of each polymorphic type.

Example:
```ts
class Generic[A] { field: A & imm; ... };
...
// Here, `A` is `U32`
let foo: Generic[U32] = ...;

// Here, `A` is `MyClass`
let bar: Generic[MyClass] = ...;
```

In the example above, there is one polymorphic class, `Generic[A]` where `A` is the polymorphic type parameter, and two concrete classes (specialisations): `Generic[U32]` and `Generic[MyClass]`.

The compiler generates the required specialisations and replaces each definition with its respective concrete classes.
Without loadable code (separate compilation processes), the compiler then removes the generic class.
Since generic classes can only be used by specialisation, all other (potentially infinite) possibilities of types parameters don't exist in the program, so the generic class is elided and we're left with only the specialisations.
The effect of loadable code is still uncertain in this design and under discussion.

The end result would be:
```ts
// Definition of monomorphic classes only
class Generic_U32 { field: U32 & imm; ... }
class Generic_MyClass { field: MyClass & imm; ... }

// Direct use of the concrete classes
let foo : Generic_U32 = ...;
let bar : Generic_MyClass = ...;
```

Note that type unions (`A | B`) are still monomorphic types.
To resolve which implementation works with each inner type (`A` or `B`) one must use `match` inside the method.

Example:
```ts
interface Addable[A]
{
  // Adds itself to other returning the same type of other
  // All numeric types implement this interface
  +(self, other: A): A;
}
class Generic[A] {
  field: A & imm;
  // Add matches on the implementation of a + method, via Addable
  add(other: A & imm) {
    match self.field
    { x: Addable[A] => x + other }
    { x: String => x.append(other) } // Some other notion of "adding"
    { _ => throw Something } // Or some better error handling
  }
}
...
// This creates a monomorphic union type
let foo: Generic[U32 | String] = ...;

// This creates a monomorphic single type
let bar: Generic[U32] = ...;

// This will do dynamic dispatch of `field.+(other)`
let baz: Generic[U32 | F64] = ...;
```

On the first case, `foo` is a (monomorphic) union between `U32` and `String`, which means, at run-time, either can be passed to `add` and we can't decide at compile time.
So the implementation retains the `match` and will check if the types implement `Addable` (both do).
Note that both `field` and `other` must have the same type (both are `A`).

However, on the second case, `bar` is a compile-time single monomorphic type, which guarantees that it will always implement the `Numeric` interface and therefore there's only one `match` statement that fits.
In this case, the compiler is free to elide the `match` altogether and emit only the case where the type is `Addable`, ie. `self.field + other`.
In this case, the type is guaranteed to be `U32` and the call to `+` is a static call to `U32::+(U32, U32)`.

Finally, in the last case, `baz`'s types are both `Addable`, but the `+` operator for each is different.
When searching for which operator to call on `field + other`, the types aren't guaranteed to be the same, so `field` can be `U32` while `other` `F64`.
In fact, because there is no restriction (`match`) for `other`, it's still `U32 | F64`, and there are no `+` methods on either `U32` or `F64` with the signature `+(Self & imm, (U32 | F64) & imm) : Self & imm`, so this is a type error.

The correct part of the code above results in the following compile-time concrete types:
```ts
// (U32 | String) need different initialisation and add match
class Generic_U32_String {
  field: (U32 | String) & imm;
  // Generic add
  add(other : (U32 | String) & imm) {
    // Different behaviour for each type
    match self.field
    { x: U32 => U32::+(field, other) }
    { x: String => String::append(x, other) }
    // All cases exhausted, no need for error handling
  }
}

// Just U32 elides everything
class Generic_U32 {
  field : U32 & imm;
  // Specific add
  add(other : U32 & imm) {
    U32::+(field, other)
  }
}

// Then replace the definitions with the monomorphic types
let foo : Generic_U32_String = ...;
let bar : Generic_U32 = ...;
```

The compiler output will not contain any functions or data structures for the unspecialised generic polymorphic type `Generic[A]`.