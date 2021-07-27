# Verona Type System

## WARNING

This is a preliminary document under heavy discussion.
The ideas here are mostly right, the details, not necessarily.
There will be a tutorial document that will be correct in all counts, but this isn't it.

## Overview

Verona is a statically typed language, with a structural and algebraic type system.
That is an information-dense sentence, so let's take pieces in turn.

Verona's type system is such that objects always have exactly one, fully-specified type (its "concrete" type).
Verona's type safety ensures that there is no mechanism to view an object of one concrete type as though it had another.
However, variables may be typed by a set of constraints instead of a concrete type;
a variable so typed may refer to objects of various concrete types, so long as each type satisfies all the constraints.
For example, variables can have abstract (interface) types if the objects bound to it are in a set of concrete (class) types that implement that interface.
Another example is a type union, where a variable can be one of many types, but the code must deal with all possible types.
'Statically typed' means that the verification of these properties is done at compile time.

The 'structural' bit means that open-world types can be defined by their shape: If an interface defines only a method `foo()` then any class that implements a method `foo()` implements this interface.
This is in contrast to nominal typing, such as in Java, where an object implements an interface only if it adopts that interface.
Objective-C supports both, with protocols defining nominal types but dynamic dispatch and explicit method checks providing structural typing.

The 'algebraic' bit means that we can define types in terms of intersections or unions on other type expressions.
For example, we can say `Integer | String` to define a set of constraints on a type that mean that it must either be an integer or a string (whatever they are: they may be concrete types, or they may be interfaces) or something like `Copyable & Hashable` to say any type that meets the constraints of the copyable and hashable interfaces.

All types should be known at compile time and the compiler can assert that the objects and operations have the correct type at all times.
Some types are inferred by the compiler, but there is no guessing: there's either one clear inferred type or it's an error.

At a glance, Verona types are:
 * Interfaces: abstract types that specify a set of fields and methods that an underlying concrete type must implement.
   Interfaces may provide default implementations of methods that a class can adopt.
 * Classes: concrete types.
   A class may inherit from other classes but this does not make the class a subtype of the class from which it inherits.
   (Verona does not have a subtype relationship between concrete types, only between interface types.)
   Classes may explicitly adopt interfaces, in which case they adopt the default implementations from the interface, but conformance to an interface is entirely a property of the resulting structural type.
 * Capability rights: A Verona pointer is a *capability*: an unforgeable token of authority that permits access to an object. 
   The set of rights that a capability holds are associated with a type, for example. `iso`, `mut`, `imm` and `readonly`.
 * Unions: The concrete type of the object must fulfil all of the constraints of at least one out of two or more type expressions, for example the union type `(A | B)` must fulfil all of the requirements of either `A` or `B`.
   Unions may refer to concrete types (for example, if `A` and `B` are classes then a variable of type `A | B` may be a reference to an instance of either class) or other type expressions, in which case the variable may refer to any concrete type that meets the requirements of either set of constraints.
 * Intersections: The concrete type of the object must fulfil all of the constraints of two or more type expressions, for example the union type `(A & B)` must fulfil all of the requirements of both `A` and `B`, which may be interfaces or other type expressions.
   No type satisfies the union type of two distinct concrete types, so if `A` and `B` are both class types then `A` & `B` is a type that is not possible to satisfy.

Verona manages memory by associating objects with 'regions'.
Each region has a single 'sentinel' object; all live objects in a region are reachable from its sentinel without crossing into another region.
To ensure data-race freedom, Verona limits programs to non-concurrent access to all objects of each region by carefully tracking the flow of references to the region's sentinel.

Capability types are the mechanism Verona uses to offer such language and runtime guarantees:
 * `iso`, or `isolated`, means this is a pointer to an the sentinel object in a region.
   Every region has a single sentinel that dominates all of the live objects in the region.
   Exactly one `iso` pointer exists for any region and the behaviour accessing any mutable object must hold the `iso` pointer for the sentinel object of the corresponding region.
 * `mut`, or `mutable`, means this object can be modified.
   All mutable objects are contained by a region, the behaviour modifying the object must own the `iso` pointer that refers to the sentinel object of the region.
 * `imm`, or `immutable`, means no one can write to, so anyone can read from at any time.
   You can make `iso` regions immutable but not the other way round, as there's no guarantees who's still reading from it.
 * `readonly` defines local immutability.
   No object reached via a `readonly` capability may be modified via that the object is not guaranteed to be immutable.
   `mut` and `readonly` capabilities to the same object may be visible at the same time.

Since capabilities, unions and intersections are types themselves, an `(A & imm)` is completely different from `(A & mut)` or `(A & iso)`.
Specifically, you cannot convert `mut` to `imm` (as you would with a `const` cast in C++), or vice versa.
A `freeze` operation consumes an `iso`, destroying the only copy of the capability that is required to mutate any object in a region, and returns an `imm` capability.
There is no corresponding `unfreeze` operation to consume an `imm` capability and provide any kind of mutable capability.
Most importantly, capabilities are not type traits like C++'s `const int * const`, they have specific monotonic semantics that change the set of operations that are permitted with the resulting capability.

## Module / Class model

In Verona, modules are classes.
Classes and functions inside modules are just nested types and methods of the module's class.

Modules are composed of all Verona files in a specific directory.
But there is no special relationship between modules in hierarchical directory structure.
Sub-directories are not part of the module, nor visible in any special way to the module in the parent directory.

For example, the following directory structure:
```
myproject +-- main.verona
          |-- driver.verona
          `-- packages +-- Module1 +-- foo.verona
                       |           `-- bar.verona
                       `-- Module2 +-- baz.verona
                                   |-- ber.verona
                                   `-- Module3 +-- mod3.verona
```

translates to the following module structure:
```ts
class myproject {
  // both main and driver inserted here
}
class Module1 {
  // both foo and bar inserted here
}
class Module2 {
  // both baz and ber inserted here
}
class Module3 {
  // mod3 inserted here
}
```

The structure of modules in sub-directories should be used for organisational purposes and won't change the visibility of their definitions.

The discovery path for modules is done in the following order:
1. **Local path.**
   For example, `using Module3` inside `Module2` will find it directly.
2. **The project's `packages` directory.**
   For example, `using Module2` inside `Module1` will find it via `packages`.
3. **The compiler's standard library location.**
   For example, `using numbers` inside any package will find the standard library implementation.
   The location of the standard library is up to the compiler and irrelevant from the users' perspective.

If a module cannot be found in any of these three locations, the compiler emits an error.

The use of relative paths (example `using ../../Module1` inside `Module3`) is allowed.
However, the package maintainer is responsible to making sure the structure holds on any user environment.
This usually means you should not go beyond the boundaries of your own directory structure.
Specifically, it is dangerous for `Module2`'s implementation to refer to `Module1` in that fashion, which could cause lookup failure on ill-formed directory structures.

Mechanisms for importing packages using URIs (ex. `github:developer:project:version`) is under discussion but we have not finalised how it would look like.
The consensus is to have a closed interpretation of how to handle each URI to guarantee integrity and security of the modules.
New handlers should be incorporated into the compiler via pull requests, and not via third-party plugins, which could potentially be exploitable.

### Visibility

There are only two types of visibility: `public` and `module private`.
Module private limits the visibility to classes and files on the same module, while public extends the visibility to all modules.

Verona does not provide any equivalent of `private` or `protected` from C++, Java, and so on.
Visibility is primarily a software-engineering abstraction to prevent implementation details leaking past abstraction boundaries.
Verona treats a module as an abstraction and packaging boundary.

Any form of protection within an abstraction boundary is purely advisory: the author of the module can trivially modify the code to override a specifier.
As such, we do not believe that class-private has any value in Verona: any code that is able to modify an object's package-private fields is part of the same abstraction and distribution unit and so can be updated at the same time as any changes to that field's behaviour, without affecting any external APIs.

Verona does not allow subtyping of concrete types and so has no need of a `protected` visibility specifier.

The concept of visibility is different depending on what we refer to.

If public:
* Classes can be used as types by other modules.
* Methods can be called by other modules.
* Fields can be directly read and written to by other modules.

Since sub-directories have no special relationship with their parent directories, `module private` does not permit modules in parent/child directory to see each others' private types.

## Classes and Interfaces

Classes and interfaces have a similar structure:
 * Fields: elements of a specific type that have a runtime space allocated.
 * Methods: functions that belong to a type.

The declaration is trivial:
```ts
// Helps receivers know they can call a function named increment
interface Example
{
  // All derived classes will have the common field
  var common : I64;

  // Classes that implement this interface must implement this method and provide the field
  // Classes that inherit from this interface must implement this method but get the field from the definition above
  increment();
};

// This class inherits, so it gets the field `common` from the interface
class MyType : Example
{
  let field : I32;

  // Some other method
  innerSum(x : I32) : I32
  {
    let res : I32 = x + field;
    res
  }

  // Implements `increment`
  increment()
  {
    self.common += 1;
  }
};

// This class aims to implement the interface without explicit inheritance, so it must provide the `common` field
class OtherType {
  // Must provide the field
  var common : I64;

  // Implements `increment`
  increment()
  {
    self.common += 2;
  }
};
```

### Fields

Define objects that belong to the class or interface and will be instantiated (ie. use memory) on every instance of the class.

Fields can be initialised in two ways:
 * `var`: Must be initialised on construction, but can change after.
 * `let`: Must be initialised on construction, and cannot change later, like const fields in C++.

They can also have a keyword `embed`, which means they're placed in the struct directly, as opposed to the default of being a pointer to the data.

For example
```ts
// This struct layout would be { F64, F64 }
class Point
{
  embed var x : F64;
  embed var y : F64;
};

// This struct layout would be { Ref[Point], Ref[Point] }, same as { UIntPtr, UIntPtr }
// The following definition requires three separate memory allocations, one for each point and one for the line, which contains pointers to each point.
class Line
{
  var orig : Point;
  var dest : Point;
}
// In contrast, this version requires a single memory allocation for the line, which contains two points:
class Line
{
  embed orig : Point;
  embed dest : Point;
}
```

### Methods

Methods are functions that belong to a class or interface which can have arguments and a return type.

Methods can be called statically or dynamically.
Static calls are either directly (like `Class::method()`) or via an object (like `obj.method()`) if the compiler can infer the concrete type.
Dynamic calls are either direct (via `Interface::method(object)`) or via an object (like `obj.method()`) otherwise.
Methods that change the fields in the class must receive a `self : Self` argument and use as the field accessor (ex. `self.field`).

In the following example, all calls are statically dispatched:
```ts
interface IFace
{
  method2(self : Self);
}

class Foo
{
  method1() { ... }
  method2(self : Self) { ... }
}

...

// Class syntax, static call
Foo::method1();

// Interface syntax, static call (because obj has the known concrete type Foo
let obj = Foo;
IFace::method2(obj);

// Object syntax, static call
var x : Foo = ...;
x.method1(); // Calls Foo::method1()
x.method2(); // Calls Foo::method2(x)
```

In the next example, interfaces may stop the compiler from finding the concrete type:
```ts
interface Bar
{
  method1() { ... }
  method2(self : Self) { ... }
}

...

// Object syntax, dynamic call
var y : Bar = ...;
y.method1(); // Could be Foo, but not sure, so use `y`'s vtable to call `Foo::method1()`
y.method2(); // Use `y`'s vtable to call `Foo::method2(y)`
```

The compiler will make sure that all classes that implement an interface will have consistent vtable layout so that `Interface::method()` doesn't end up with multiple levels of indirection.

### Constructors

Verona's built-in `new` function fully constructs an object and so must be called with values for all of the fields in the object.
It may be used only from static methods on the class and so is usually used by factory methods.
By convention, factory methods are called `create()` and there is syntactic sugar to allow `T()` to implicitly call `T::create()`.
However, there is no requirement that a constructor be called `create`, they can be called anything as long as they're static methods that return an object of the class' type.

Example:
```ts
class Capibara
{
  let name : String & imm;
  var weight : F32; // Kgs

  // The sugar constructor
  create() : Capibara
  {
    var self : Capibara = new Capibara("Ash", 35); // Healthy weight for an adult capibara
    return self;
  }

  // Create a baby capibara
  newborn(name : String) : Capibara
  {
    var self : Capibara = new Capibara(name, 3); // Tiny baby!
    self; // Returns `self`
  }

  eat(self: Capibara, food : F32)
  {
    self.weight += food;
  }

  poop(self: Capibara)
  {
    self.weight -= 0.1; // Tiny poop!
  }
};
```

On the example above, the `newborn` method creates a `Capibara` with a specific name and a default weight.
The `name` can't change, so we declare it with `let`, while the `weight` does change, so it's a `var`.

There's nothing special about the method `newborn` in comparison with others, apart from the fact that it allocates, initialises and returns an object of the same type.

## Destructors

Destructors are still under debate, but they won't have to deallocate memory like in C++, just execute finalisation tasks like closing file handles, running some cleanup routines, checking final consistency, etc.

## Primitive Types

Verona treats numeric types specially in the compiler but their representation isn't special in the language.

Verona's numeric types are singleton types (no fields) that have their machine representation based on their names.
So, `I32` is a signed integer with 32-bit width (in LLVM's parlance, an `i32`) and so on.

The full list of primitive numeric types are:
 * Unsigned integers: `U8`, `U16`, `U32`, `U64` and `U128`
 * Signed integers: `I8`, `I16`, `I32`, `I64` and `I128`
 * Floating Point: `F32` and `F64`
 * System types: `ULength` and `ILength` (size large enough to describe any offset into the largest array that it's possible to allocate)

Note, integer literals have no defined type and their types (and representation) will be inferred by the compiler depending on the type of the variable being assigned to.

For example:
```ts
// In this case, `42` has type `I32`
let x : I32 = 42;

// In this case, `42` has type `U64`
let x : U64 = 42;

// These are all compiler errors
let x = 42; // No type to infer
let x : I32 = 3.1415; // Type casts not allowed

```

## Type Unions

Type unions means a variable can be any type in the list.

For example:
```ts
foobar(a : (A | B | C))
{
  ...
}

main()
{
  let a : A;
  let b : B;
  let d : D;

  // This is fine
  foobar(a);
  foobar(b);

  // This is not
  foobar(d); // ERROR
}
```

The implementation of `foobar` will have to cope with all three types, either via a `match` statement or taking advantage of some behaviour that is common to all three (say, an interface they all implement).

The compiler is free, however, to optimise the calls by creating two versions of `foobar` and replace the calls to the generic `foobar` for their specific ones.
If the semantics are defined at compile time, this can simplifies a `match` statement into their specific matches and even inline the call altogether.

Example:
```ts
// The generic one, if there are still dynamic cases
foobar(a : (A | B | C))
{
  match (a)
  {
    { A => /* code for A */ }
    { B => /* code for B */ }
    { C => /* code for C */ }
  }
}

// The specific specialisations used in defined cases
foobar(a : A)
{
  // code for A
}
foobar(b : B)
{
  // code for B
}
```

## Type Intersections

Intersection means a type must conform to all types involved.

For example:
```ts
interface Serialisable { toString() : String; };

main()
{
  let x : (A & Serialisable & mut) = ...;
}
```

In the above example, the type of `x` must conform to both `A` and `Serialisable` guarantees (methods, fields) as well as the `mutable` capability.

References to objects need capabilities to be useful (read/write to fields, call methods), but references without capabilities can still be used for pattern matching objects at run time.

Type aliases can be used to make it easier to use types, for example:
```ts
class String { ... };

// We mostly want strings as values
type StringImm = (InternalString & imm);

// But we may want to change it
type StringMut = (InternalString & mut);
```

## Tuples

Tuples allow unnamed types to encompass multiple objects of different types.
Unlike classes, tuples don't have named fields.

Here's a tuple example:
```ts
// Declare a tuple type
var x : (String, U64);

// Tuple literals
x = ("hi", 42); // Type inference looks at `x`'s type for literals

// Deconstruct and update x, with s and i holding the old values
(var s, var i) = x = ("bye", 21);

// Deconstruct, consuming values and make x an invalid reference
(var s, var i) = consume(x);

// Deconstructs, x may be invalid or not after this
(var s, var i) = x;

// Possible syntax to access tuple elements by reference
x._1 = "ciao";
print(x._1);
```

The semantics of reading and writing to tuples is still under discussion.

## Self type

The `Self` type is used on interfaces to refer to the concrete type that implements the interface.

When concrete types implement an interface, they must implement all methods that the interface provides.
However, the interface type can be implemented by any number of concrete classes, and so cannot know what would be the concrete type implementing it.
That's where `Self` types come in: they're the concrete type implementing the interface.

Example:
```ts
interface EqualComparable
{
  ==(Self & readonly, Self & readonly): Bool;
}

interface Addable
{
  +(Self & readonly, Self & readonly): Self & imm;
}

class MyClass: EqualComparable, Addable
{
  // Some internal field
  field: U32 & mut;

  // Implements the EqualComparable interface
  ==(lhs: MyClass & readonly, rhs: MyClass & readonly): Bool
  {
    lhs.field == rhs.field
  }

  // Implements the Addable interface
  +lhs: MyClass & readonly, rhs: MyClass & readonly): MyClass & imm
  {
    lhs.field + rhs.field
  }
}
```

In the example above, the `MyClass` argument type _matches_ the `Self` type in the interface declaration because it is the same type as the class declaring it.

The interfaces above can also be implemented with [F-bound polymorphism](https://en.wikipedia.org/wiki/Bounded_quantification):

```ts
interface EqualComparable[A]
{
  ==(A, A) : Bool;
}

interface Addable[A]
{
  +(A, A): A;
}
```

In the bounded case, `A` is a type parameter that concrete classes implement by passing their own types, like `MyClass & readonly`.

Trying to use interfaces to implement a similar concept would incur in various problems in the type system and push some verifications from compile-time check to run-time checks.
Even if no exceptions are ever thrown, having to check them at run time every time would slow down programs and inhibit other compiler optimisations.

Below is an example of the pitfals encountered if we didn't use `Self` types.

If instead we had to use the interface's name, each argument (and return value) in the methods `==` and `+` would match _any_ types that implement the interfaces.

All problems are identified in comments, and the code below will not compile.

Example:
```ts
// This is NOT what we want!
interface Addable
{
  // Each argument and return value can be of a different Addable type!
  +(Addable & readonly, Addable & readonly) : Addable & imm;
}

class Foo : Addable
{
  // Implements interface, but operates on any Addable, which isn't what we want
  +(lhs: Addable & readonly, rhs: Addable & readonly) : Addable & imm
  {
    // You can't match against all possible Addable types (open world)
    match (lhs)
    {
      { Foo =>
        // We need to match both lhs and rhs
        // This is a horrible practice!
        match(rhs)
        {
          // This is where the logic goes, and gets lost in boilerplate
          { Foo => ... }
          // This becomes a run-time error, which is not what we want either
          { _ => Throw... }
        }
      }
      // This becomes a run-time error, which is not what we want either
      { _ => Throw... }
    }
  }

  // Doesn't implement the interface, but operates on Foo only, which is what we wanted
  +(lhs: Foo & readonly, rhs: Foo & readonly) : Foo & imm { ... } // ERROR: Redefinition of `+`
}
class Bar : Addable
{
  // Similar implementation as Foo here
  ...
}
class Baz : Addable
{
  // Similar implementation as Foo here
  ...
}

function() {
  var foo = Foo;
  var bar = Bar;
  // Type error if implemented as +(Foo, Foo), as it doesn't implement the interface.
  var addable : Addable = foo;
  // This matches the interface, but won't do what you want and will throw a run-time error
  var baz : Baz & imm = foo + bar; // ERROR: Doesn't catch throw
}
```

As demonstrated, `Self` is not just a convenience type, but a requirement to have the type checks occur at compile time and produce correct code without run time errors.