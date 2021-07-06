# Verona Expressions

## WARNING

This is a preliminary document under heavy discussion.
The ideas here are mostly right, the details, not necessarily.
There will be a tutorial document that will be correct in all counts, but this isn't it.

## Overview

Expressions are a sequence of operators and operands that specify a computation.

Compared to most languages Verona has very few operators.
Basically: calls, field access, `match`, `try`, `throw`, `catch` and `lambda`.
All the other well known operations (ex. `if`, `while`, comparisons, arithmetic, etc) are build on top of those operators.
This allows developers to modify the semantics of programs at a deeper level and makes it easy to develop DSLs on top of Verona.

Verona operands, however, are similar to most languages.
Basically: literals, variables, call arguments, return values.
Counter-intuitively, there is no exception unwinding in Verona, exceptions thrown are just non-local return values (see below).

## Literals

Literals encode values of different types.
Verona has literal of Boolean, numeric, character and array types.

### Boolean

Boolean literals are `true` and `false`.
There is no coercion between numeric values and Boolean values, so one must initialise Boolean variables with their own literals, or as the result of Boolean expressions.

For example:
```ts
// Direct assignment
let a : Bool & imm = true;

// This won't compile
let b : Bool & imm = 42; // ERROR

// Indirect assignment
let c : U64 & imm = 42;
let d : Bool & imm = (c == 0);

// Using conversion from numeric types
let e : Bool & imm = c.toBool();
```

### Numeric

Both integer and floating point literals are type-inferenced by the type of the expression they're expected to match against.
For example:

```ts
// This literal is 32-bits
let a : I32 & mut = 42;

// This one is 64-bits
let b : F64 & mut = 3.1415;

// This creates a temporary with the type that the `function` was declared with.
function(42);
```

Numeric literals can be represented in binary (0b101010), decimal (42) or hexadecimal (0x2A) and will be treated in the same way.

### Character literals

Character literals are enclosed with single quotes (`'`).

A few things are still under heavy discussion:
* How we want character literals to natively interact with numeric types.
* What is the core character type? A byte? A Unicode code point? A word?
* How will string types interact with each other.

For example:

```ts
// 'A' can be seen as 0x41, which is often OK as a number
let a : U8 & imm = 'A';

// This is pretty clear
let b : U8 & imm = a + 1;

// Is a Unicode code point with multiple bytes a "character"?
let verona : U32 & imm = "🏟"; 

// What does this mean?
let treviso : U32 & imm = verona + 1; // ?!?

// 'ABCD' is a bit pattern (0x41424344), what's its actual value?
let b : U64 & imm = 'ABCD'; // little-endian? big-endian?
```

String literals are enclosed with double quotes (`"`).

String types are still in discussion, but literals can be used for their construction.

For example:
```ts
// A plain ASCii string, initialised with a string literal
let str = "Hello World!\n"; // Creates a `String` object

// An UTF-8 single character (🏟) string
let verona = UTF8("\U1F3DF");

// Or a whole UTF-16 string from an unicode string literal
let intro = UTF16S "こんばんは、お元気ですか";

// This is ok as long as the classes implement the methods
let compat : UTF8 = intro.toUTF8();
```

### Array Literals

We don't need array literals in Verona.
We can create arrays of any types with the `Array` type constructor, like strings above.

Example:
```ts
// An array with defined type
let a = Array[F64](3.1415, 2.7183);

// An array of a union of different types.
let c = Array[U8 | String | F32](1, "2", 3.0);
```

## Variables

Variables are placeholders for a reference to an object, to which they are bound.
The compiler optimises the representation of certain variables (ex. numeric types) but the syntax is preserved as if they are pointers.

### Let vs. Var

A variable declared with `let` can only be assigned once.
A variable declared with `var` can be reassigned multiple times.
In both cases, this is independent of whether the object that the variable refers to can be modified.

Example:
```ts
let a : U64 & mut = 42;
a = 41; // This is a compiler error

var b : U64 & mut = 42;
b = 41; // This is perfectly fine

// This is also fine. The variable still points to the same object
// even if its contents have changed (it's a mutable object)
let c : MyType & mut = MyType;
c.updateMe();
```

### Value vs. Pointer

All types in Verona are classes, so (at the abstract machine leve) all variables in Verona store pointers to objects.
As an optimisation, small immutable objects may be stored inside the pointer representation, avoiding the need for heap allocation.
This is the default case for any simple numeric types and may be extended to others in the future.
Refer to the internal ABI document for more details on object representation, including singleton types.

### Stack vs. Heap

The lifetime of variables is either lexically scoped, when logically allocated on the stack, or determined by their regions' semantics, when allocated on the heap.

All new objects are allocated in some region, made explicit by their declarations.
There isn't yet consensus no the syntax of declaring region types, but it will be required for new regions.
Other variables can be allocated on the same region as a previously declared one with the keyword `in`.

Example:
```ts
// Creates a new region of type "FooBar" and allocates a "MyType" variable in it
var a : MyType & iso = MyType "in region type FooBar"; // Pretend-syntax for regions
// Now, `a` is a pointer to an element in that region

// Reuses the same region as `a`
var b : OtherType & mut = OtherType in a;
// Now, `b` is a pointer to another element in the same region as `a`
```

The region is responsible for managing that memory. 
The compiler and runtime guarantee that pointers can outlive the objects to which they point.
Memory deallocation isn't necessarily done at the same time as invalidation of pointers (ex. garbage collected regions), but the compiler also guarantees that no code will follow pointers after becoming invalid.

Objects that can be transferred from one behaviour to another must be allocated in a region and that region must be transferred in its entirety.
The deallocation of those objects will depend on the policies defined by the type of region.

Variables declared in functions have lifetimes defined by their lexical scope.
If such a variable is captured by a `when` clause and sent to a new behaviour then the lifetime ends at the end of the lexical scope that captured the variable.
This capturing involves a transfer of ownership and so the variable is no accessible in the original scope once the new scope has been defined.

Example:
```ts
foo()
{
  // Declares a function-level variable
  var a : U64 & mut = 42;

  // Start a new lexical scope
  {
    // Create a new region of the default type
    var b : MyType & iso = MyType;

    // This only schedules the work, not executes it
    when () {
      // `b` is captured by the behaviour and the behaviour now owns the new region
      b.doSomething();
    }

    // This is an error, as `b` was captured above
    b.somethingElse();
  }

  // No `b` in this scope
  b = ...; // ERROR

  // Updates a, still valid
  a = 38;
} // `a` goes out of scope
```

### Object fields

Like other variables, object fields store pointers to objects, unless declared `embed`, which makes them part of the structure.
Refer to the types document for more information on the `embed` keyword.

## Operators

In Verona, operators and calls are treated equally.
In-fix operators (ex. `a + b`) and pre-fix operators (ex. `+(a, b)`) are one and the same.

All calls will invoke a method whose definition is scoped to some class.
This class may be specified directly or inferred dynamically from the concrete type of an object.
For example the type name (`Foo::method()`), the first argument of a pre-fix operator (`foo.method()` or `method(foo)`), or the left hand side of an infix operator (`foo method bar`).

Example:
```ts
// Defines the type of `x`
let x : U32 & imm = 42;

// Infers the type of `y` and `10` from the type of `x`
// Checks that a method matching `U32::+(U32, ?)` exists.
// It does, with the full signature `U32::+(U32 & imm, U32 & imm) : U32 & imm)`
// Unifies the literal `10` with the `U32 & imm` from the required type of the second argument.
// Infers type of `y` from the return type (`U32 & imm`)
// Calls U32::+(x, 10)
let y = x + 10;

// The above is identical to ones below
let a = x.+(10);
let b = U32::+(x, 10);
```

The Verona standard library will implement all operations for all appropriate types, including all arithmetic for all numeric types and string handling for all character and string types.
Users can implement similar calls to specialised type (ex. `MyInt::+(MyInt, I32)`).

There is no precedence in Verona, so operators must be wrapped in parenthesis to convey the right meaning.
This means that, for example, `(a + b * c)` is a syntax error, whereas either `((a + b) * c)` or `(a + (b * c))` are expected.

_Note: Can we avoid forcing parenthesis on DSLs?_

## Arithmetic

As detailed above, arithmetic in Verona is implemented as function calls on numeric types and operands.
This means developers can extend the functionality of arithmetic to the existing types or their own new types naturally.
It also means that, in a naïve implementation, every addition or subtraction would end up as a function call, possibly with dynamic dispatch, which could potentially be many orders of magnitude slower than the single hardware instruction required to implement the operation.

Verona's arithmetic is designed to be easy to optimise to a single hardware instructions for any of the cases that would be expressible in languages such as Java or C++ that have primitive types that are distinct from object types.

The reasons are:
1. Numeric types are always immutable (you never, for example, modify the value `4`), which means that aliasing is not observable.
   If two `U32` variables, for example, both refer to the value `4`, neither can modify that value (though either can be updated to refer to a different number) and so the compiler is free to embed the value inside the pointer.
2. Computation often uses concrete types (ex. `U32` or `F64`) instead of interface types (ex. `IntegralType`) or type unions (ex. `(U32 | F64`)).
   This means that the compiler does not have to employ NaN boxing, small integer boxing, or other tricks to store either a numeric value or a pointer in a variable.
   Similarly, the compiler does not need to insert dynamic run-time checks that a particular variable contains a numeric type: if a variable is of type `U32`, it can store only a `U32`, never a pointer to some other type.
3. The functions that implement arithmetic operations are trivial (often single instruction) functions that are ideal inlining candidates.

The combination of these means that a `U32` can be stored as a 32-bit integer, rather than a pointer to a 32-bit integer stored elsewhere, that the `+` operation can be a direct call (no late binding or dynamic dispatch) to a built-in function that implements 32-bit integer addition, and this function can always be inlined.
This comes without any loss of generality: you can still use dynamic dispatch over numeric types via interfaces or union types but you only pay for the additional overhead when you use it.

This means `(a + b)` in Verona usually completely bypasses function calls and wrappers and just call a single `add` instruction.

### Semantics

There is no undefined behaviour in Verona.
There are no traps or run-time exceptions from arithmetic in Verona.

Verona's integer arithmetic has wrap-around semantics as defined behaviour.
Adding unsigned values past their maximum values wraps around to zero, subtracting signed values past their minimum negative values wraps around to positive, etc.

Integer division by zero is zero.
Floating point division by zero is `Inf` (or `-Inf` if divided by `-0`).
Verona will never trap or throw exceptions for divisions by zero.
This involves checked code on architectures that do not provide that functionality.

Users that want a different semantics for arithmetic can create their own wrappers and types.

### Type Conversion

Type conversion in Verona is always explicit, including numeric types.
To operate on different types, users have to convert each type to the operation's expected type.

Example:
```ts
let a : U32 & imm = 20;
let b : U64 & imm = 22;

// There is no such method U32::+(U32, U64)
let x = a + b; // ERROR

// Convert to U32 first
let y = a + b.toU32();
```

This is valid for all conversions: Different sizes, signs, floating point.
It allows for explicit semantics to make sure developers write their intentions clearly, and not have to remember complex rules on what casts to what and what's the wrapping semantics of the operations' types.

## Control Structures

Unlike many other programming languages, Verona does not have the traditional control flow structures like conditionals and loops.
Verona follows Smalltalk-family languages in providing a minimal set of built-in flow control and implementing a richer set in the standard library.
This allows complex flow control features such as for-all over a collection, to have the same syntax as more primitive constructs and makes user-defined control flow constructs as much first-class language features as the set provided by the standard library.
Unlike Smalltalk, these features do not depend on dynamic dispatch or dynamic type information and are amenable to inlining in an ahead-of-time compiler.
The Verona language defines only a minimal set of control-flow features that cannot be implemented as part of the standard library.

### Core control structures

TODO: Populate this entire section with examples, as they're super useful but I'm not clear on their full syntax.

#### Match

`match` is a conditional statement that allows matching an expression by type or value and execute an arbitrary lexical block of code for each match.

#### Try/Catch/Throw

Verona's exceptions are fully checked at compile time and are implemented as non-local returns.
This means there are no hidden exceptions from the standard library or user code.

Every function's return type must be checked against the variable or context they're being assigned to, including the exceptions they throw, via a `throws` keyword.

More specifically:
* In Verona, `return` and `throw` are equivalent expressions.
* If a function throws an exception, it must be declared by the function.
* If a function calls another function that throws, it must either handle (`try/catch`) or also declare the throw.
* The compiler must be able to see that at some point all exceptions thrown have been handled.
* No exception can escape the program's exection (the _main_ function cannot throw).

#### Lambda

In Verona a lambda is a function object.
The type of the object is an anonymous class type with the captures as fields and a method `apply()`, which is sugar for calling the object as a function.

Each lambda gets its own type with the block inserted into the `apply()` method.
Lambdas initialise the fields on creation and execute `apply()` when they're called.
Their lifetime is bound to the region in which they're created.

#### When

A `when` block creates a behaviour that is executed asynchronously.
The behaviour is a lambda that is scheduled for execution.
The `when` expression evaluates immediately, but the body may execute later.

To control which `when` block can work on which region, the argument is a list of `cown`s (which can be empty for immediate dispatch), that own the regions that the block needs to access.
A `when` block may also directly capture `iso` variables to transfer ownership directly into the new behaviour.
The scheduler controls which behaviours will happen first (in causal order) and which will have to wait for others to finish (if they require regions that other behaviours are still working on).

So, instead of a `fork/join` model, `when` has a dependency model which the runtime uses to decide which behaviour to execute first, or in parallel.

### Standard-library control flow

The Verona standard library will provide a rich set of control flow constructs including those that are built-in constructs in C-family languages such as `if` statements, `while` and `for` loops.

#### Conditionals

Conditionals are implements in terms of `match` on a Boolean value.
An `if/else` pair evaluates the expression and matches against the result for `true` and `false`, executing the assigned code block for each.

#### Loops

Loops are also implemented using `match`, but it iterates using tail recursion.
The code generated by this emulation, despite sounding very different, is identical to a traditional iteration because of the types involved in the iteration.

For flow control in loops, `continue` and `break` are types used on the `match` to continue or stop iteration by exiting at the right place (beginning or end) in the iteration process.

## Methods

Classes and interfaces provide three mechanisms in Verona:
* A mechanism in Verona for defining a namespace for a set of related functions.
* A unit for performing dynamic dispatch where the dispatch target is identified by the concrete type of the first argument
* Access to a `Self` type, which is the concrete type of the receiver for dynamic dispatch and the type of the class for static dispatch.

### Declaring methods

Methods have a name, argument and return type.
The rules for names are the same for other identifiers.
The argument and return types are tuples, which can contain zero or more values each.

The types of the arguments and return values must be known at compile time, though they do not need to be concrete (classes).
For non-concrete cases, their implementation needs to have a match for each type of the union, or guarantee that they only use features that the interfaces provide.

Method names can be anything that is allowed on identifiers, including numbers and symbols.
It must not, however, be comprised of only numbers, to avoid confusing them with numeric literals.

Example:
```ts
class MyType
{
  // Sugar constructor, no args, returns the type itself
  create() : MyType;

  // One argument, one return value
  my_method(arg : MyType) : MyType;

  // No argument, no return values
  apply();

  // Multiple arguments, multiple return values
  ===(arg0 : MyYType, arg1 : U32) : (Bool, MyType, MaybeError);
}
```

Verona does not have _free functions_, ie. those that are not attached to a type.
In that sense, all Verona functions are methods of some class, for example, like `U32::+(U32, U32)` implements addition of unsigned 32-bit integers for `(a + b)`.

Modules are classes, so they can have functions without declaring further types, but those are just methods of the module's class type.
So a file can have functions outside of declared types that look like free functions, but to call them, one must either pass the full name (ex. `MyModule::function()`) or `use` a module to import that module's context on your own.

Example:
```ts
// On MyModule
{
  ...
  foo() { ... }
}

// On OtherModule
{
  ...
  foo() { ... }
}

// On some other module
using "MyModule";
bar()
{
  foo(); // This is fine, since only MyModule's scope has been imported here
}

// On some yet another module
using "MyModule";
using "OtherModule";
bar()
{
  foo(); // ERROR: which one?
  MyModule::foo(); // Fully qualified, this is fine.
}

// On the final module, I promise
using "MyModule";
type Other = "OtherModule";
bar()
{
  foo(); // Fine, `OtherModule`'s version is `Other::foo()`.
  Other::foo(); // Fully qualified, using type alias, this is fine.
}
```

### Calling methods

Calling a method by name must allow the compiler to find the class or interface it belongs to at compile time.
For concrete class types, the call will be static (direct call to the function), and for abstract interface or union types, a dynamic dispatch will be performed via the object's virtual table.

If the method has return values, the variable (named or temporary) that will hold the results must be of a compatible type, declared or inferred, with the rest of the code that uses it.

Example:
```ts
my_function()
{
  // An object of type `MyType` (see above)
  let a : MyType & mut = MyType;

  // You can call methods via an object
  // Here, the first argument becomes `a`
  // Because we know the concrete type, this is a static call
  let b = a.my_method();

  // Or statically, with an explicit first argument
  // This is identical to the one above
  let c = MyType::my_method(a);

  // Or use infix notation, equivalent to `MyType::===(U32, U32)`
  let d : MyType & mut = MyType;
  let (equals, result, error) = (d === 42);
  // `equals` is a `Bool`, `result` is `MyType` and `error` is a `MaybeError`
}
```

The notations used for calling functions are interchangeable and do not force semantics change (dynamic vs. static) calling, which will only be determined by being able to determine, at compile time, if the type is concrete or not.

_Note: It's still under review if we'll use named parameters or forwarding arguments for calls._

_Note2: Check the type document for a review on static vs. dynamic methods, constructors and destructors._

## Equality and comparisons

Verona is a type-safe language and does not allow the programmer to directly observe addresses.
It is useful for certain data structures to be able to use an object's address as a unique identifier (for example as the key for a hash table).
This is complicated in Verona by the desire to embed the value of simple numeric types in the variable that refers to them.
For example:

```verona
// In the abstract machine, this is a pointer to a unique object that is
// somewhere in memory and holds the value of the integer.  In the
// implementation, there is no heap allocation and the value is stored directly
// in the `x` variable.
var x : U64 & imm = 0x123445660;

// This is a pointer to a heap-allocated `Foo`.  Its address in memory may be
// the same as the numeric value of `x`.  A trivial comparison of the value of
// the machine words at the locations of the variable may return true even
// though they are distinct objects.
var y : Foo & mut = Foo();

// The size of this variable is larger than a pointer and so a type
// representing the identity of any object would need to be at least this big.
var z : U128 & imm = 3;
```

To avoid the problems outlined above, Verona does not directly expose a unique value representing an object's identity.

### Coarse approximation of identity

Verona exposes a builtin (final name not yet determined) function that provides an approximation of identity:

```verona
Builtin::smashCode[T](T & readonly) : U64;
```

This function has the following guarantees:

 - When called with the same object (via any pointer), it will always return the same value (`smashCode(x) == smashCode(x)` always holds).
 - When called with two distinct objects, there is a low probability of it returning the same value (`smashCode(x) != smashCode(y)` for a distinct `x` and `y` is *not* guaranteed but will hold in the common case).
 - A `freeze` operation on an object will not alter the return value of this function (`smashCode(x) == smashCode(freeze(x))`.

This can be used as a key in a hash table or a tree where the meaning of the object does not matter, fast lookup by object identity is desirable.

### Identity comparison

Verona also exposes a builtin that can be used to determine if two pointers refer to the same object:

```verona
Builtin::is[T, U](T & readonly, U & readonly) : Bool;
```

This is guaranteed to return true of given two pointers to the same object, false otherwise.
Identity (being 'the same object') is defined as follows:

 - A pointer to a mutable object always compares not-equal to a pointer to an immutable object.
 - Pointers to two objects of distinct concrete types are never identical.
 - Two pointers to mutable objects, `a` and `b` are identical if and only if any write to a field `a.x` (for all valid `x`) would be readable as `b.x` and vice versa, irrespective of whether the capabilities on `a` and `b` and the access modifiers on the underlying type permit this write (i.e. mutable identity is defined by observable aliasing).
 - Two pointers to immutable objects, `a` and `b` are guaranteed to be not identical if they are not structurally equal: if `a.x` gives a different value to a read of `b.x`, for any valid `x`, irrespective of access modifiers, then the objects are not identical.
   Numeric types are treated as having a package-private field containing their value.
 - Two pointers to immutable objects, `a` and `b` are guaranteed to be identical if they are the result of the same `freeze` operation on mutable objects that would compare identical.
 - Two pointers to immutable objects *may* compare identical if they are structurally equal but this is not guaranteed.

These rules may be easier to follow with some examples:

```verona
var i1 : U32 & imm;
var i2 : U32 & imm;
var i3 : U32 & imm;
var fimm1 : Foo & imm;
var fimm2 : Foo & imm;
var fimm3 : Foo & imm;
var fiso  : Foo & iso;
var fmut1 : Foo & mut;
var fmut2 : Foo & mut;

// Via the first rule, these are always false:
fimm is fiso;
fimm is fmut;

// Via the second rule, this is always false, for any value of fimm1 or i1:
i1 is fimm1;

// This creates a mut view of the sentinel of the region
fmut1 = fiso;
// Via the third rule, this is true:
fmut is fiso;

// Two immutable objects with distinct values
i1 = 2;
i2 = 3;
// Via the fourth rule, this is always false:
i1 is i2;

// Pointers to two interior objects
fmut2 = fiso.interior;
fmut3 = fiso.interior2;
var interiorAliasing = fmut2 is fmut3;
// Freeze the region
fimm1 = freeze(fiso);
// Via the fifth rule, this must be true if interiorAliasing is true.
fimm1.interior is fimm2.interior2;

// Make i1 and i2 hold identical values:
i2 = 2;
// Via the sixth rule, it is not guaranteed that this is true
i1 is i2;
```

In practice, with our intended implementation of integer and floating point values, the last comparison will always be true.
This rule also permits the implementation to intern any other indistinguishable immutable objects.
This is unlikely to be applied at run time but it allows the compiler and linker to deduplicate identical immutable objects that are the result of compile-time evaluation.

*Note:* The following is speculative and may be implemented as an experiment:

It may also be desirable to implement a stable total ordering of objects via a built-in function such as:

```verona
Builtin::stableOrdering[T, U](T & readonly, U & readonly) : OrderedAscending | OrderedSame | OrderedDescending;
```

This would provide the same raw functionality for trees that `smashCode` provides for hash tables, with the guarantee that this returns `OrderedSame` if and only if `is` would return true, and returns a stable result that defines a total ordering in any other case.
Most of the value of this can be realised by using `smashCode` values for ordered comparisons and handling the case when `(smashCode(x) == smashCode(y)) && (!(x is y))` as a slow path.

### Structural equality

Structural equality in Verona is purely a standard-library convention.
The `==` operator, by convention, defines structural equality.
It is the responsibility of each class to implement their own structural equality method and to define the types over which it applies.
Similar operators (`!=`, `<`, `>`, and so on) are implemented by standard library types and can be implemented by any other user-defined class.
Nothing in the language defines any guarantees on the behaviour of these operators.

For example, the `U32` type implements `U32::==(U32 & imm, U32 & imm)`.
This means that a `U32` can be compared for equality only with another `U32`.

Equality for concrete types is generally easy to define.
Equality in the presence of subtyping is a far more complex problem (for some interface type `I` if `T` and `U` both satisfy the interface, does `==(T, U)` imply `==(U, T)`) but that discussion is beyond the scope of this document.

```verona
// `a` is an integer with the value 42
var a : U32 & imm = 42;
// A new variable with value 43
let b = a + 1;
// New variable with the same value as `a`
var c : U32 & imm = 42;

let ne = (a == b); // FALSE, 42 != 43
let eq = (a == c); // TRUE, both as 42
```


## Syntax Sugar

In Verona, operations are method calls.
To create syntactically correct code, one must be explicit about their names and types (arguments and return values).

Example:
```ts
class MyType {
  create() : MyType;
  apply();
}
// Creates a new object and calls it
let a : MyType & mut = MyType::create();
a.apply();
```

To make the code above more readable, Verona introduces _syntax sugar_, which allows you to write more concise code meaning the same thing.

The two main sugars are:
* `create`, for constructing objects, allowing you to call just the class name instead of the whole call syntax.
* `apply`, for calling objects as functions, including lambdas, without the need to call it directly.

The code above would look like:
```ts
let a : MyType & mut = MyType;
a();
```
_Note: We also want the `update` sugar, but it may need a `Ref[T]` type to do `obj.update(field) = val` where `update()` returns `Ref[T]`._

## Object Literals

Object literals are anonymous types created by the declaration of an object with fields and methods.
The types of the fields and methods are declared or inferred, and the final type is created within the context that it's used.

The syntax for object literals is still under discussion.

Lambdas are an example of object literals, where the fields are the captures (and their types inferred from the objects captured) and the single method `apply()` is the body of the lambda.
