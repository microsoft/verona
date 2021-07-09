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

With the introduction of Unicode standards, characters are no longer single-bytes.
However, the size of a character can be different depending on which Unicode standard you use (ex. `UTF-8` vs. `UTF-16`).
But each Unicode standard has the appropriate byte-sized _escape sequence_ to represent a character with multiple bytes, so literals can still be a string of bytes.

It's still under discussion if we want character literals to natively interact with numeric types.
For example:

```ts
// 'A' can be seen as 0x41, which is often OK
let a : U8 & imm = 'A';

// But 'ABCD' is now a bit pattern (0x41424344)
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

// Or a whole UTF-16 string from an UTF-16 literal
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

// An array of integers. Which type? Depends on who uses `a`!
let b = Array(1, 2, 3);

// An array of (integer | String | float), again, depends on use
// This is not the same as a tuple: every argument can be any of the possible types.
let c = Array(1, "2", 3.0);
```

## Variables

Variables are placeholders for a reference to an object, to which they are bound.
The compiler optimises the representation of certain variables (ex. numeric types) but the syntax is preserved as if they are pointers.

### Let vs. Var

A variable declared with `let` can only be assigned once.
A variable declared with `var` can be reassigned multiple times.

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

All types in Verona are classes, so all variables in Verona store pointer objects.
As an optimisation, small immutable objects may be stored inside the pointer representation, avoiding the need for heap allocation.
This is the default case for any simple numeric types and may be extended to others in the future.
Refer to the internal ABI document for more details on object representation, including singleton types.

### Stack vs. Heap

The lifetime of variables is either automatic, when allocated on the stack, or determined by their regions' semantics, when allocated on the heap.

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

The region is responsible for managing that memory. The compiler guarantees that pointer are always valid.
Memory deallocation isn't necessarily done at the same time as invalidation of pointers (ex. garbage collected regions), but the compiler also guarantees that no code will follow pointers after becoming invalid.

Objects that can be captured (lambda, behaviour, function calls) need to go on the heap.
The deallocation of those objects will depend on which type of region and their policies.

Stack variables, however, have automatic lifetime given by the duration of their lexical scopes.
For that reason, stack variables are captured (ie. moved) when used by an asynchronous behaviour (`when`).

Example:
```ts
foo()
{
  // Declares a function-level variable
  var a : U64 & mut = 42;

  // Start a new lexical scope
  {
    // Assume MyType::create() specifies the region (which could be the stack)
    var b : MyType & iso = MyType;

    // This only schedules the work, not executes it
    when () {
      // `b` is captured by the behaviour and is invalid afterwards.
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

All operators are calls on a descriptor: either an object or a type declaration that indicates which type the method is to be found.
For example the type name (`Foo::method()`), the first argument of a pre-fix operator (`foo.method()` or `method(foo)`), or the left hand side of an infix operator (`foo method bar`).

Example:
```ts
// Defines the type of `x`
let x : U32 & imm = 42;

// Infers the type of `y` and `10` from the type of `x`
// Checks that `U32::+(U32, U32)` exists
// Calls U32::+(x, 10)
let y = x + 10;

// The above is identical to ones below
let a = x.+(10);
let b = U32::+(x, 10);
```

The Verona standard library will implement all operations for all appropriate types, including all arithmetic for all numeric types and string handling for all character and string types.
Users can implement similar calls to specialised type (ex. `MyInt::+(MyInt, I32)`).

There is no precedence in Verona, so operators must be wrapped in parenthesis to convey the right meaning.
This means that, unlike other languages, `(a + b * c)` is a syntax error, while either `((a + b) * c)` or `(a + (b * c))` are expected.

_Note: Can we avoid forcing parenthesis on DSLs?_

## Arithmetic

As detailed above, arithmetic in Verona is implemented as function calls on numeric types and operands.
This means developers can extend the functionality of arithmetic to the existing types or their own new types naturally.
But it also means that every addition or subtraction could end up as call on objects, which could potentially be many orders of magnitude slower than standard hardware operations.

But Verona's arithmetic is often as fast as calling the right hardware instructions.

The reasons are:
1. As explained in the types document, numeric types are singleton classes, ie. they don't have fields, only methods.
   The compiler treats them differently and uses their machine representation instead of a pointer to an allocated memory.
2. Computation often uses concrete types (ex. `U32` or `F64`) instead of interface types (ex. `IntegralType`) or type unions (ex. `(U32 | F64`)).
   The compiler can only match concrete types to their machine equivalent.
   It can try to infer, match and separate concrete types from their declared collection types, but that's an optimisation, not a guarantee.
3. Direct code generation of the hardware instructions for arithmetic functions.
   Those functions are often short and simple and are always inlined by the compiler.

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
It does however, implement them using the core Verona control flow structures and make available to programmers, so that programming in Verona doesn't become a burden.

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
The `when` block returns immediately, but the execution can be done at any time later.

To control which `when` block can work on which region, the argument is a list of regions (which can be empty for immediate dispatch), that the block needs to access.
The scheduler controls which behaviours will happen first (in causal order) and which will have to wait for others to finish (if they require regions that other behaviours are still working on).

So, instead of a `fork/join` model, `when` has a dependency model which the runtime uses to decide which behaviour to execute first, or in parallel.

### Emulated semantics

Despite not having `if` and `while` as native statements, Verona uses the other control structures to implement them, arriving at the usual semantics.

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
For non-concrete cases, their implementation needs to be complete for closed-world cases (ex. account for all types in a union) but not for open-world cases (ex. interfaces), as long as they only use features that the interfaces provides.

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
For concrete class types, the call will be static (direct call to the function), and for abstract interface or union types, a dynamic dispatch will be performed to the object's virtual table.

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

## Equality

Verona has two types of equality: by address and structural.

### Address equality

Address equality, compared with the keyword `is`, compares the address of the expressions, ie. their locations.
Two variables can only have equal addresses if they are aliases to each other.

Address equality implies type equality.
Variables of non-compatible types are never equal.

Example:
```ts
// Some variable
var a : U32 & mut = 42;
// An alias to the same address
let b = a;
// A new variable with the same value (but different address)
var c : U32 & mut = 42;

let eq = (a is b); // TRUE, `b` is an alias to `a`
let ne = (a is c); // FALSE, `c` is a new variable

// Types can be alias to sub-types
interface Foo {}
class Bar : Foo {}

let d : Bar & mut = Foo;
let e : Foo & mut = d;

let eq = (d is e); // TRUE, `e` is an alias to `d` of a compatible type

// Singleton types are always equal to themselves but not other singleton types
class None {}
class All {}

let always_eq = (None is None); // TRUE
let never_eq = (None is All); // FALSE
```

Singleton types (other than numeric types) have no machine representation, so all variables point to the same object and
they always compare equal because they are aliases to each other.
Different singleton types are different objects, therefore variables pointing to different singleton types always compare unequal.

Note that, even though literals have a machine representation for their values, the addresses that they are stored for `a` and `c` are different, and therefore they are not the _same variable_, even though they have the same value.

### Structural equality

Structural equality, compared with the method `==` on appropriate types, is a comparison of the values in the object that the variable points to, including numeric literal values.
It is the responsibility of each class to implement their own structural equality method.

User objects can be compared structurally, simply by implementing the `==` method.
Similar operators (`!=`, `<`, `>`, etc) are also implemented in the standard library and can be used by any user class in the same way.

Example:
```ts
// Some variable
var a : U32 & imm = 42;
// A new variable with different value
let b = a + 1;
// A new variable with the same value (but different address)
var c : U32 & imm = 42;

let ne = (a == b); // FALSE, 42 != 43
let eq = (a == c); // TRUE, both as 42
```

_Note: There is nothing special about structural comparison methods.
 They're methods like any other that just happens to "implement" comparisons.
 You can do whatever you want with your own types for `==`._

## Syntax Sugar

In Verona, every operation is a method call, including lambda execution, object creation, destruction, etc.
To create syntactically correct code, one must be explicit about everything.

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

## Partial Application

Partial application refers to the process of fixing some arguments of a call and returning a lambda with the remaining arguments to be passed to the lambda.
As most of Verona's high-level functionality, it will be implemented in the Verona standard library as a method called `~`, using type lists for the arguments.

TODO: Add an example here.
